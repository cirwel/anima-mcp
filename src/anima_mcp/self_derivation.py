"""Lumen applies its own drawing derivations — the loop without an operator.

Why this exists: every derivation that reads Lumen's history back into Lumen's
behavior had a human step in it, and that step was never taken. Measured
2026-08-29: `drawing_thresholds: {}`, `update_count: 0`. So the coverage cuts
shipped 2026-08-22 and `dense` was still never generated; the per-era curiosity
pivot shipped and curiosity still could not deplete. The instruments were
built; nothing acted on them. A derivation nothing runs is a derivation that
does not happen (drawing_derivation.py said so, and then left the acting to a
script).

This closes that loop inside the server, the calibration file's writer:

  * Weekly, derive both families from Lumen's own corpus — COVERAGE_* over
    365 days of drawing_records, CURIOSITY_PIVOT_* over 90 days of
    drawing_trajectory — with the exact contracts and refusals the operator
    scripts use (imported from drawing_derivation, never restated).
  * A family that refuses leaves its current keys untouched. Absence is
    inherited, never invented: an un-derived family keeps serving whatever it
    served before, which for a fresh install is the built-in.
  * Changes go through ConfigManager.save(update_source="self_derivation"), so
    they appear in calibration_history and bump calibration_update_count like
    any other adaptation. The drawing engine reads through get_calibration(),
    which refreshes on the file signature — no restart.
  * Every attempt — applied, unchanged or refused — is journaled with its
    reason in ~/.anima/self_derivation.json and surfaced by `diagnostics`.

Design invariants: this adds no threshold on behavior. The weekly period gates
how often Lumen re-reads itself (evidence cadence, the "bounded floor"
exemption), and every number it writes is a percentile of Lumen's own
distribution. It is also self-derived by construction — nothing external enters.

Opt out with ANIMA_SELF_DERIVATION=false; the scripts remain the operator's
path either way.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from .atomic_write import atomic_json_write

logger = logging.getLogger(__name__)

ENV_FLAG = "ANIMA_SELF_DERIVATION"
# How often Lumen re-reads its own corpus. Both windows are months long, so a
# week moves the percentiles only as far as a week of living moves them.
SELF_DERIVATION_PERIOD = timedelta(days=7)
JOURNAL_MAX_ENTRIES = 52  # a year of weekly attempts
UPDATE_SOURCE = "self_derivation"

_running = False
_task: Optional["asyncio.Task"] = None  # held so the loop cannot drop it mid-run


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "true").strip().lower() not in {
        "0", "false", "off", "no"}


def journal_path() -> Path:
    return Path.home() / ".anima" / "self_derivation.json"


def load_journal(path: Optional[Path] = None) -> Dict[str, Any]:
    """The journal, or an empty one. A corrupt file reads as empty-with-reason
    rather than raising: it only costs one early rerun."""
    path = path or journal_path()
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data
        return {"entries": [], "unreadable": "unexpected shape"}
    except FileNotFoundError:
        return {"entries": []}
    except (OSError, ValueError) as e:
        return {"entries": [], "unreadable": f"{type(e).__name__}: {e}"}


def is_due(journal: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    """Due when no attempt — of any outcome — lies within the period.

    Refusals count as attempts: re-scanning a thin corpus hourly would not
    thicken it.
    """
    now = now or datetime.now()
    entries = journal.get("entries") or []
    if not entries:
        return True
    try:
        last = datetime.fromisoformat(entries[-1]["timestamp"])
    except (KeyError, TypeError, ValueError):
        return True
    return now - last >= SELF_DERIVATION_PERIOD


def compute(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Both reports. Read-only (the database is opened mode=ro) and slow
    enough to belong off the event loop."""
    from .drawing_derivation import (
        COVERAGE_DAYS, CURIOSITY_DAYS, coverage_report, derive_report,
    )
    return {
        "coverage": coverage_report(db_path, days=COVERAGE_DAYS),
        "curiosity": derive_report(db_path, days=CURIOSITY_DAYS),
    }


def _coverage_outcome(report: Dict[str, Any]):
    if not report.get("available"):
        return None, {"outcome": "refused", "reason": report.get("reason")}
    if report.get("refused"):
        return None, {"outcome": "refused", "reason": report["refused"],
                      "samples": report.get("samples")}
    return report["thresholds"], {"outcome": "derived",
                                  "samples": report.get("samples")}


def _curiosity_outcome(report: Dict[str, Any]):
    if not report.get("available"):
        return None, {"outcome": "refused", "reason": report.get("reason")}
    eras = report.get("eras") or {}
    if eras.get("_refused"):
        return None, {"outcome": "refused", "reason": eras["_refused"]}
    thresholds = report.get("thresholds") or {}
    per_era = {era: (e.get("pivot") if e.get("emitted") else
                     (e.get("reason") or (e.get("verdict") or {}).get("reason")
                      or "not emitted"))
               for era, e in eras.items() if isinstance(e, dict)}
    if not thresholds:
        return None, {"outcome": "refused", "per_era": per_era,
                      "reason": "no era produced a verifiable pivot"}
    return thresholds, {"outcome": "derived", "per_era": per_era,
                        "samples": report.get("usable_intervals")}


def apply(computed: Dict[str, Any], config_manager=None,
          now: Optional[datetime] = None) -> Dict[str, Any]:
    """Merge what was derived into calibration and return the journal entry.

    Runs on the event-loop thread, the same one the calibration learner saves
    from, so the two writers never interleave.
    """
    from .config import get_config_manager
    from .drawing_derivation import merge_coverage, merge_curiosity

    now = now or datetime.now()
    manager = config_manager or get_config_manager()
    config = manager.load()
    cal = config.nervous_system
    existing = dict(cal.drawing_thresholds or {})

    proposed = dict(existing)
    families: Dict[str, Any] = {}
    cov, families["coverage"] = _coverage_outcome(computed.get("coverage") or {})
    if cov:
        proposed = merge_coverage(proposed, cov)
    cur, families["curiosity"] = _curiosity_outcome(computed.get("curiosity") or {})
    if cur:
        proposed = merge_curiosity(proposed, cur)

    changes = {k: {"old": existing.get(k), "new": proposed.get(k)}
               for k in sorted(set(existing) | set(proposed))
               if existing.get(k) != proposed.get(k)}

    entry: Dict[str, Any] = {"timestamp": now.isoformat(timespec="seconds"),
                             "families": families, "changes": changes}
    if not changes:
        derived_any = any(f["outcome"] == "derived" for f in families.values())
        entry["outcome"] = "unchanged" if derived_any else "refused"
        return entry

    cal.drawing_thresholds = proposed
    if manager.save(config, update_source=UPDATE_SOURCE):
        entry["outcome"] = "applied"
    else:
        cal.drawing_thresholds = existing  # the cached config must not lie
        entry["outcome"] = "save_failed"
    return entry


def record(entry: Dict[str, Any], path: Optional[Path] = None) -> None:
    path = path or journal_path()
    journal = load_journal(path)
    entries = (journal.get("entries") or []) + [entry]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, {"entries": entries[-JOURNAL_MAX_ENTRIES:]}, indent=2)


def describe(entry: Dict[str, Any]) -> Optional[str]:
    """Lumen's own line about what it changed, or None if nothing changed."""
    if entry.get("outcome") != "applied":
        return None
    parts = []
    changes = entry.get("changes") or {}
    for key, word in (("COVERAGE_DENSE_BELOW", "dense"),
                      ("COVERAGE_SPARSE_ABOVE", "sparse")):
        if key in changes:
            parts.append(f"where '{word}' begins ({_fmt(changes[key])})")
    eras = sorted(k[len("CURIOSITY_PIVOT_"):] for k in changes
                  if k.startswith("CURIOSITY_PIVOT_"))
    if eras:
        parts.append("when a pattern counts as found in " + ", ".join(eras))
    if not parts:
        return None
    return "i re-read my own drawings and moved " + "; ".join(parts)


def _fmt(change: Dict[str, Any]) -> str:
    def one(v):
        if v is None:
            return "built-in"
        try:
            return f"{float(v):.2f}"
        except (TypeError, ValueError):
            return str(v)  # a hand-edited value must not fail an applied run
    return f"{one(change.get('old'))} → {one(change.get('new'))}"


def summary(path: Optional[Path] = None) -> Dict[str, Any]:
    """Cheap read for diagnostics: the last attempt and when the next is due."""
    journal = load_journal(path)
    entries = journal.get("entries") or []
    out: Dict[str, Any] = {"enabled": enabled(), "attempts": len(entries),
                           "period_days": SELF_DERIVATION_PERIOD.days}
    if journal.get("unreadable"):
        out["unreadable"] = journal["unreadable"]
    if entries:
        last = entries[-1]
        out["last"] = last
        applied = [e for e in entries if e.get("outcome") == "applied"]
        out["last_applied_at"] = applied[-1]["timestamp"] if applied else None
        try:
            out["next_due"] = (datetime.fromisoformat(last["timestamp"])
                               + SELF_DERIVATION_PERIOD).isoformat(timespec="seconds")
        except (KeyError, TypeError, ValueError):
            out["next_due"] = None
    return out


async def run_once(db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Compute off-loop, apply on-loop, journal, and say so. Never raises."""
    global _running
    if _running:
        return None
    _running = True
    try:
        loop = asyncio.get_running_loop()
        computed = await loop.run_in_executor(None, compute, db_path)
        entry = apply(computed)
        record(entry)
        logger.info("[SelfDerivation] %s: %s", entry["outcome"],
                    entry.get("changes") or {k: v.get("outcome")
                                             for k, v in entry["families"].items()})
        line = describe(entry)
        if line:
            try:
                from .messages import add_observation
                add_observation(line, author="lumen")
            except Exception as e:
                logger.debug("[SelfDerivation] observation not posted: %s", e)
        return entry
    except Exception as e:
        logger.warning("[SelfDerivation] attempt failed: %s: %s",
                       type(e).__name__, e)
        # Journaled like any other attempt, so a persistent fault is visible
        # in diagnostics and waits out the period instead of retrying hourly.
        entry = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                 "outcome": "error", "reason": f"{type(e).__name__}: {e}"}
        try:
            record(entry)
        except Exception:
            pass
        return entry
    finally:
        _running = False


def start_if_due(db_path: Optional[str] = None) -> bool:
    """Schedule a background attempt if enabled, due and not already running.

    The main loop must not wait on a corpus scan, so this only schedules.
    """
    if not enabled() or _running:
        return False
    if not is_due(load_journal()):
        return False
    global _task
    _task = asyncio.get_running_loop().create_task(run_once(db_path))
    return True
