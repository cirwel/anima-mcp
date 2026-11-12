"""Subsystem health monitoring for Lumen.

Tracks liveness (heartbeats) and functional health (probes) for each
subsystem. Queryable via MCP and rendered on the LCD health screen.

Usage:
    from .health import get_health_registry

    registry = get_health_registry()
    registry.register("growth", probe=lambda: _growth is not None)
    ...
    registry.heartbeat("growth")  # call each loop iteration
    ...
    status = registry.status()  # {"growth": {"status": "ok", ...}, ...}
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional


ProbeResult = bool | Dict[str, Any]


# How long before a missing heartbeat means "stale"
HEARTBEAT_STALE_SECONDS = 30.0

# How often to run functional probes
PROBE_INTERVAL_SECONDS = 60.0

OUTBOUND_HEARTBEAT_PROVISIONING_GRACE_SECONDS = 24 * 60 * 60


def _env_value(path: Path, key: str) -> str:
    """Read one simple dotenv assignment without evaluating shell code."""
    if not path.exists():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value.strip()
    return ""


def outbound_heartbeat_health() -> Dict[str, Any]:
    """Report whether the off-host dead-man's switch is provisioned.

    The URL is intentionally reduced to a boolean.  Detailed health is an
    observability surface and must never become a secret-disclosure surface.
    """
    home = Path.home()
    env_path = Path(os.environ.get("ANIMA_ENV_FILE", home / ".anima/anima.env"))
    marker = Path(os.environ.get(
        "ANIMA_HEARTBEAT_INERT_MARK", home / ".anima/.heartbeat-inert"
    ))
    grace = OUTBOUND_HEARTBEAT_PROVISIONING_GRACE_SECONDS

    try:
        configured = bool(
            os.environ.get("ANIMA_HEARTBEAT_URL", "").strip()
            or _env_value(env_path, "ANIMA_HEARTBEAT_URL")
        )
    except (OSError, UnicodeError) as exc:
        return {
            "ok": False,
            "state": "unknown",
            "configured": False,
            "error": f"heartbeat configuration unreadable: {type(exc).__name__}",
        }

    if configured:
        return {"ok": True, "state": "provisioned", "configured": True}

    now = time.time()
    source = "timer marker"
    try:
        if marker.exists():
            raw = marker.read_text(encoding="utf-8").strip()
            first_seen = float(raw)
            if not math.isfinite(first_seen) or first_seen <= 0 or first_seen > now + 300:
                raise ValueError("invalid first-seen timestamp")
        else:
            # The timer may not have run yet.  Bound that blind spot using the
            # operator config's age, then the shipped template's age.
            source = "configuration mtime"
            fallback = env_path if env_path.exists() else Path(__file__).resolve().parents[2] / "config/anima.env.example"
            first_seen = fallback.stat().st_mtime
        age = max(0.0, now - first_seen)
    except (OSError, UnicodeError, ValueError, OverflowError) as exc:
        return {
            "ok": False,
            "state": "unknown",
            "configured": False,
            "error": f"heartbeat inert state unreadable: {type(exc).__name__}",
        }

    return {
        "ok": age <= grace,
        "state": "provisioning_grace" if age <= grace else "inert",
        "configured": False,
        "inert_for_seconds": round(age),
        "grace_seconds": grace,
        "age_source": source,
    }


@dataclass
class SubsystemHealth:
    """Health state for a single subsystem."""
    name: str
    last_heartbeat: float = field(default_factory=time.time)
    probe_fn: Optional[Callable[[], ProbeResult]] = None
    last_probe_time: float = 0.0
    last_probe_ok: bool = True
    last_probe_error: str = ""
    registered: bool = True
    stale_threshold: float = 0.0  # 0 = use global default
    debounce_seconds: float = 0.0  # 0 = no debounce (instant transitions)
    optional: bool = False  # capability may legitimately not exist on this host
    probe_only: bool = False  # no loop heartbeat; status is entirely the probe
    _first_bad_at: float = 0.0  # when status first went non-ok
    _ever_ok: bool = False  # probe has succeeded (or reached a verdict) at least once this process
    last_probe_details: Optional[Dict[str, Any]] = None

    def heartbeat(self) -> None:
        self.last_heartbeat = time.time()

    def _get_stale_threshold(self) -> float:
        """Get effective stale threshold (per-subsystem or global default)."""
        return self.stale_threshold if self.stale_threshold > 0 else HEARTBEAT_STALE_SECONDS

    def run_probe(self) -> bool:
        """Run functional probe if enough time has passed. Returns probe result."""
        now = time.time()
        if self.probe_fn is None:
            self.last_probe_ok = True
            self.last_probe_time = now
            self.last_probe_details = None
            return True

        if now - self.last_probe_time < PROBE_INTERVAL_SECONDS:
            return self.last_probe_ok  # Return cached result

        self.last_probe_time = now
        try:
            result = self.probe_fn()
            evaluated = False
            if isinstance(result, dict):
                self.last_probe_details = dict(result)
                self.last_probe_ok = bool(result.get("ok", False))
                evaluated = result.get("evaluated") is True
            else:
                self.last_probe_details = None
                self.last_probe_ok = bool(result)
            # 2026-11-12: a probe that reached a real verdict has WORKED, even
            # when the verdict is bad.  Arming the guard only on ok meant an
            # optional threshold probe whose first evaluation crossed its
            # threshold read "absent" -- a measured fault hidden as a missing
            # capability.  Opt-in (`"evaluated": True` in the details dict), so
            # bool probes and dict probes without the key behave as before.
            if self.last_probe_ok or evaluated:
                self._ever_ok = True
            self.last_probe_error = "" if self.last_probe_ok else "probe returned False"
        except Exception as e:
            self.last_probe_ok = False
            self.last_probe_details = None
            self.last_probe_error = str(e)[:100]
        return self.last_probe_ok

    def _raw_status(self) -> str:
        """Compute raw status without debounce: ok, absent, stale, degraded, missing."""
        if not self.registered:
            return "missing"

        now = time.time()
        heartbeat_age = now - self.last_heartbeat if self.last_heartbeat > 0 else float("inf")
        heartbeat_stale = (
            False if self.probe_only
            else heartbeat_age > self._get_stale_threshold()
        )

        # Run probe (respects internal cooldown)
        self.run_probe()

        # An optional capability that has NEVER worked in this process is not
        # broken — it is not present. Reporting it as "degraded" saturates
        # overall() so a real failure elsewhere cannot change the top line.
        # Note the `_ever_ok` guard: once the capability HAS worked, a later
        # failure is a genuine degradation and is reported as one.
        if self.optional and not self.last_probe_ok and not self._ever_ok:
            return "absent"

        if heartbeat_stale and not self.last_probe_ok:
            return "missing"  # No heartbeat AND probe failing
        if not self.last_probe_ok:
            return "degraded"
        if heartbeat_stale:
            return "stale"
        return "ok"

    def get_status(self) -> str:
        """Compute current status with debounce: ok, stale, degraded, missing.

        When debounce_seconds > 0, non-ok status must persist for that
        duration before being reported. This prevents flickering on
        transient failures.
        """
        raw = self._raw_status()

        if raw == "ok":
            self._first_bad_at = 0.0
            return "ok"

        # "absent" is a steady state, not a transient failure — debouncing it
        # would report a healthy-looking "ok" for the grace period on every
        # single call, which is the failure-toward-healthy this file forbids.
        if raw == "absent":
            self._first_bad_at = 0.0
            return "absent"

        if self.debounce_seconds <= 0:
            return raw

        now = time.time()
        if self._first_bad_at == 0.0:
            self._first_bad_at = now
            return "ok"  # grace period starts

        if now - self._first_bad_at >= self.debounce_seconds:
            return raw  # debounce expired, report real status

        return "ok"  # still within grace period

    @property
    def is_debouncing(self) -> bool:
        """True if currently suppressing a bad status during grace period."""
        return self._first_bad_at > 0.0 and self._raw_status() != "ok"

    def to_dict(self) -> Dict[str, Any]:
        """Full status for MCP/API responses."""
        now = time.time()
        heartbeat_ago = round(now - self.last_heartbeat, 1) if self.last_heartbeat > 0 else None
        status = self.get_status()
        result: Dict[str, Any] = {
            "status": status,
            "last_heartbeat_ago_s": heartbeat_ago,
        }
        if self.probe_fn is not None:
            if status == "absent":
                # Not "failed" — this capability was never present here.
                result["probe"] = "absent: optional capability unavailable on this host"
            else:
                result["probe"] = "ok" if self.last_probe_ok else f"failed: {self.last_probe_error}"
            if self.last_probe_details is not None:
                result["probe_details"] = dict(self.last_probe_details)
        if self.is_debouncing:
            result["debouncing"] = True
        return result


class HealthRegistry:
    """Central registry for subsystem health monitoring."""

    def __init__(self) -> None:
        self._subsystems: Dict[str, SubsystemHealth] = {}
        self._creation_time = time.time()

    def register(
        self, name: str,
        probe: Optional[Callable[[], ProbeResult]] = None,
        stale_threshold: float = 0.0,
        debounce_seconds: float = 0.0,
        optional: bool = False,
        probe_only: bool = False,
    ) -> None:
        """Register a subsystem for health tracking.

        Args:
            name: Subsystem identifier (e.g., "growth", "sensors").
            probe: Optional callable returning True if subsystem is functional,
                   or a details dict containing an ``ok`` value. Called at most
                   once per PROBE_INTERVAL_SECONDS. A details dict may add
                   ``"evaluated": True`` to say it reached a real verdict from
                   real evidence; for an ``optional`` subsystem that counts as
                   having worked, so a bad first verdict is "degraded" while
                   "cannot tell yet" (``ok`` False, no ``evaluated``) stays
                   "absent".
            stale_threshold: Per-subsystem stale threshold in seconds.
                   0 = use global HEARTBEAT_STALE_SECONDS default (30s).
                   Subsystems with longer check-in intervals should set this
                   higher to avoid false-positive stale warnings.
            debounce_seconds: Grace period before reporting non-ok status.
                   0 = instant transitions (default, backward compatible).
                   Use ~6s for fast subsystems to filter transient failures.
            optional: True if this capability may legitimately not exist on
                   this host (e.g. no audio path). Such a subsystem reports
                   "absent" rather than "degraded" while it has never worked,
                   and does not drag overall(). The moment it DOES work once,
                   a later failure is reported as a real degradation.
            probe_only: True for configuration/external-state checks that have
                   no recurring in-process heartbeat. Their probe alone owns
                   status, preventing registration time from looking healthy.
        """
        if name in self._subsystems:
            # Update probe if re-registering
            self._subsystems[name].probe_fn = probe
            self._subsystems[name].registered = True
            if stale_threshold > 0:
                self._subsystems[name].stale_threshold = stale_threshold
            if debounce_seconds > 0:
                self._subsystems[name].debounce_seconds = debounce_seconds
            self._subsystems[name].optional = optional
            self._subsystems[name].probe_only = probe_only
            return

        self._subsystems[name] = SubsystemHealth(
            name=name,
            probe_fn=probe,
            registered=True,
            stale_threshold=stale_threshold,
            debounce_seconds=debounce_seconds,
            optional=optional,
            probe_only=probe_only,
        )

    def heartbeat(self, name: str) -> None:
        """Record a heartbeat for a subsystem. Call this each loop iteration."""
        sub = self._subsystems.get(name)
        if sub is not None:
            sub.heartbeat()
        else:
            # Auto-register on first heartbeat (no probe)
            self._subsystems[name] = SubsystemHealth(name=name, registered=True)
            self._subsystems[name].heartbeat()

    def status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all registered subsystems."""
        return {name: sub.to_dict() for name, sub in sorted(self._subsystems.items())}

    def overall(self) -> str:
        """Overall system health: ok, degraded, or unhealthy.

        "absent" subsystems do not contribute: an optional capability that
        was never present on this host is not a fault, and letting it pin
        the top line at "degraded" means a real fault elsewhere cannot move
        it. They remain visible per-subsystem.
        """
        statuses = [sub.get_status() for sub in self._subsystems.values()]
        if not statuses:
            return "unknown"
        if any(s in ("missing",) for s in statuses):
            return "unhealthy"
        if any(s in ("stale", "degraded") for s in statuses):
            return "degraded"
        return "ok"

    def summary_line(self) -> str:
        """One-line summary for structured logging."""
        parts = []
        for name, sub in sorted(self._subsystems.items()):
            parts.append(f"{name}={sub.get_status()}")
        return " ".join(parts)

    def subsystem_names(self) -> list:
        """Ordered list of registered subsystem names."""
        return sorted(self._subsystems.keys())

    def get_subsystem(self, name: str) -> Optional[SubsystemHealth]:
        """Get a specific subsystem's health state."""
        return self._subsystems.get(name)


# Singleton
_registry: Optional[HealthRegistry] = None


def get_health_registry() -> HealthRegistry:
    """Get or create the global health registry singleton."""
    global _registry
    if _registry is None:
        _registry = HealthRegistry()
    return _registry
