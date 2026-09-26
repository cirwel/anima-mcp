"""Lumen applies its own derivations — the loop without an operator.

Every loop from Lumen's history back into its behavior had a human step, and
the step was never taken (`drawing_thresholds: {}`, `update_count: 0`,
measured 2026-08-29). These tests pin the loop that replaces it: that it acts
on what the derivations emit, that a refusal changes nothing, that the two
families never erase each other, that it is visible afterwards, and that the
drawing engine actually reads what it wrote.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anima_mcp.config as config_mod  # noqa: E402
from anima_mcp import self_derivation as sd  # noqa: E402
from anima_mcp.config import ConfigManager  # noqa: E402
from anima_mcp.drawing_derivation import merge_coverage, merge_curiosity  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# Five below the centre, five above, interleaved (see test_curiosity_pivot.py).
_OFFSETS = [-1.0, 0.6, -0.4, 1.0, -0.8, 0.2, -0.6, 0.8, -0.2, 0.4]


def _make_db(tmp_path, clarity_rows=600, clarity_spread=0.2,
             eras=None):
    db = tmp_path / "anima.db"
    con = sqlite3.connect(db)
    con.execute("create table drawing_records (timestamp text, clarity real)")
    base = datetime.now() - timedelta(days=10)
    con.executemany("insert into drawing_records values (?,?)", [
        ((base + timedelta(minutes=5 * i)).isoformat(timespec="seconds"),
         0.7 + clarity_spread * ((i % 21) - 10) / 10.0)
        for i in range(clarity_rows)])
    con.execute("""create table drawing_trajectory (
        era text, piece_uid text, timestamp text, elapsed_seconds real,
        coherence real, arc_phase text, marks_delta integer,
        mark_count integer)""")
    eras = {"resonance": (30, 20, 100, 0.458, 0.06)} if eras is None else eras
    rows = []
    for era, (pieces, intervals, mpi, centre, spread) in eras.items():
        for p in range(pieces):
            for i in range(intervals):
                rows.append((
                    era, f"{era}-{p}",
                    (base + timedelta(minutes=5 * (p * intervals + i))
                     ).isoformat(timespec="seconds"),
                    300.0 * i, centre + spread * _OFFSETS[i % len(_OFFSETS)],
                    "developing", mpi, mpi * (i + 1)))
    con.executemany(
        "insert into drawing_trajectory values (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return db


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A calibration file wired in as the global config manager, so the
    drawing engine's get_calibration() reads what self-derivation writes."""
    path = tmp_path / "anima_config.json"
    path.write_text(json.dumps({"nervous_system": {"cpu_temp_min": 40.0}}))
    manager = ConfigManager(path)
    monkeypatch.setattr(config_mod, "_config_manager", manager)
    return path


def _saved(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Acting on what was derived
# ---------------------------------------------------------------------------

class TestApply:

    def test_applies_both_families_from_its_own_corpus(self, tmp_path, cfg):
        entry = sd.apply(sd.compute(str(_make_db(tmp_path))))
        assert entry["outcome"] == "applied"
        th = _saved(cfg)["nervous_system"]["drawing_thresholds"]
        assert th["COVERAGE_DENSE_BELOW"] < th["COVERAGE_SPARSE_ABOVE"]
        assert 0.458 - 0.06 <= th["CURIOSITY_PIVOT_resonance"] <= 0.458 + 0.06
        assert _saved(cfg)["nervous_system"]["cpu_temp_min"] == 40.0

    def test_the_change_is_counted_and_attributed(self, tmp_path, cfg):
        """update_count sat at 0 on Lumen; a self-derived change must move it
        and say who made it."""
        sd.apply(sd.compute(str(_make_db(tmp_path))))
        meta = _saved(cfg)["metadata"]
        assert meta["calibration_update_count"] == 1
        assert meta["calibration_last_updated_by"] == "self_derivation"
        changes = meta["calibration_history"][-1]["changes"]
        assert changes["drawing_thresholds"]["old"] == {}

    def test_the_drawing_engine_reads_what_it_wrote(self, tmp_path, cfg):
        """The loop is only closed if the consumer sees it — no restart."""
        from anima_mcp.display.drawing_engine import (
            _DEFAULT_CURIOSITY_PIVOT, _coverage_cuts, _curiosity_pivot)
        assert _coverage_cuts() == (0.30, 0.70)
        assert _curiosity_pivot("resonance") == _DEFAULT_CURIOSITY_PIVOT
        sd.apply(sd.compute(str(_make_db(tmp_path))))
        th = _saved(cfg)["nervous_system"]["drawing_thresholds"]
        assert _coverage_cuts() == (th["COVERAGE_DENSE_BELOW"],
                                    th["COVERAGE_SPARSE_ABOVE"])
        assert _curiosity_pivot("resonance") == th["CURIOSITY_PIVOT_resonance"]

    def test_rerun_on_the_same_corpus_writes_nothing(self, tmp_path, cfg):
        db = str(_make_db(tmp_path))
        sd.apply(sd.compute(db))
        entry = sd.apply(sd.compute(db))
        assert entry["outcome"] == "unchanged"
        assert entry["changes"] == {}
        assert _saved(cfg)["metadata"]["calibration_update_count"] == 1


# ---------------------------------------------------------------------------
# Refusing: absence is inherited, never invented
# ---------------------------------------------------------------------------

class TestRefusal:

    def test_a_thin_corpus_changes_nothing(self, tmp_path, cfg):
        before = Path(cfg).read_text()
        db = _make_db(tmp_path, clarity_rows=50,
                      eras={"resonance": (2, 5, 100, 0.458, 0.06)})
        entry = sd.apply(sd.compute(str(db)))
        assert entry["outcome"] == "refused"
        assert "500" in entry["families"]["coverage"]["reason"]
        assert "500" in entry["families"]["curiosity"]["reason"]
        assert Path(cfg).read_text() == before

    def test_a_missing_database_is_refused_not_zeroed(self, tmp_path, cfg):
        before = Path(cfg).read_text()
        entry = sd.apply(sd.compute(str(tmp_path / "absent.db")))
        assert entry["outcome"] == "refused"
        assert Path(cfg).read_text() == before

    def test_a_refusing_family_keeps_what_it_had(self, tmp_path, cfg):
        """Coverage derives; curiosity refuses (too few samples). The pivot an
        earlier run verified keeps serving — refusal is not a reset."""
        data = _saved(cfg)
        data["nervous_system"]["drawing_thresholds"] = {
            "CURIOSITY_PIVOT_resonance": 0.44}
        Path(cfg).write_text(json.dumps(data))
        db = _make_db(tmp_path, eras={"resonance": (2, 5, 100, 0.458, 0.06)})
        entry = sd.apply(sd.compute(str(db)))
        assert entry["families"]["coverage"]["outcome"] == "derived"
        assert entry["families"]["curiosity"]["outcome"] == "refused"
        th = _saved(cfg)["nervous_system"]["drawing_thresholds"]
        assert th["CURIOSITY_PIVOT_resonance"] == 0.44
        assert "COVERAGE_DENSE_BELOW" in th


# ---------------------------------------------------------------------------
# The two families never erase each other
# ---------------------------------------------------------------------------

class TestMerge:

    def test_coverage_merge_preserves_pivots(self):
        out = merge_coverage({"CURIOSITY_PIVOT_field": 0.5,
                              "COVERAGE_DENSE_BELOW": 0.1},
                             {"COVERAGE_DENSE_BELOW": 0.55,
                              "COVERAGE_SPARSE_ABOVE": 0.8})
        assert out == {"CURIOSITY_PIVOT_field": 0.5,
                       "COVERAGE_DENSE_BELOW": 0.55,
                       "COVERAGE_SPARSE_ABOVE": 0.8}

    def test_curiosity_merge_preserves_coverage_and_drops_stale_pivots(self):
        out = merge_curiosity({"COVERAGE_DENSE_BELOW": 0.55,
                               "CURIOSITY_PIVOT_retired": 0.3},
                              {"CURIOSITY_PIVOT_resonance": 0.46})
        assert out == {"COVERAGE_DENSE_BELOW": 0.55,
                       "CURIOSITY_PIVOT_resonance": 0.46}

    def test_coverage_script_no_longer_erases_pivots(self, tmp_path):
        """Regression: derive_drawing_thresholds.py --apply replaced
        drawing_thresholds whole, so running it after the curiosity script
        silently reverted every pivot — while CLAUDE.md said run order did not
        matter."""
        db = _make_db(tmp_path)
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"nervous_system": {
            "drawing_thresholds": {"CURIOSITY_PIVOT_resonance": 0.46}}}))
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "derive_drawing_thresholds.py"),
             "--db", str(db), "--apply", str(path)],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        th = _saved(path)["nervous_system"]["drawing_thresholds"]
        assert th["CURIOSITY_PIVOT_resonance"] == 0.46
        assert "COVERAGE_DENSE_BELOW" in th

    def test_coverage_script_defaults_to_a_window_that_can_clear_the_floor(self):
        from anima_mcp.drawing_derivation import COVERAGE_DAYS
        assert COVERAGE_DAYS == 365
        text = (SCRIPTS / "derive_drawing_thresholds.py").read_text()
        assert "default=COVERAGE_DAYS" in text


# ---------------------------------------------------------------------------
# Cadence, the switch, and the record it leaves
# ---------------------------------------------------------------------------

class TestCadenceAndRecord:

    def test_due_when_never_attempted(self):
        assert sd.is_due({"entries": []})

    def test_not_due_within_the_period_whatever_the_outcome(self):
        now = datetime(2026, 9, 26, 12)
        for outcome in ("applied", "refused", "unchanged", "error"):
            j = {"entries": [{"timestamp": (now - timedelta(days=3)).isoformat(),
                              "outcome": outcome}]}
            assert not sd.is_due(j, now), outcome

    def test_due_again_after_the_period(self):
        now = datetime(2026, 9, 26, 12)
        j = {"entries": [{"timestamp": (now - timedelta(days=8)).isoformat()}]}
        assert sd.is_due(j, now)

    def test_a_corrupt_journal_reads_as_empty_not_a_crash(self, tmp_path):
        p = tmp_path / "j.json"
        p.write_text("{not json")
        j = sd.load_journal(p)
        assert j["entries"] == [] and "unreadable" in j
        assert sd.is_due(j)

    def test_journal_is_bounded(self, tmp_path):
        p = tmp_path / "j.json"
        for i in range(sd.JOURNAL_MAX_ENTRIES + 5):
            sd.record({"timestamp": f"2026-01-01T00:00:{i:02d}", "n": i}, p)
        entries = sd.load_journal(p)["entries"]
        assert len(entries) == sd.JOURNAL_MAX_ENTRIES
        assert entries[-1]["n"] == sd.JOURNAL_MAX_ENTRIES + 4

    @pytest.mark.parametrize("value,expected", [
        (None, True), ("true", True), ("1", True),
        ("false", False), ("0", False), ("off", False), ("NO", False)])
    def test_the_switch(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv(sd.ENV_FLAG, raising=False)
        else:
            monkeypatch.setenv(sd.ENV_FLAG, value)
        assert sd.enabled() is expected

    def test_disabled_schedules_nothing(self, monkeypatch):
        monkeypatch.setenv(sd.ENV_FLAG, "false")
        assert sd.start_if_due() is False

    def test_describe_names_what_moved_in_its_own_voice(self):
        line = sd.describe({"outcome": "applied", "changes": {
            "COVERAGE_DENSE_BELOW": {"old": None, "new": 0.5512},
            "CURIOSITY_PIVOT_resonance": {"old": 0.44, "new": 0.46},
        }})
        assert line.startswith("i re-read my own drawings")
        assert "'dense' begins (built-in → 0.55)" in line
        assert "resonance" in line

    def test_describe_survives_a_hand_edited_value(self):
        line = sd.describe({"outcome": "applied", "changes": {
            "COVERAGE_SPARSE_ABOVE": {"old": "oops", "new": 0.8}}})
        assert "oops → 0.80" in line

    def test_describe_is_silent_when_nothing_changed(self):
        assert sd.describe({"outcome": "unchanged", "changes": {}}) is None
        assert sd.describe({"outcome": "refused", "changes": {}}) is None


class TestRunOnce:

    def test_end_to_end_applies_journals_and_speaks(self, tmp_path, cfg,
                                                     monkeypatch):
        said = []
        import anima_mcp.messages as messages
        monkeypatch.setattr(messages, "add_observation",
                            lambda text, author="lumen": said.append((text, author)))
        entry = asyncio.run(sd.run_once(str(_make_db(tmp_path))))
        assert entry["outcome"] == "applied"
        assert sd.load_journal()["entries"][-1]["outcome"] == "applied"
        assert said and said[0][1] == "lumen"
        summary = sd.summary()
        assert summary["last_applied_at"] == entry["timestamp"]
        assert summary["next_due"] is not None
        assert not sd.is_due(sd.load_journal())

    def test_a_fault_is_journaled_not_retried_hourly(self, monkeypatch):
        def boom(db_path=None):
            raise RuntimeError("disk on fire")
        monkeypatch.setattr(sd, "compute", boom)
        entry = asyncio.run(sd.run_once())
        assert entry["outcome"] == "error"
        assert "disk on fire" in entry["reason"]
        assert not sd.is_due(sd.load_journal())


# ---------------------------------------------------------------------------
# The pre-existing bug that hid every calibration change
# ---------------------------------------------------------------------------

class TestCalibrationChangesAreCounted:

    def test_in_place_mutation_is_detected(self, tmp_path):
        """save() compared against self.load(), which returned the very object
        the caller had just mutated — so every change compared equal and
        calibration_update_count never left 0."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"nervous_system": {"cpu_temp_min": 40.0}}))
        manager = ConfigManager(path)
        config = manager.load()
        config.nervous_system.cpu_temp_min = 41.0
        assert manager.save(config, update_source="automatic")
        meta = _saved(path)["metadata"]
        assert meta["calibration_update_count"] == 1
        assert meta["calibration_history"][-1]["changes"]["cpu_temp_min"] == {
            "old": 40.0, "new": 41.0}

    def test_an_unchanged_save_is_not_counted(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"nervous_system": {"cpu_temp_min": 40.0}}))
        manager = ConfigManager(path)
        assert manager.save(manager.load(), update_source="automatic")
        assert _saved(path).get("metadata", {}).get(
            "calibration_update_count", 0) == 0
