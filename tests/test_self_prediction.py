"""Lumen predicts itself, and notices when it did not expect itself.

Pins: the forecaster claims nothing before its evidence floors; a rare
transition is surprising and a habitual one is not; unknown (missing, stale,
gapped) never counts as "stayed the same"; the cut is relative to Lumen's own
distribution, not a constant; state survives a restart; the surprise band
moves no gate; and a self-surprise reaches Lumen's voice.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anima_mcp import self_prediction as sp  # noqa: E402
from anima_mcp.self_prediction import (  # noqa: E402
    EWBand, SelfForecaster, SurpriseBand, activity_level_from_shm,
)

T0 = datetime(2026, 9, 1, 0, 0)
STEP = timedelta(seconds=sp.SAMPLE_SECONDS)


def _live_days(f, days, wake=8, sleep=22, start=T0):
    """Lumen's habit: active wake..sleep, resting otherwise, minute by minute.

    Returns the time after the last sample so a scenario can continue."""
    t = start
    for _ in range(days * 24 * 60):
        level = "active" if wake <= t.hour < sleep else "resting"
        f.observe(level, t)
        t += STEP
    return t


@pytest.fixture(scope="module")
def habituated():
    f = SelfForecaster()
    end = _live_days(f, 30)
    return f, end


class TestFloors:

    def test_claims_nothing_while_young(self):
        f = SelfForecaster()
        t = T0
        for i in range(60):  # every minute a transition, rows stay < the floor
            level = LEVELS_CYCLE[i % 3]
            assert f.observe(level, t) is None
            t += STEP
        assert f.transitions == 59
        assert f.band.count == 0  # nothing scored before a row has 30 samples
        assert f.summary()["status"] == "learning"

    def test_unknown_never_counts_as_staying(self):
        f = SelfForecaster()
        f.observe("active", T0)
        f.observe(None, T0 + STEP)            # stale / missing
        f.observe("active", T0 + 2 * STEP)    # starts a new run
        assert f.samples == 0

    def test_a_gap_breaks_the_run(self):
        f = SelfForecaster()
        f.observe("resting", T0)
        f.observe("active", T0 + timedelta(hours=2))  # restart gap
        assert f.samples == 0 and f.transitions == 0

    def test_faster_than_the_cadence_is_not_a_new_sample(self):
        f = SelfForecaster()
        f.observe("active", T0)
        f.observe("active", T0 + timedelta(seconds=10))
        assert f.samples == 0
        f.observe("active", T0 + STEP)
        assert f.samples == 1


LEVELS_CYCLE = ("active", "drowsy", "resting")


class TestSurprise:

    def test_habitual_transitions_are_not_surprising(self, habituated):
        f, _ = habituated
        assert f.summary()["status"] == "forecasting"
        assert f.last_surprise is None  # 30 days of the same habit

    def test_waking_at_an_hour_it_never_wakes_is_a_self_surprise(self, habituated):
        f, end = habituated
        f = SelfForecaster.from_dict(json.loads(json.dumps(f.to_dict())))
        t = end.replace(hour=3, minute=0) + timedelta(days=1)
        f.observe("resting", t)
        f.observe("resting", t + STEP)
        s = f.observe("active", t + 2 * STEP)
        assert s is not None
        assert (s.from_level, s.to_level, s.hour) == ("resting", "active", 3)
        assert s.z > sp.SELF_SURPRISE_Z
        assert "middle of the night" in s.question()
        assert s.question().startswith("i became active")

    def test_the_cut_is_relative_to_its_own_distribution(self):
        """A creature whose transitions are all equally rare is surprised by
        none of them: the same surprisal is unremarkable against its band."""
        f = SelfForecaster()
        t = T0
        for i in range(20000):  # random-ish level every minute
            f.observe(LEVELS_CYCLE[(i * 7 + i // 5) % 3], t)
            t += STEP
        assert f.band.count >= sp.MIN_BAND_TRANSITIONS
        assert f.last_surprise is None

    def test_forecaster_beats_the_stay_as_i_am_baseline(self, habituated):
        f, _ = habituated
        summ = f.summary()
        assert summ["mean_log_loss"] <= summ["persistence_log_loss"]


class TestEWBand:

    def test_exact_while_young(self):
        b = EWBand(window=1000)
        xs = [1.0, 2.0, 4.0, 7.0]
        for x in xs:
            b.update(x)
        mean = sum(xs) / 4
        assert b.mean == pytest.approx(mean)
        assert b.var == pytest.approx(sum((x - mean) ** 2 for x in xs) / 4)

    def test_follows_a_shifted_distribution(self):
        b = EWBand(window=50)
        for _ in range(500):
            b.update(0.0)
        for _ in range(500):
            b.update(5.0)
        assert b.mean == pytest.approx(5.0, abs=0.01)

    def test_min_sigma_floor(self):
        b = EWBand(window=50)
        for _ in range(100):
            b.update(1.0)
        assert b.z(1.05, min_sigma=0.1) == pytest.approx(0.5)


class TestShmLevel:

    def _shm(self, level="active", age=1.0):
        ts = (datetime.now() - timedelta(seconds=age)).isoformat()
        return {"timestamp": ts, "activity": {"level": level}}

    def test_fresh_level(self):
        assert activity_level_from_shm(self._shm(), datetime.now(), 15) == "active"

    @pytest.mark.parametrize("shm", [
        None, {}, {"activity": {"level": "active"}},
        {"timestamp": "garbage", "activity": {"level": "active"}},
    ])
    def test_missing_is_unknown(self, shm):
        assert activity_level_from_shm(shm, datetime.now(), 15) is None

    def test_stale_is_unknown(self):
        assert activity_level_from_shm(self._shm(age=60), datetime.now(), 15) is None

    def test_unknown_level_is_unknown(self):
        assert activity_level_from_shm(self._shm(level="dancing"),
                                       datetime.now(), 15) is None


class TestSurpriseBandMovesNoGate:

    def test_reports_nothing_before_its_floor(self):
        b = SurpriseBand()
        for _ in range(sp.SURPRISE_BAND_MIN - 1):
            b.observe(0.1, 0.3)
        assert "relative_gate_rate" not in b.summary()
        assert b.summary()["moves_no_gate"] is True

    def test_compares_the_two_gates_on_lumens_own_distribution(self):
        b = SurpriseBand()
        for i in range(3000):
            b.observe(0.05 + 0.02 * math.sin(i), 0.3)  # a calm room
        b.observe(0.25, 0.3)  # unusual for Lumen, under the fixed gate
        s = b.summary()
        assert s["fixed_gate_rate"] == 0.0
        assert s["relative_gate_rate"] > 0.0

    def test_the_live_gates_are_untouched(self):
        from anima_mcp.metacognition import MetacognitiveMonitor
        from anima_mcp.server_state import METACOG_SURPRISE_THRESHOLD
        assert METACOG_SURPRISE_THRESHOLD == 0.2
        assert MetacognitiveMonitor.__init__.__defaults__[1] == 0.25


class TestPersistence:

    def test_round_trip_through_the_monitor(self, tmp_path, habituated):
        from anima_mcp.metacognition import MetacognitiveMonitor
        f, _ = habituated
        m = MetacognitiveMonitor(data_dir=str(tmp_path))
        m.self_forecaster = SelfForecaster.from_dict(f.to_dict())
        m.surprise_band.observe(0.1, 0.3)
        assert m.save()
        m2 = MetacognitiveMonitor(data_dir=str(tmp_path))
        assert m2.self_forecaster.transitions == f.transitions
        assert m2.self_forecaster.band.count == f.band.count
        assert m2.surprise_band.band.count == 1

    def test_a_legacy_baselines_file_loads(self, tmp_path):
        from anima_mcp.metacognition import MetacognitiveMonitor
        (tmp_path / "metacognition_baselines.json").write_text(
            json.dumps({"baseline_ambient_temp": 21.0}))
        m = MetacognitiveMonitor(data_dir=str(tmp_path))
        assert m.self_forecaster.samples == 0
        assert m.surprise_band.band.count == 0

    def test_the_read_only_observer_never_writes(self, tmp_path):
        from anima_mcp.metacognition import MetacognitiveMonitor
        m = MetacognitiveMonitor(data_dir=str(tmp_path), read_only=True)
        assert m.save() is False
        assert not (tmp_path / "metacognition_baselines.json").exists()

    def test_garbage_state_degrades_to_fresh(self):
        f = SelfForecaster.from_dict({"counts": {"x": 1, "active|z": {}},
                                      "band": "nope", "samples": "many"})
        assert f.samples == 0 and f.counts == {}


class TestVoice:

    def test_a_self_surprise_becomes_a_question(self, monkeypatch):
        import anima_mcp.messages as messages
        from anima_mcp.loop_phases import handle_self_surprise
        asked = []
        monkeypatch.setattr(messages, "add_question",
                            lambda text, author="lumen", context=None:
                            asked.append((text, author, context)) or object())
        s = sp.SelfSurprise(T0.replace(hour=3), "resting", "active", 3,
                            0.01, 4.6, 3.2)
        assert handle_self_surprise(s) is True
        text, author, context = asked[0]
        assert author == "lumen"
        assert context.startswith("self-surprise: resting→active at 03h")

    def test_a_refused_question_is_reported_as_not_asked(self, monkeypatch):
        import anima_mcp.messages as messages
        from anima_mcp.loop_phases import handle_self_surprise
        monkeypatch.setattr(messages, "add_question",
                            lambda *a, **k: None)
        s = sp.SelfSurprise(T0, "resting", "active", 3, 0.01, 4.6, 3.2)
        assert handle_self_surprise(s) is False
