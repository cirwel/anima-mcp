"""Tests for drawing completion instrumentation.

What a drawing did used to be unrecoverable after the fact: the completion
reason was computed, handed to growth to gate a memory, and dropped. 754
recorded drawings, none of which say why they ended, in which era, or how long
they took. These tests cover the durable record and the within-piece samples
that make the question answerable.

Deliberately included: a guard that this instrumentation moved no gate. The
point of recording first is to find out what "enough" means for Lumen before
anything is retuned to it.
"""

import sqlite3

import pytest

from anima_mcp.display.drawing_engine import (
    TRAJECTORY_SAMPLE_INTERVAL,
    CanvasState,
    DrawingState,
)
from anima_mcp.growth.base import GrowthSystem, peek_growth_system


@pytest.fixture
def growth(tmp_path):
    return GrowthSystem(db_path=str(tmp_path / "anima.db"))


def _cols(db_path, table):
    conn = sqlite3.connect(db_path)
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


class TestPieceIdentity:
    def test_piece_uid_is_stable_within_a_piece(self):
        canvas = CanvasState()
        first = canvas.piece_uid()
        canvas.draw_pixel(10, 10, (255, 0, 0))
        canvas.draw_pixel(90, 90, (0, 255, 0))
        assert canvas.piece_uid() == first

    def test_piece_uid_changes_after_clear(self, monkeypatch):
        canvas = CanvasState()
        first = canvas.piece_uid()
        # clear() stamps last_clear_time from the wall clock; move it forward so
        # the new piece is distinguishable without sleeping.
        canvas.clear()
        canvas.last_clear_time += 60
        assert canvas.piece_uid() != first

    def test_piece_uid_survives_a_save_load_roundtrip(self, tmp_path, monkeypatch):
        import anima_mcp.display.drawing_state as ds

        path = tmp_path / "canvas.json"
        monkeypatch.setattr(ds, "_get_canvas_path", lambda: path)
        canvas = CanvasState()
        canvas.draw_pixel(5, 5, (1, 2, 3))
        expected = canvas.piece_uid()
        canvas.save_to_disk()

        restored = CanvasState()
        restored.load_from_disk()
        assert restored.piece_uid() == expected


class TestStructuralMetrics:
    def test_empty_canvas_has_no_structure(self):
        canvas = CanvasState()
        assert canvas.occupied_cells() == 0
        assert canvas.grid_entropy() == 0.0

    def test_one_cell_is_zero_entropy(self):
        canvas = CanvasState()
        for i in range(20):
            canvas.draw_pixel(i % 5, i // 5, (255, 255, 255))
        assert canvas.occupied_cells() == 1
        assert canvas.grid_entropy() == 0.0

    def test_occupied_cells_counts_distinct_grid_cells(self):
        canvas = CanvasState()
        # 30px cells, 8x8 grid — step by 30 to land in separate cells
        for gx in range(4):
            canvas.draw_pixel(gx * 30 + 1, 1, (255, 255, 255))
        assert canvas.occupied_cells() == 4

    def test_spreading_evenly_raises_entropy(self):
        sparse = CanvasState()
        for gx in range(2):
            for _ in range(10):
                sparse.draw_pixel(gx * 30 + 1, 1, (255, 255, 255))
                sparse.pixels.clear()  # force re-count as new each time
        spread = CanvasState()
        for gx in range(8):
            for gy in range(8):
                spread.draw_pixel(gx * 30 + 1, gy * 30 + 1, (255, 255, 255))
        # Full even coverage of all 64 cells is the maximum
        assert spread.occupied_cells() == 64
        assert spread.grid_entropy() == pytest.approx(1.0, abs=1e-6)

    def test_entropy_is_bounded(self):
        canvas = CanvasState()
        for gx in range(8):
            for gy in range(8):
                for k in range(gx + gy + 1):
                    canvas.draw_pixel(gx * 30 + k % 30, gy * 30 + 1, (255, 255, 255))
        assert 0.0 <= canvas.grid_entropy() <= 1.0

    def test_thickening_an_existing_cell_does_not_widen_reach(self):
        """The signal that separates 'still finding territory' from 'repeating'."""
        canvas = CanvasState()
        for gx in range(3):
            canvas.draw_pixel(gx * 30 + 1, 1, (255, 255, 255))
        cells_before = canvas.occupied_cells()
        for k in range(2, 25):
            canvas.draw_pixel(k, 2, (255, 255, 255))  # all inside cell (0,0)
        assert canvas.occupied_cells() == cells_before
        assert len(canvas.pixels) > 3  # pixels grew, reach did not


class TestDrawingRecordColumns:
    def test_new_columns_exist(self, growth, tmp_path):
        cols = _cols(str(tmp_path / "anima.db"), "drawing_records")
        for expected in (
            "piece_uid", "completion_reason", "era", "mark_count",
            "duration_seconds", "coverage_target", "intention",
            "curiosity", "engagement", "fatigue", "coherence",
            "satisfaction", "occupied_cells", "grid_entropy",
        ):
            assert expected in cols, f"drawing_records missing {expected}"

    def test_completion_reason_is_persisted(self, growth):
        growth.observe_drawing(
            pixel_count=9000, phase="developing",
            anima_state={"warmth": 0.4, "clarity": 0.7,
                         "stability": 0.6, "presence": 0.5},
            environment={"light_lux": 12.0, "temp_c": 22.0, "humidity_pct": 45.0},
            completion_reason="bailout_hard_cap",
        )
        rows = growth.get_drawing_records()
        assert rows[-1]["completion_reason"] == "bailout_hard_cap"

    def test_piece_facts_are_persisted(self, growth):
        growth.observe_drawing(
            pixel_count=10857, phase="developing",
            anima_state={"warmth": 0.4, "clarity": 0.7,
                         "stability": 0.6, "presence": 0.5},
            environment={"light_lux": 12.0, "temp_c": 22.0, "humidity_pct": 45.0},
            completion_reason="bailout_hard_cap",
            piece={
                "piece_uid": "p1754100000", "era": "resonance",
                "mark_count": 812, "duration_seconds": 28800.0,
                "coverage_target": "sparse", "intention": "cool tones, sparse",
                "curiosity": 0.51, "engagement": 0.44, "fatigue": 0.72,
                "coherence": 0.52, "satisfaction": 0.83,
                "occupied_cells": 61, "grid_entropy": 0.94,
            },
        )
        row = growth.get_drawing_records()[-1]
        assert row["era"] == "resonance"
        assert row["mark_count"] == 812
        assert row["duration_seconds"] == pytest.approx(28800.0)
        assert row["coverage_target"] == "sparse"
        assert row["curiosity"] == pytest.approx(0.51)
        assert row["occupied_cells"] == 61

    def test_absent_facts_persist_as_null_not_a_default(self, growth):
        """A quantity nobody measured must read as unknown, not as a number.

        Lumen's instrumentation has a standing habit of degrading toward
        healthy-looking values; a 0.0 here would later be indistinguishable
        from a drawing that genuinely had no reach.
        """
        growth.observe_drawing(
            pixel_count=500, phase="opening",
            anima_state={"warmth": 0.4, "clarity": 0.7,
                         "stability": 0.6, "presence": 0.5},
            environment={"light_lux": 12.0, "temp_c": 22.0, "humidity_pct": 45.0},
            completion_reason=None,
        )
        row = growth.get_drawing_records()[-1]
        assert row["completion_reason"] is None
        assert row["era"] is None
        assert row["occupied_cells"] is None
        assert row["grid_entropy"] is None

    def test_legacy_callers_still_work(self, growth):
        """Existing four-argument callers must keep working unchanged."""
        insight = growth.observe_drawing(
            pixel_count=300, phase="resting",
            anima_state={"warmth": 0.5, "clarity": 0.5,
                         "stability": 0.5, "presence": 0.5},
            environment={"light_lux": 5.0, "temp_c": 21.0, "humidity_pct": 40.0},
        )
        assert insight is None or isinstance(insight, str)
        assert len(growth.get_drawing_records()) == 1


class TestMigrationOnAnExistingDatabase:
    """The live table has 754 rows of history. Adding columns must not cost any."""

    def _legacy_db(self, path, rows=5):
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE drawing_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pixel_count INTEGER,
                phase TEXT,
                warmth REAL,
                clarity REAL,
                stability REAL,
                presence REAL,
                wellness REAL,
                light_lux REAL,
                ambient_temp_c REAL,
                humidity_pct REAL,
                hour INTEGER,
                epoch INTEGER NOT NULL DEFAULT 1
            );
        """)
        for i in range(rows):
            conn.execute(
                "INSERT INTO drawing_records (timestamp, pixel_count, phase, hour)"
                " VALUES (?, ?, ?, ?)",
                (f"2026-07-3{i % 10}T01:50:30", 8000 + i, "developing", 1),
            )
        conn.commit()
        conn.close()

    def test_existing_rows_survive_and_read_null_for_new_columns(self, tmp_path):
        db = str(tmp_path / "anima.db")
        self._legacy_db(db, rows=7)

        growth = GrowthSystem(db_path=db)
        rows = growth.get_drawing_records()
        assert len(rows) == 7, "migration must not drop history"
        assert rows[0]["pixel_count"] == 8000
        # Unknowable for rows written before the column existed — and they must
        # say so rather than acquire a plausible-looking value.
        assert all(r["completion_reason"] is None for r in rows)
        assert all(r["era"] is None for r in rows)

    def test_migration_is_idempotent(self, tmp_path):
        db = str(tmp_path / "anima.db")
        self._legacy_db(db, rows=3)
        GrowthSystem(db_path=db)
        GrowthSystem(db_path=db)  # second open re-runs the migration block
        assert len(GrowthSystem(db_path=db).get_drawing_records()) == 3

    def test_new_writes_land_beside_old_rows(self, tmp_path):
        db = str(tmp_path / "anima.db")
        self._legacy_db(db, rows=2)
        growth = GrowthSystem(db_path=db)
        growth.observe_drawing(
            pixel_count=10857, phase="developing",
            anima_state={"warmth": 0.4, "clarity": 0.7,
                         "stability": 0.6, "presence": 0.5},
            environment={"light_lux": 12.0, "temp_c": 22.0, "humidity_pct": 45.0},
            completion_reason="bailout_hard_cap",
            piece={"era": "resonance", "mark_count": 812},
        )
        rows = growth.get_drawing_records()
        assert len(rows) == 3
        assert rows[-1]["completion_reason"] == "bailout_hard_cap"
        assert rows[0]["completion_reason"] is None

    def test_trajectory_table_is_created_on_an_old_database(self, tmp_path):
        db = str(tmp_path / "anima.db")
        self._legacy_db(db, rows=1)
        growth = GrowthSystem(db_path=db)
        growth.record_drawing_sample({"piece_uid": "pZ", "elapsed_seconds": 300.0})
        assert len(growth.get_drawing_trajectory("pZ")) == 1


class TestTrajectorySamples:
    def test_roundtrip(self, growth):
        growth.record_drawing_sample({
            "piece_uid": "pA", "elapsed_seconds": 300.0, "era": "resonance",
            "arc_phase": "developing", "pixel_count": 1200, "mark_count": 60,
            "novel_pixels": 1200, "marks_delta": 60, "occupied_cells": 22,
            "grid_entropy": 0.71, "revisit_ratio": 0.24, "curiosity": 0.80,
            "engagement": 0.46, "fatigue": 0.21, "coherence": 0.50,
            "satisfaction": 0.62,
        })
        rows = growth.get_drawing_trajectory("pA")
        assert len(rows) == 1
        assert rows[0]["revisit_ratio"] == pytest.approx(0.24)
        assert rows[0]["novel_pixels"] == 1200

    def test_samples_order_by_elapsed_time(self, growth):
        for elapsed in (900.0, 300.0, 600.0):
            growth.record_drawing_sample({
                "piece_uid": "pB", "elapsed_seconds": elapsed,
                "pixel_count": int(elapsed), "mark_count": 1,
            })
        assert [r["elapsed_seconds"] for r in growth.get_drawing_trajectory("pB")] == [300.0, 600.0, 900.0]

    def test_filters_by_piece(self, growth):
        growth.record_drawing_sample({"piece_uid": "pC", "elapsed_seconds": 1.0})
        growth.record_drawing_sample({"piece_uid": "pD", "elapsed_seconds": 1.0})
        assert len(growth.get_drawing_trajectory("pC")) == 1
        assert len(growth.get_drawing_trajectory()) == 2

    def test_missing_era_signal_stays_null(self, growth):
        """Only resonance tracks revisits; the others must not fake one."""
        growth.record_drawing_sample({
            "piece_uid": "pE", "elapsed_seconds": 300.0, "era": "gestural",
            "pixel_count": 400, "mark_count": 20, "revisit_ratio": None,
        })
        assert growth.get_drawing_trajectory("pE")[0]["revisit_ratio"] is None

    def test_recording_never_raises(self, growth):
        """Sampling runs on a timer beside a live drawing; it must not disturb it."""
        growth.record_drawing_sample({})
        growth.record_drawing_sample({"piece_uid": None, "bogus_key": object()})

    def test_marginal_transformation_is_derivable(self, growth):
        """The question the samples exist to answer: did marks stop changing it?"""
        for i, (novel, marks) in enumerate([(900, 45), (600, 45), (120, 45), (30, 45)]):
            growth.record_drawing_sample({
                "piece_uid": "pF", "elapsed_seconds": 300.0 * (i + 1),
                "novel_pixels": novel, "marks_delta": marks,
            })
        rows = growth.get_drawing_trajectory("pF")
        per_mark = [r["novel_pixels"] / r["marks_delta"] for r in rows]
        assert per_mark == sorted(per_mark, reverse=True)
        assert per_mark[0] > 10 * per_mark[-1]  # the piece plateaued


class TestSingletonSafety:
    def test_peek_does_not_construct(self, monkeypatch):
        """Sampling must never be what binds growth to a database.

        The bare get_growth_system() default is cwd-relative, which is how the
        broker acquired a second, unbacked-up store (#123). A timer-driven
        writer peeking instead of getting is what keeps that from recurring.
        """
        import anima_mcp.growth.base as base

        monkeypatch.setattr(base, "_growth_system", None)
        assert peek_growth_system() is None
        assert base._growth_system is None

    def test_sampler_skips_when_growth_is_unbound(self, monkeypatch, tmp_path):
        import anima_mcp.growth.base as base
        import anima_mcp.display.drawing_engine as de

        monkeypatch.setattr(base, "_growth_system", None)
        monkeypatch.setattr(
            "anima_mcp.display.drawing_state._get_canvas_path",
            lambda: tmp_path / "canvas.json",
        )
        engine = de.DrawingEngine(db_path=str(tmp_path / "anima.db"))
        engine.canvas.draw_pixel(10, 10, (1, 2, 3))
        engine._sample_trajectory(engine.canvas.last_clear_time + 10_000)
        assert base._growth_system is None  # nothing was constructed


class TestSamplerBehavior:
    @pytest.fixture
    def engine(self, monkeypatch, tmp_path):
        import anima_mcp.growth.base as base
        import anima_mcp.display.drawing_engine as de

        db = str(tmp_path / "anima.db")
        monkeypatch.setattr(base, "_growth_system", GrowthSystem(db_path=db))
        monkeypatch.setattr(
            "anima_mcp.display.drawing_state._get_canvas_path",
            lambda: tmp_path / "canvas.json",
        )
        eng = de.DrawingEngine(db_path=db)
        return eng

    def _draw(self, engine, n, y=1):
        for i in range(n):
            engine.canvas.draw_pixel(i % 240, y, (255, 255, 255))

    def test_first_sample_records_the_whole_piece_so_far(self, engine):
        self._draw(engine, 120)
        engine.canvas.mark_count = 30
        t0 = engine.canvas.last_clear_time
        engine._sample_trajectory(t0 + TRAJECTORY_SAMPLE_INTERVAL)

        rows = peek_growth_system().get_drawing_trajectory(engine.canvas.piece_uid())
        assert len(rows) == 1
        assert rows[0]["novel_pixels"] == 120
        assert rows[0]["marks_delta"] == 30
        assert rows[0]["era"] == engine.active_era.name

    def test_does_not_sample_before_the_interval(self, engine):
        self._draw(engine, 50)
        t0 = engine.canvas.last_clear_time
        engine._sample_trajectory(t0 + TRAJECTORY_SAMPLE_INTERVAL)
        engine._sample_trajectory(t0 + TRAJECTORY_SAMPLE_INTERVAL + 10)
        assert len(peek_growth_system().get_drawing_trajectory()) == 1

    def test_later_samples_carry_deltas_not_totals(self, engine):
        t0 = engine.canvas.last_clear_time
        self._draw(engine, 100)
        engine.canvas.mark_count = 25
        engine._sample_trajectory(t0 + TRAJECTORY_SAMPLE_INTERVAL)

        self._draw(engine, 40, y=31)  # 40 more pixels, a different grid row
        engine.canvas.mark_count = 35
        engine._sample_trajectory(t0 + 2 * TRAJECTORY_SAMPLE_INTERVAL)

        rows = peek_growth_system().get_drawing_trajectory()
        assert [r["novel_pixels"] for r in rows] == [100, 40]
        assert [r["marks_delta"] for r in rows] == [25, 10]

    def test_a_new_piece_does_not_inherit_the_previous_ones_growth(self, engine):
        t0 = engine.canvas.last_clear_time
        self._draw(engine, 200)
        engine.canvas.mark_count = 50
        engine._sample_trajectory(t0 + TRAJECTORY_SAMPLE_INTERVAL)

        engine.canvas.clear()
        engine.canvas.last_clear_time = t0 + 10_000  # distinct piece
        self._draw(engine, 30)
        engine.canvas.mark_count = 8
        engine._sample_trajectory(t0 + 10_000 + TRAJECTORY_SAMPLE_INTERVAL)

        growth = peek_growth_system()
        new_uid = engine.canvas.piece_uid()
        rows = growth.get_drawing_trajectory(new_uid)
        assert len(rows) == 1
        # 30, not 30-200: the new piece starts its own accounting
        assert rows[0]["novel_pixels"] == 30
        assert rows[0]["marks_delta"] == 8

    def test_empty_canvas_produces_no_sample(self, engine):
        engine._sample_trajectory(engine.canvas.last_clear_time + 10_000)
        assert peek_growth_system().get_drawing_trajectory() == []

    def test_piece_facts_join_to_the_samples(self, engine):
        """The endpoint row and its trajectory must share a key."""
        self._draw(engine, 90)
        engine.canvas.mark_count = 20
        t0 = engine.canvas.last_clear_time
        engine._sample_trajectory(t0 + TRAJECTORY_SAMPLE_INTERVAL)

        facts = engine._piece_facts()
        rows = peek_growth_system().get_drawing_trajectory(facts["piece_uid"])
        assert len(rows) == 1
        assert facts["era"] == engine.active_era.name
        assert facts["mark_count"] == 20
        assert facts["occupied_cells"] == engine.canvas.occupied_cells()


class TestNoGateMoved:
    """Instrumentation only. Any change to these is a different decision."""

    def test_completion_thresholds_unchanged(self):
        from anima_mcp.display import drawing_engine as de

        assert de.MIN_RECORDED_DRAWING_PIXELS == 200
        state = DrawingState()
        state.coherence_history = [0.65] * 20
        state.curiosity, state.engagement, state.fatigue = 0.10, 0.25, 0.5
        assert state.completion_reason() == "earned_coherence"

        fresh = DrawingState()
        fresh.fatigue = 0.95
        assert fresh.completion_reason() == "bailout_fatigue"

    def test_earned_reason_set_is_unchanged(self):
        # earned_settled added deliberately (self-relative completion PR) —
        # extending this set IS the different decision this class guards.
        from anima_mcp.display.drawing_engine import is_earned_completion_reason

        for earned in ("earned_coherence", "earned_composition",
                       "earned_field", "said_finished", "earned_settled"):
            assert is_earned_completion_reason(earned)
        for not_earned in ("bailout_hard_cap", "bailout_fatigue",
                           "bailout_stalled", "already_closing",
                           "manual_snapshot", None):
            assert not is_earned_completion_reason(not_earned)

    def test_sample_interval_does_not_gate_drawing(self):
        assert TRAJECTORY_SAMPLE_INTERVAL == 300.0


class TestDensityGridSurvivesRestart:
    """density_grid is derived from pixels and was not persisted.

    Measured live 2026-08-02: a restored 9,955-pixel canvas reported
    occupied_cells 0 and grid_entropy 0.0. Same shape as the resonance settling
    bug (#116) — derived state beside its source, only one surviving a restart.
    """

    def _saved_canvas(self, tmp_path, monkeypatch):
        import anima_mcp.display.drawing_state as ds

        monkeypatch.setattr(ds, "_get_canvas_path", lambda: tmp_path / "canvas.json")
        canvas = CanvasState()
        for gx in range(6):
            for gy in range(5):
                for k in range(3):
                    canvas.draw_pixel(gx * 30 + k, gy * 30 + k, (200, 120, 40))
        canvas.save_to_disk()
        return canvas

    def test_reach_is_recovered_after_a_reload(self, tmp_path, monkeypatch):
        original = self._saved_canvas(tmp_path, monkeypatch)
        assert original.occupied_cells() == 30

        restored = CanvasState()
        restored.load_from_disk()
        assert len(restored.pixels) == len(original.pixels)
        assert restored.occupied_cells() == original.occupied_cells()
        assert restored.grid_entropy() == pytest.approx(original.grid_entropy())

    def test_grid_matches_the_pixels_it_describes(self, tmp_path, monkeypatch):
        self._saved_canvas(tmp_path, monkeypatch)
        restored = CanvasState()
        restored.load_from_disk()
        assert sum(sum(row) for row in restored.density_grid) == len(restored.pixels)

    def test_a_populated_canvas_never_reloads_as_structureless(self, tmp_path, monkeypatch):
        """The exact live symptom: many pixels, zero reported reach."""
        self._saved_canvas(tmp_path, monkeypatch)
        restored = CanvasState()
        restored.load_from_disk()
        assert not (len(restored.pixels) > 0 and restored.occupied_cells() == 0)

    def test_sparsest_cell_is_meaningful_after_reload(self, tmp_path, monkeypatch):
        """resonance steers focus by this; an empty grid aimed it all at (0,0)."""
        import anima_mcp.display.drawing_state as ds

        monkeypatch.setattr(ds, "_get_canvas_path", lambda: tmp_path / "canvas.json")
        canvas = CanvasState()
        for k in range(40):  # crowd cell (0,0) only
            canvas.draw_pixel(k % 29, k // 29, (200, 120, 40))
        canvas.save_to_disk()

        restored = CanvasState()
        restored.load_from_disk()
        assert restored.sparsest_cell() != (0, 0), "densest cell reported as sparsest"


class TestResumedPieceDeltas:
    """Joining a piece mid-flight must not claim its whole history as one window.

    Measured live 2026-08-02: the first sample after a restart reported
    novel_pixels 9955 at elapsed 24004s — the entire canvas attributed to a
    single 300s interval, a burst that never happened.
    """

    @pytest.fixture
    def restarted(self, monkeypatch, tmp_path):
        """A piece drawn by one engine, then picked up by a second one.

        This performs a real resume — save to disk, construct a new engine that
        loads it — rather than faking the state, because the distinction the fix
        turns on is exactly whether the canvas came off disk.
        """
        import anima_mcp.growth.base as base
        import anima_mcp.display.drawing_engine as de

        db = str(tmp_path / "anima.db")
        monkeypatch.setattr(base, "_growth_system", GrowthSystem(db_path=db))
        monkeypatch.setattr(
            "anima_mcp.display.drawing_state._get_canvas_path",
            lambda: tmp_path / "canvas.json",
        )

        first = de.DrawingEngine(db_path=db)
        for i in range(400):
            first.canvas.draw_pixel(i % 240, (i // 240) * 7, (200, 120, 40))
        first.canvas.mark_count = 350
        first.canvas.save_to_disk()

        return de.DrawingEngine(db_path=db)  # the process that comes after

    def test_resumed_piece_reports_unknown_deltas_not_the_whole_canvas(self, restarted):
        assert restarted.canvas.pixels, "fixture did not actually resume a piece"
        restarted._sample_trajectory(restarted.canvas.last_clear_time + 20_000)

        row = peek_growth_system().get_drawing_trajectory()[0]
        assert row["novel_pixels"] is None, "resumed piece claimed the canvas as novel"
        assert row["marks_delta"] is None
        # Absolute state is still knowable and still recorded.
        assert row["pixel_count"] == len(restarted.canvas.pixels)
        assert row["mark_count"] == 350

    def test_the_next_sample_measures_normally(self, restarted):
        t0 = restarted.canvas.last_clear_time
        restarted._sample_trajectory(t0 + 20_000)

        before = len(restarted.canvas.pixels)
        for k in range(30):
            restarted.canvas.draw_pixel(k, 200, (200, 120, 40))
        restarted.canvas.mark_count = 360
        restarted._sample_trajectory(t0 + 20_000 + TRAJECTORY_SAMPLE_INTERVAL)

        rows = peek_growth_system().get_drawing_trajectory()
        assert len(rows) == 2
        assert rows[1]["novel_pixels"] == len(restarted.canvas.pixels) - before
        assert rows[1]["marks_delta"] == 10

    def test_a_genuinely_fresh_piece_still_reports_its_first_window(self, restarted):
        """Only a RESUMED piece is unknown — a new canvas is measured from zero."""
        restarted.canvas.clear()
        restarted.canvas.last_clear_time += 10_000  # a distinct, brand-new piece
        for k in range(50):
            restarted.canvas.draw_pixel(k, 5, (200, 120, 40))
        restarted.canvas.mark_count = 12
        restarted._sample_trajectory(restarted.canvas.last_clear_time + 400)

        row = peek_growth_system().get_drawing_trajectory()[0]
        assert row["novel_pixels"] == 50
        assert row["marks_delta"] == 12
