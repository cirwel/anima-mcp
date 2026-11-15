"""Drawing process recording (display/drawing_replay.py) and its renderer.

The contract under test: a replay, applied record by record, ends on exactly
the pixels of the PNG it sits beside — across restarts, lost flushes and
pieces that began before recording existed — and recording never changes
what Lumen draws.
"""

import importlib.util
import json
import random
from pathlib import Path

import pytest

import anima_mcp.display.drawing_engine as de
from anima_mcp.display.drawing_engine import DrawingEngine
from anima_mcp.display.drawing_replay import (
    ReplayRecorder,
    final_pixels,
    iter_records,
    replay_path_for,
)
from anima_mcp.display.drawing_state import CanvasState

from conftest import make_anima

REPO = Path(__file__).resolve().parents[1]


def _load_renderer():
    spec = importlib.util.spec_from_file_location(
        "render_drawing_replay", REPO / "scripts" / "render_drawing_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def canvas_path(monkeypatch, tmp_path):
    path = tmp_path / "canvas.json"
    monkeypatch.setattr(de, "_get_canvas_path", lambda: path)
    return path


def _recorder_for(canvas, tmp_path, era="gestural"):
    rec = ReplayRecorder(lambda: tmp_path / "canvas.replay.jsonl")
    rec.open(canvas, era)
    canvas._replay_tap = rec.pending
    return rec


def _mark(canvas, rec, pts, color=(200, 100, 50), t=None):
    for x, y in pts:
        canvas.draw_pixel(x, y, color)
    rec.record_mark(gesture="dot", eisv=(0.5, 0.5, 0.2, 0.0, 0.4), arc_phase="opening", now=t)


class TestRecorder:

    def test_marks_overwrites_and_order_survive(self, tmp_path):
        canvas = CanvasState()
        rec = _recorder_for(canvas, tmp_path)
        _mark(canvas, rec, [(1, 1), (2, 2)], (10, 10, 10), t=100.0)
        _mark(canvas, rec, [(2, 2), (3, 3)], (20, 20, 20), t=130.0)  # draws over (2,2)
        rec.flush()
        png = tmp_path / "lumen_drawing_x.png"
        out = rec.finalize(png, canvas, reason="earned_settled")

        assert out == replay_path_for(png)
        assert out.name == "lumen_drawing_x.replay.jsonl.gz"
        records = list(iter_records(out))
        kinds = [r["type"] for r in records]
        assert kinds == ["header", "mark", "mark", "end"]
        assert records[1]["px"] == [1, 1, 10, 10, 10, 2, 2, 10, 10, 10]
        assert records[2]["t"] - records[1]["t"] == pytest.approx(30.0)
        assert records[-1]["reason"] == "earned_settled"
        # The PNG cannot say (2,2) was drawn twice; the replay can.
        assert final_pixels(records) == canvas.pixels

    def test_mark_with_no_pixels_writes_nothing(self, tmp_path):
        canvas = CanvasState()
        rec = _recorder_for(canvas, tmp_path)
        rec.record_mark(gesture="dot", eisv=(0, 0, 0, 0, 0), arc_phase="opening")
        rec.flush()
        kinds = [r["type"] for r in iter_records(rec.path())]
        assert kinds == ["header"]

    def test_pre_existing_piece_is_a_labelled_base(self, tmp_path):
        canvas = CanvasState()
        canvas.draw_pixel(5, 5, (1, 2, 3))  # drawn before any recorder existed
        rec = _recorder_for(canvas, tmp_path)
        records = list(iter_records(rec.path()))
        assert records[1]["type"] == "base"
        assert records[1]["why"] == "pre-existing"
        assert final_pixels(records) == canvas.pixels

    def test_restart_resumes_same_piece_and_reconciles(self, tmp_path):
        canvas = CanvasState()
        rec = _recorder_for(canvas, tmp_path)
        _mark(canvas, rec, [(1, 1)])
        rec.flush()
        _mark(canvas, rec, [(9, 9)])  # never flushed: the process dies here
        del canvas.pixels[(1, 1)]     # and the persisted canvas disagrees too

        rec2 = _recorder_for(canvas, tmp_path)  # same piece_uid -> resume
        records = list(iter_records(rec2.path()))
        kinds = [r["type"] for r in records]
        assert kinds == ["header", "mark", "base", "resume"]
        base = records[2]
        assert base["why"] == "reconcile"
        assert base["px"][:2] == [9, 9]
        assert base["erase"] == [1, 1]
        assert final_pixels(records) == canvas.pixels

    def test_new_piece_truncates_previous_log(self, tmp_path):
        canvas = CanvasState()
        rec = _recorder_for(canvas, tmp_path)
        _mark(canvas, rec, [(1, 1)])
        rec.flush()
        canvas.clear()
        canvas.last_clear_time += 1  # a distinct piece uid
        rec.start_piece(canvas, "field")
        records = list(iter_records(rec.path()))
        assert [r["type"] for r in records] == ["header"]
        assert records[0]["era"] == "field"
        assert records[0]["piece"] == canvas.piece_uid()

    def test_finalize_reconciles_a_lost_flush(self, tmp_path, monkeypatch):
        canvas = CanvasState()
        rec = _recorder_for(canvas, tmp_path)
        _mark(canvas, rec, [(1, 1)])
        rec.flush()
        canvas.draw_pixel(7, 7, (5, 5, 5))  # a pixel the log never saw as a mark
        rec.pending.clear()
        out = rec.finalize(tmp_path / "p.png", canvas)
        records = list(iter_records(out))
        assert records[-2]["type"] == "base" and records[-2]["why"] == "reconcile"
        assert final_pixels(records) == canvas.pixels

    def test_switched_off_piece_gets_no_replay_rather_than_a_wrong_one(self, tmp_path, monkeypatch):
        import anima_mcp.display.drawing_replay as dr
        monkeypatch.setattr(dr, "MAX_IN_PROGRESS_BYTES", 10)
        canvas = CanvasState()
        rec = _recorder_for(canvas, tmp_path)
        _mark(canvas, rec, [(1, 1)])
        rec.flush()  # file now over the (patched) cap
        assert rec.finalize(tmp_path / "p.png", canvas) is None

    def test_recorder_failure_never_raises(self, tmp_path):
        canvas = CanvasState()
        rec = ReplayRecorder(lambda: (_ for _ in ()).throw(OSError("disk gone")))
        rec.open(canvas, "gestural")
        canvas._replay_tap = rec.pending
        _mark(canvas, rec, [(1, 1)])
        rec.flush()
        assert rec.finalize(tmp_path / "p.png", canvas) is None
        assert canvas.pixels == {(1, 1): (200, 100, 50)}

    def test_torn_final_line_is_skipped(self, tmp_path):
        p = tmp_path / "x.replay.jsonl"
        p.write_text('{"type":"header","v":1}\n{"type":"mark","t":1,"px":[1,1,9,9,9]}\n{"type":"ma')
        assert [r["type"] for r in iter_records(p)] == ["header", "mark"]


class TestEngineWiring:

    def _draw_until(self, engine, marks, monkeypatch):
        monkeypatch.setattr(random, "random", lambda _r=random.Random(7): _r.random() * 0.01)
        anima = make_anima(warmth=0.6, clarity=0.7, stability=0.6, presence=0.7)
        for _ in range(marks):
            engine.draw(anima)

    def test_saved_drawing_has_replay_that_ends_on_its_pixels(self, canvas_path, tmp_path, monkeypatch):
        engine = DrawingEngine()
        self._draw_until(engine, 60, monkeypatch)
        assert engine.canvas.pixels
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        saved = engine.canvas_save(manual=True)
        replay = replay_path_for(Path(saved))
        assert replay.exists()
        records = list(iter_records(replay))
        assert records[0]["type"] == "header"
        assert sum(r["type"] == "mark" for r in records) > 0
        assert all(len(r["s"]) == 5 for r in records if r["type"] == "mark")
        assert final_pixels(records) == engine.canvas.pixels

    def test_clear_starts_a_new_log_for_the_new_piece(self, canvas_path, monkeypatch):
        engine = DrawingEngine()
        self._draw_until(engine, 20, monkeypatch)
        engine.canvas.drawing_paused_until = 0.0
        engine.canvas_clear(persist=True, already_saved=True)
        log = canvas_path.with_name("canvas.replay.jsonl")
        records = list(iter_records(log))
        assert [r["type"] for r in records] == ["header"]
        assert records[0]["piece"] == engine.canvas.piece_uid()

    def test_forced_era_switch_mid_piece_is_recorded(self, canvas_path, monkeypatch):
        engine = DrawingEngine()
        self._draw_until(engine, 20, monkeypatch)
        engine.set_era("field", force_immediate=True)
        records = list(iter_records(canvas_path.with_name("canvas.replay.jsonl")))
        assert records[-1]["type"] == "era" and records[-1]["era"] == "field"

    def test_recording_does_not_change_what_is_drawn(self, canvas_path, tmp_path, monkeypatch):
        """Same seeds, tap on vs. tap off: identical canvases."""
        def run(tap_on):
            random.seed(11)
            import numpy as np
            np.random.seed(11)
            engine = DrawingEngine()
            if not tap_on:
                engine.canvas._replay_tap = None
            anima = make_anima(warmth=0.6, clarity=0.7, stability=0.6, presence=0.7)
            for _ in range(200):
                engine.draw(anima)
            return dict(engine.canvas.pixels)

        canvas_path.unlink(missing_ok=True)
        with_tap = run(True)
        canvas_path.unlink(missing_ok=True)
        canvas_path.with_name("canvas.replay.jsonl").unlink(missing_ok=True)
        without_tap = run(False)
        assert with_tap and with_tap == without_tap


class TestRenderer:

    def _replay(self, tmp_path):
        canvas = CanvasState()
        rec = _recorder_for(canvas, tmp_path)
        t = 1000.0
        for i in range(30):
            _mark(canvas, rec, [(i, i), (i + 1, i)], ((i * 8) % 256, 120, 200), t=t)
            t += 5000.0 if i == 10 else 20.0  # one very long pause
        out = rec.finalize(tmp_path / "piece.png", canvas)
        return out, canvas

    def test_wall_clock_caps_long_pauses_and_drops_downtime(self):
        r = _load_renderer()
        recs = [
            {"type": "header", "w": 240, "h": 240},
            {"type": "mark", "t": 0.0, "px": [0, 0, 1, 1, 1]},
            {"type": "mark", "t": 10.0, "px": [1, 0, 1, 1, 1]},
            {"type": "mark", "t": 5000.0, "px": [2, 0, 1, 1, 1]},
            {"type": "resume", "t": 90000.0},
            {"type": "mark", "t": 90005.0, "px": [3, 0, 1, 1, 1]},
        ]
        events, ticks, _ = r.build_timeline(recs, "wall", max_gap=120.0)
        assert [p for p, _ in events] == [0.0, 10.0, 130.0, 135.0]
        assert ticks == [130.0]
        events, _, _ = r.build_timeline(recs, "marks", max_gap=120.0)
        assert [p for p, _ in events] == [0.0, 1.0, 2.0, 3.0]

    def test_era_switch_takes_no_time_on_the_marks_clock(self):
        r = _load_renderer()
        recs = [
            {"type": "header", "w": 240, "h": 240, "era": "gestural"},
            {"type": "mark", "t": 0.0, "px": [0, 0, 1, 1, 1]},
            {"type": "era", "t": 5.0, "era": "field"},
            {"type": "mark", "t": 10.0, "px": [1, 0, 1, 1, 1]},
        ]
        events, _, _ = r.build_timeline(recs, "marks", max_gap=120.0)
        assert [(p, e["type"]) for p, e in events] == [(0.0, "mark"), (0.0, "era"), (1.0, "mark")]

    def test_last_frame_is_the_finished_piece(self, tmp_path):
        r = _load_renderer()
        replay, canvas = self._replay(tmp_path)
        records = list(iter_records(replay))
        frames = list(r.render_frames(records, seconds=1, fps=10, clock="wall",
                                      max_gap=120, scale=1, strip=False, hold=0))
        assert len(frames) == 10
        last = frames[-1].load()
        for (x, y), c in canvas.pixels.items():
            assert last[x, y] == c
        first = frames[0].load()
        assert sum(first[x, y] != (0, 0, 0) for (x, y) in canvas.pixels) <= 2

    def test_gif_and_frames_outputs(self, tmp_path):
        r = _load_renderer()
        replay, _ = self._replay(tmp_path)
        gif = tmp_path / "out.gif"
        assert r.main([str(replay), "-o", str(gif), "--seconds", "1", "--fps", "5", "--hold", "0"]) == 0
        from PIL import Image
        with Image.open(gif) as im:
            assert im.n_frames >= 2
            assert im.size[1] > im.size[0]  # strip under the canvas
        frames = tmp_path / "frames"
        assert r.main([str(replay), "-o", str(frames), "--seconds", "1", "--fps", "4",
                       "--hold", "0", "--no-strip", "--scale", "1"]) == 0
        assert len(list(frames.glob("frame_*.png"))) == 4

    def test_empty_log_refuses(self, tmp_path):
        r = _load_renderer()
        p = tmp_path / "empty.replay.jsonl"
        p.write_text(json.dumps({"type": "header", "v": 1}) + "\n")
        assert r.main([str(p), "-o", str(tmp_path / "x.gif")]) == 1
