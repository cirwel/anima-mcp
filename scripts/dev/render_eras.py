#!/usr/bin/env python3
"""Headless contact-sheet renderer for Lumen's art eras.

    python3 scripts/dev/render_eras.py . --out /tmp/eras [--eras a,b] [--pieces 12]
                                         [--seed 0] [--tick-seconds 1.2] [--light diurnal]

Look at the art before changing it, and again after. Added 2026-11-12: until
then no one had rendered the eras, and two of them (gestural, pointillist) had
been producing the same picture every piece for months while their tests
passed. To compare a change, render the committed tree from
`git archive <rev> src | tar -x -C <dir>` and the working tree with the same
--seed, then read the two sheets side by side.

<source-root> is either a repo root (containing src/anima_mcp) or the src dir
itself. Nothing is imported from anywhere else; the script asserts that.

WHAT IS REAL
  The real DrawingEngine drives every piece: DrawingEngine.draw() (draw-chance
  gate, drives, goal warmth bias, era create_state / choose_gesture /
  generate_color / place_mark / drift_focus, coverage bias, EISV step,
  attention, coherence, narrative arc) and DrawingEngine.canvas_check_autonomy()
  (novelty-settling tracker, false starts, era earned completion, every
  completion_reason incl. the 8h cap) and DrawingEngine.canvas_clear() (banks
  the disposition, draw_distinct against the last 6, DrawingGoal.from_state).
  Pieces are produced back to back in one engine, as on the device, so the
  novelty loop between consecutive pieces is exercised.

WHERE IT DEPARTS FROM THE DEVICE (all stubs are in _install_stubs / main)
  * time.time() is a simulated clock; one loop tick = --tick-seconds of it.
    The device's real draw() cadence is not known exactly; 1.2 s/tick was
    chosen because it reproduces the live median marks per piece.
  * canvas_save() is replaced: it snapshots the pixels instead of writing a
    PNG, and does NOT call growth, the message board or UNITARES.
  * canvas persistence (canvas.json) is a no-op; HOME points at a sandbox.
  * _check_lumen_said_finished() is False (no message board).
  * Light: SHM is stubbed with a "ready_shadow" residual following a simple
    day/night schedule (or a fixed regime with --light).
  * Anima is a slow bounded random walk inside the live ranges, not sensors.
  * Inner-life drives are 0 (set_drives never called); no governance pauses;
    no restarts mid-piece (so no resonance-field ghost, no resumed pieces).
  * calibration is empty (drawing_thresholds {}), i.e. built-in pivot 0.4 and
    coverage cuts 0.30/0.70.

OUTPUT (all under --out, default <script dir>/out)
  sheet_<era>[_tag].png      contact sheet, pieces in the order they were made
  zoom_<era>[_tag]/*.png     each piece alone, nearest-neighbour upscaled
  summary[_tag].json         per-piece and per-era numbers. Read-only extras the
                             harness measures (nothing here feeds the engine):
                             picture statistics, gestures per piece, share of
                             marks placed with the focus on an edge clamp,
                             behavioural coherence at every mark.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import random
import sys
import time as _time_mod
from pathlib import Path


# --------------------------------------------------------------------------- #
# Simulated clock
# --------------------------------------------------------------------------- #
class Clock:
    def __init__(self, start: float):
        self.t = start

    def now(self) -> float:
        return self.t


def _lux_for(mode: str, hour: float):
    """Gated external residual lux for a regime mode. None => "unknown"."""
    if mode == "bright":
        return 226.0
    if mode == "dim":
        return 40.0
    if mode == "dark":
        return 2.0
    if mode == "unknown":
        return None
    # diurnal: bright by day, dim in the evening/early morning, dark at night
    if 8 <= hour < 18:
        return 226.0
    if 18 <= hour < 23 or 6 <= hour < 8:
        return 40.0
    return 2.0


class AnimaWalk:
    """Slow bounded walk inside Lumen's live ranges. Own RNG, so it never
    perturbs the `random` stream the engine and eras draw from."""

    RANGES = {  # (low, high, per-tick sigma)
        "clarity": (0.45, 0.90, 0.0020),
        "stability": (0.62, 0.78, 0.0010),
        "presence": (0.700, 0.720, 0.0002),
        "warmth": (0.35, 0.50, 0.0006),
    }

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.v = {k: self.rng.uniform(lo, hi) for k, (lo, hi, _) in self.RANGES.items()}

    def step(self):
        for k, (lo, hi, s) in self.RANGES.items():
            x = self.v[k] + self.rng.gauss(0, s)
            if x < lo:
                x = 2 * lo - x
            if x > hi:
                x = 2 * hi - x
            self.v[k] = min(hi, max(lo, x))
        return self.v


# --------------------------------------------------------------------------- #
# Picture statistics (for judging sameness / composition, not for the device)
# --------------------------------------------------------------------------- #
def picture_stats(pixels: dict) -> dict:
    n = len(pixels)
    if not n:
        return {"pixels": 0}
    grid = [[0] * 8 for _ in range(8)]
    fine = [[0] * 16 for _ in range(16)]
    sx = sy = 0
    hue_bins = [0] * 12
    sat_sum = val_sum = 0.0
    for (x, y), (r, g, b) in pixels.items():
        grid[min(x // 30, 7)][min(y // 30, 7)] += 1
        fine[min(x // 15, 15)][min(y // 15, 15)] += 1
        sx += x
        sy += y
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        hue_bins[int(h * 12) % 12] += 1
        sat_sum += s
        val_sum += v
    flat = sorted(c for row in grid for c in row)
    total = sum(flat)
    # Gini of the 8x8 grid: 0 = perfectly even all-over, 1 = one cell
    cum = 0.0
    for i, c in enumerate(flat, 1):
        cum += i * c
    gini = (2 * cum) / (64 * total) - (64 + 1) / 64
    ent = 0.0
    for c in flat:
        if c:
            p = c / total
            ent -= p * math.log(p)
    hue_ent = 0.0
    for c in hue_bins:
        if c:
            p = c / n
            hue_ent -= p * math.log(p)
    cx, cy = sx / n, sy / n
    rms = math.sqrt(sum((x - cx) ** 2 + (y - cy) ** 2 for (x, y) in pixels) / n)
    centre = sum(1 for (x, y) in pixels if 60 <= x < 180 and 60 <= y < 180) / n
    return {
        "pixels": n,
        "coverage_pct": round(100 * n / 57600, 2),
        "occupied_cells_8x8": sum(1 for c in flat if c),
        "empty_blocks_16x16": sum(1 for row in fine for c in row if c == 0),
        "grid_entropy": round(ent / math.log(64), 3),
        "grid_gini": round(gini, 3),
        "max_cell_share": round(flat[-1] / total, 3),
        "centroid": [round(cx, 1), round(cy, 1)],
        "rms_radius": round(rms, 1),
        "centre_quarter_share": round(centre, 3),  # 0.25 = uniform
        "hue_entropy": round(hue_ent / math.log(12), 3),
        "dominant_hue_bin": max(range(12), key=lambda i: hue_bins[i]) * 30,
        "hue_bins": hue_bins,
        "mean_sat": round(sat_sum / n, 3),
        "mean_val": round(val_sum / n, 3),
        "grid": grid,
    }


def grid_correlation(a, b) -> float:
    fa = [c for row in a for c in row]
    fb = [c for row in b for c in row]
    ma, mb = sum(fa) / 64, sum(fb) / 64
    num = sum((x - ma) * (y - mb) for x, y in zip(fa, fb))
    da = math.sqrt(sum((x - ma) ** 2 for x in fa))
    db = math.sqrt(sum((y - mb) ** 2 for y in fb))
    return num / (da * db) if da and db else 0.0


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def piece_image(pixels: dict, scale: int = 1):
    from PIL import Image
    img = Image.new("RGB", (240, 240), (0, 0, 0))
    px = img.load()
    for (x, y), c in pixels.items():
        if 0 <= x < 240 and 0 <= y < 240:
            px[x, y] = tuple(int(v) for v in c)
    if scale != 1:
        img = img.resize((240 * scale, 240 * scale), Image.NEAREST)
    return img


def contact_sheet(era: str, pieces: list, cols: int, scale: int, path: Path, title: str):
    from PIL import Image, ImageDraw
    cell = 240 * scale
    pad, label_h, head = 8, 26, 18
    rows = math.ceil(len(pieces) / cols)
    W = pad + cols * (cell + pad)
    H = head + pad + rows * (cell + label_h + pad)
    sheet = Image.new("RGB", (W, H), (38, 38, 42))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 4), title, fill=(230, 230, 230))
    for i, p in enumerate(pieces):
        r, c = divmod(i, cols)
        x0 = pad + c * (cell + pad)
        y0 = head + pad + r * (cell + label_h + pad)
        sheet.paste(piece_image(p["pixels"], scale), (x0, y0))
        s = p["stats"]
        d.text((x0, y0 + cell + 1),
               f"#{i} {p['marks']}m {s['pixels']}px {p['hours']:.1f}h {p['reason']}",
               fill=(210, 210, 210))
        d.text((x0, y0 + cell + 13),
               f"{p['goal']} | {p['disp_short']}", fill=(150, 150, 160))
    sheet.save(path)


def _disp_short(d: dict) -> str:
    if not d:
        return "-"
    out = []
    for k, v in d.items():
        if isinstance(v, float):
            v = round(v, 1)
        if isinstance(v, (list, tuple)):
            v = "+".join(str(x)[:6] for x in v)
        out.append(f"{k[:6]}={v}")
    return " ".join(out)[:60]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_root")
    ap.add_argument("--out", default=None, help="output dir (default: <script dir>/out)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    ap.add_argument("--eras", default="gestural,pointillist,field,resonance,geometric")
    ap.add_argument("--pieces", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tick-seconds", type=float, default=1.2)
    ap.add_argument("--check-every", type=int, default=25, help="ticks between canvas_check_autonomy calls")
    ap.add_argument("--light", default="diurnal", choices=["diurnal", "bright", "dim", "dark", "unknown"])
    ap.add_argument("--max-hours", type=float, default=9.0, help="harness safety cap per piece (engine cap is 8h)")
    ap.add_argument("--zoom", type=int, default=3, help="scale for individual piece PNGs (0 = skip)")
    ap.add_argument("--sheet-scale", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.source_root).resolve()
    src = root / "src" if (root / "src" / "anima_mcp").is_dir() else root
    if not (src / "anima_mcp").is_dir():
        sys.exit(f"no anima_mcp package under {root}")
    out = Path(args.out).resolve() if args.out else Path(__file__).resolve().parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    sandbox = out / "_sandbox_home"
    sandbox.mkdir(exist_ok=True)

    # Isolate: nothing may read or write the real ~/.anima, DB or calibration.
    os.environ["HOME"] = str(sandbox)
    for k in ("ANIMA_DB", "ANIMA_CONFIG", "UNITARES_URL"):
        os.environ.pop(k, None)
    sys.path.insert(0, str(src))

    import anima_mcp
    pkg = Path(anima_mcp.__file__).resolve()
    assert str(pkg).startswith(str(src)), f"imported {pkg}, expected under {src}"
    print(f"[harness] anima_mcp from {pkg}", flush=True)

    clock = Clock(1794290400.0)  # arbitrary fixed epoch
    _time_mod.time = clock.now  # every module calls time.time() through the module

    import anima_mcp.accessors as accessors
    import anima_mcp.display.drawing_engine as de
    from anima_mcp.anima import Anima
    from anima_mcp.display.eras import list_eras
    from datetime import datetime as _real_dt

    class _SimDatetime:
        @staticmethod
        def now():
            return _real_dt.fromtimestamp(clock.t)

    de.datetime = _SimDatetime  # canvas_clear reads the hour for the goal's quadrant

    def _sim_hour() -> float:
        d = _real_dt.fromtimestamp(clock.t)
        return d.hour + d.minute / 60.0

    def _shm():
        lux = _lux_for(args.light, _sim_hour())
        if lux is None:
            return {"light_attribution": {"status": "warming", "external_lux_residual": None}}
        return {"light_attribution": {"status": "ready_shadow", "external_lux_residual": lux}}

    accessors._get_last_shm_data = _shm

    summary = {"args": vars(args), "source": str(src), "eras": {}}
    wanted = [e.strip() for e in args.eras.split(",") if e.strip()]
    for era_name in wanted:
        if era_name not in list_eras():
            print(f"[harness] unknown era {era_name}, skipping")
            continue
        era_idx = sorted(list_eras()).index(era_name)
        base_seed = args.seed * 100003 + era_idx * 1009
        random.seed(base_seed)
        clock.t = 1794290400.0

        # Fresh sandbox per era so no canvas.json / mood state leaks between eras
        for f in (sandbox / ".anima").glob("*") if (sandbox / ".anima").exists() else []:
            if f.is_file():
                f.unlink()

        import io
        import contextlib
        quiet = io.StringIO()
        with contextlib.redirect_stderr(quiet):
            eng = de.DrawingEngine(db_path=None, identity_store=None)
        eng.canvas.save_to_disk = lambda: True            # no canvas.json writes
        eng._persist_canvas_progress = lambda *a, **k: None
        eng._check_lumen_said_finished = lambda: False     # no message board

        pieces: list = []
        gesture_counts: dict = {}
        mark_log: list = []
        sw = {"runs": 0, "name": 0, "last": None}
        c_log: list = []   # behavioural coherence at every mark (what drawing_trajectory samples)
        walk = AnimaWalk(base_seed + 7)

        def _save_stub(announce=False, manual=False, _eng=eng):
            """Stands in for canvas_save(): snapshot instead of PNG+growth+UNITARES."""
            c = _eng.canvas
            if not c.pixels:
                return None
            if not manual and len(c.pixels) < de.MIN_RECORDED_DRAWING_PIXELS:
                return None
            st = _eng.intent.state
            es = _eng.intent.era_state
            try:
                disp = es.disposition() if es is not None else {}
            except Exception:
                disp = {}
            started = c.last_clear_time or c.drawing_start_time
            goal = _eng.drawing_goal
            pieces.append({
                "pixels": dict(c.pixels),
                "marks": c.mark_count,
                "hours": (clock.t - started) / 3600.0,
                "reason": c.last_completion_reason or "untagged",
                "goal": (goal.description if goal else "no goal")
                        + (f" q{goal.initial_quadrant}" if goal and goal.initial_quadrant is not None else ""),
                "coverage_target": getattr(goal, "coverage_target", None),
                "disposition": disp,
                "disp_short": _disp_short(disp),
                "curiosity": round(st.curiosity, 3),
                "engagement": round(st.engagement, 3),
                "fatigue": round(st.fatigue, 3),
                "arc": st.arc_phase,
                "C": round(_eng._behavioral_C, 3),
                "gestures": dict(gesture_counts),
                # share of marks placed with the focus on an edge clamp (<=20px
                # from an edge), whole piece / last quarter of the piece
                "edge_mark_share": round(sum(mark_log) / max(1, len(mark_log)), 3),
                "edge_mark_share_last_quarter": round(
                    sum(mark_log[-max(1, len(mark_log) // 4):]) / max(1, len(mark_log) // 4), 3),
                "gesture_runs": sw["runs"],
                "gesture_name_switches": sw["name"],
                "mean_C": round(sum(c_log) / max(1, len(c_log)), 4),
                "median_C": round(sorted(c_log)[len(c_log) // 2], 4) if c_log else None,
                "stats": picture_stats(c.pixels),
            })
            gesture_counts.clear()
            mark_log.clear()
            c_log.clear()
            sw.update(runs=0, name=0, last=None)
            c.last_save_time = clock.t
            c.drawings_saved += 1
            c.consecutive_false_starts = 0
            # Re-seed so piece k is reproducible whatever came before it.
            random.seed(base_seed + len(pieces))
            return f"sim://{era_name}/{len(pieces) - 1}"

        eng.canvas_save = _save_stub

        with contextlib.redirect_stderr(quiet):
            res = eng.set_era(era_name, force_immediate=True)
            assert res.get("success"), res
            # First piece starts the way every later one does: through canvas_clear,
            # so it gets a DrawingGoal. Drop the throwaway state first so its
            # disposition is not banked as a phantom "previous piece".
            eng.intent.era_state = None
            v = walk.step()
            eng.last_anima = Anima(v["warmth"], v["clarity"], v["stability"], v["presence"], readings=None)
            eng.canvas.drawing_paused_until = 0.0
            eng.canvas_clear(persist=False, already_saved=True)

            tick = 0
            abandoned = 0
            forced = 0
            piece_start_tick = 0
            wall0 = _real_time()
            while len(pieces) < args.pieces:
                tick += 1
                clock.t += args.tick_seconds
                v = walk.step()
                anima = Anima(v["warmth"], v["clarity"], v["stability"], v["presence"], readings=None)
                n_before = len(pieces)
                if clock.t >= eng.canvas.drawing_paused_until and len(eng.canvas.pixels) <= 15000:
                    m0 = eng.intent.mark_count
                    fx0, fy0 = eng.intent.focus_x, eng.intent.focus_y
                    es0 = eng.intent.era_state
                    rem0 = es0.gesture_remaining if es0 is not None else 0
                    eng.draw(anima)
                    if eng.intent.mark_count != m0 and eng.intent.era_state is not None:
                        g = eng.intent.era_state.gesture
                        gesture_counts[g] = gesture_counts.get(g, 0) + 1
                        # runs = gesture runs begun (choose_gesture calls);
                        # name switches = what the engine's fatigue counts in a
                        # mark-by-mark era (gesture name differs from the last mark's)
                        sw["runs"] += rem0 <= 0
                        sw["name"] += sw["last"] is not None and sw["last"] != g
                        sw["last"] = g
                        # Harness-only instrumentation: was this mark placed with
                        # the focus on/outside an era's edge clamp (field/pointillist 15, others 20)?
                        on_edge = (min(fx0, fy0) <= 20.0 or max(fx0, fy0) >= 220.0)
                        mark_log.append(on_edge)
                        c_log.append(eng._behavioral_C)
                if tick % args.check_every == 0:
                    r = eng.canvas_check_autonomy(anima)
                    if r == "abandoned":
                        abandoned += 1
                        gesture_counts.clear()
                        mark_log.clear()
                        c_log.clear()
                        sw.update(runs=0, name=0, last=None)
                # Harness-only safety net; the engine's own 8h cap should fire first.
                if (len(pieces) == n_before
                        and (tick - piece_start_tick) * args.tick_seconds > args.max_hours * 3600):
                    forced += 1
                    eng.canvas.last_completion_reason = "HARNESS_FORCED"
                    eng.canvas.drawing_paused_until = 0.0
                    if eng.canvas_save():
                        eng.canvas_clear(persist=False, already_saved=True)
                    else:
                        eng.canvas.pixels[(0, 0)] = (1, 1, 1)
                if len(pieces) != n_before:
                    piece_start_tick = tick
        wall = _real_time() - wall0

        # ---- outputs ----
        tag = f"_{args.tag}" if args.tag else ""
        sheet_path = out / f"sheet_{era_name}{tag}.png"
        title = (f"{era_name} | {len(pieces)} consecutive pieces | seed {args.seed} | "
                 f"light={args.light} | tick={args.tick_seconds}s | src={src}")
        contact_sheet(era_name, pieces, args.cols, args.sheet_scale, sheet_path, title)
        zoom_paths = []
        if args.zoom:
            zdir = out / f"zoom_{era_name}{tag}"
            zdir.mkdir(exist_ok=True)
            for i, p in enumerate(pieces):
                zp = zdir / f"{era_name}_{i:02d}.png"
                piece_image(p["pixels"], args.zoom).save(zp)
                zoom_paths.append(str(zp))

        grids = [p["stats"]["grid"] for p in pieces]
        corrs = [grid_correlation(grids[i], grids[j])
                 for i in range(len(grids)) for j in range(i + 1, len(grids))]

        def _col(key):
            return [p["stats"][key] for p in pieces]

        def _ms(vals):
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1))
            return [round(m, 3), round(sd, 3), round(min(vals), 3), round(max(vals), 3)]

        era_summary = {
            "sheet": str(sheet_path),
            "zoom": zoom_paths,
            "abandoned_false_starts": abandoned,
            "harness_forced": forced,
            "wall_seconds": round(wall, 1),
            "reasons": {r: sum(1 for p in pieces if p["reason"] == r) for r in {p["reason"] for p in pieces}},
            "marks[mean,sd,min,max]": _ms([p["marks"] for p in pieces]),
            "hours[mean,sd,min,max]": _ms([p["hours"] for p in pieces]),
            "pixels[mean,sd,min,max]": _ms(_col("pixels")),
            "coverage_pct": _ms(_col("coverage_pct")),
            "occupied_cells_8x8": _ms(_col("occupied_cells_8x8")),
            "empty_blocks_16x16(of 256)": _ms(_col("empty_blocks_16x16")),
            "grid_entropy": _ms(_col("grid_entropy")),
            "grid_gini": _ms(_col("grid_gini")),
            "max_cell_share": _ms(_col("max_cell_share")),
            "centre_quarter_share(0.25=uniform)": _ms(_col("centre_quarter_share")),
            "rms_radius": _ms(_col("rms_radius")),
            "hue_entropy(1=all hues)": _ms(_col("hue_entropy")),
            "mean_sat": _ms(_col("mean_sat")),
            "mean_val": _ms(_col("mean_val")),
            "layout_corr_between_pieces[mean,sd,min,max]": _ms(corrs) if corrs else None,
            "edge_mark_share": _ms([p["edge_mark_share"] for p in pieces]),
            "edge_mark_share_last_quarter": _ms([p["edge_mark_share_last_quarter"] for p in pieces]),
            "gesture_runs": _ms([p["gesture_runs"] for p in pieces]),
            "gesture_name_switches": _ms([p["gesture_name_switches"] for p in pieces]),
            "mean_C_over_piece": _ms([p["mean_C"] for p in pieces]),
            "median_C_over_piece": _ms([p["median_C"] for p in pieces]),
            "final_C": _ms([p["C"] for p in pieces]),
            "final_curiosity": _ms([p["curiosity"] for p in pieces]),
            "final_fatigue": _ms([p["fatigue"] for p in pieces]),
            "pieces": [
                {k: v for k, v in p.items() if k not in ("pixels",)} for p in pieces
            ],
        }
        for p in era_summary["pieces"]:
            p["stats"] = {k: v for k, v in p["stats"].items() if k not in ("grid", "hue_bins")}
        summary["eras"][era_name] = era_summary

        print(f"\n=== {era_name}: {len(pieces)} pieces, {wall:.0f}s wall, "
              f"{abandoned} false starts, {forced} forced ===")
        for k, v in era_summary.items():
            if k in ("pieces", "zoom"):
                continue
            print(f"  {k}: {v}")
        for i, p in enumerate(pieces):
            s = p["stats"]
            print(f"  #{i:02d} marks={p['marks']:5d} px={s['pixels']:5d} h={p['hours']:.1f} "
                  f"{p['reason']:18s} cells={s['occupied_cells_8x8']:2d} gini={s['grid_gini']:.2f} "
                  f"ctr={s['centre_quarter_share']:.2f} hueH={s['hue_entropy']:.2f} "
                  f"C={p['C']:.2f} cur={p['curiosity']:.2f} fat={p['fatigue']:.2f} "
                  f"edge={p['edge_mark_share']:.2f}/{p['edge_mark_share_last_quarter']:.2f} "
                  f"g={p['gestures']} | {p['goal']} | {p['disp_short']}", flush=True)

    tag = f"_{args.tag}" if args.tag else ""
    (out / f"summary{tag}.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\n[harness] summary: {out / f'summary{tag}.json'}")


_real_time = _time_mod.perf_counter

if __name__ == "__main__":
    main()
