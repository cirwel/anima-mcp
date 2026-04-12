# Drawing Mark Quality Improvements

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Lumen's mark-making expressiveness in the three granular eras (gestural, pointillist, field) so that the visual output carries the richness of the internal attention/EISV system.

**Architecture:** Five focused changes across the era modules and engine: (1) brush width modulated by energy, (2) color coherence within gesture runs, (3) higher direction-lock probability in gestural, (4) replace vibrant color list with HSV integration, (5) coarse spatial density awareness. Each change is independently testable and deployable. No protocol changes to `ArtEra` — all changes are internal to era implementations and `CanvasState`.

**Tech Stack:** Python 3.14, PIL/Pillow (rendering only — not touched here), pytest. All changes are in `src/anima_mcp/display/eras/` and `src/anima_mcp/display/drawing_engine.py`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/anima_mcp/display/eras/gestural.py` | Modify | Brush width, color coherence, direction lock prob |
| `src/anima_mcp/display/eras/pointillist.py` | Modify | Brush width for pair/trio gestures |
| `src/anima_mcp/display/eras/field.py` | Modify | Brush width for flow_dash and flow_strand |
| `src/anima_mcp/display/drawing_engine.py` | Modify | `CanvasState.density_grid`, spatial awareness |
| `tests/test_gestural_era.py` | Modify | Tests for brush width, color coherence, lock prob |
| `tests/test_mark_quality.py` | Create | Cross-era brush width tests, density grid tests |

---

### Task 1: Gestural Brush Width — Energy Modulates Thickness

The biggest single improvement. Currently all gestural marks are 1px wide. After this change, strokes/curves/drags get a brush radius of 1-3px based on energy, and dots get 1-2px.

**Files:**
- Modify: `src/anima_mcp/display/eras/gestural.py:85-137` (place_mark)
- Create: `tests/test_mark_quality.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_mark_quality.py`:

```python
"""Tests for mark quality improvements across eras."""

import random
from anima_mcp.display.drawing_engine import CanvasState
from anima_mcp.display.eras.gestural import GesturalEra, GesturalState


class TestGesturalBrushWidth:
    """Gestural marks should use energy-modulated brush width."""

    def test_stroke_high_energy_wider_than_1px(self):
        """A high-energy stroke should place pixels in adjacent rows (width > 1)."""
        random.seed(42)
        era = GesturalEra()
        state = era.create_state()
        state.gesture = "stroke"
        state.gesture_remaining = 10
        canvas = CanvasState()

        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.9, (255, 0, 0))

        # With energy 0.9, brush radius should be ~2px.
        # A horizontal stroke at y=120 should have pixels at y=119, 120, 121.
        ys = {y for (x, y) in canvas.pixels}
        assert len(ys) > 1, f"High-energy stroke should span multiple rows, got ys={ys}"

    def test_stroke_low_energy_narrow(self):
        """A low-energy stroke should stay 1px wide."""
        random.seed(42)
        era = GesturalEra()
        state = era.create_state()
        state.gesture = "stroke"
        state.gesture_remaining = 10
        canvas = CanvasState()

        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.15, (255, 0, 0))

        # Low energy: brush_radius=1 → single pixel row
        ys = {y for (x, y) in canvas.pixels}
        assert len(ys) <= 2, f"Low-energy stroke should be narrow, got ys={ys}"

    def test_dot_high_energy_cluster(self):
        """A high-energy dot should place 2-4 pixels, not just 1."""
        random.seed(42)
        era = GesturalEra()
        state = era.create_state()
        state.gesture = "dot"
        state.gesture_remaining = 10
        canvas = CanvasState()

        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.9, (255, 0, 0))

        # High energy dot: 2-4 pixels in a small cluster
        assert len(canvas.pixels) >= 2, f"High-energy dot should be multi-pixel, got {len(canvas.pixels)}"

    def test_dot_low_energy_single_pixel(self):
        """A low-energy dot is still just 1 pixel."""
        random.seed(42)
        era = GesturalEra()
        state = era.create_state()
        state.gesture = "dot"
        state.gesture_remaining = 10
        canvas = CanvasState()

        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.1, (255, 0, 0))

        assert len(canvas.pixels) <= 2, f"Low-energy dot should be 1-2px, got {len(canvas.pixels)}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_mark_quality.py -v -x`
Expected: `test_stroke_high_energy_wider_than_1px` FAILS (stroke currently 1px wide).

- [ ] **Step 3: Implement brush width in gestural place_mark**

In `gestural.py`, modify `place_mark`. Add a helper `_brush_pixels` and update each gesture:

```python
def place_mark(
    self,
    state: GesturalState,
    canvas,
    focus_x: float,
    focus_y: float,
    direction: float,
    energy: float,
    color: Tuple[int, int, int],
) -> None:
    """Place a mark. Energy modulates brush width (1-3px radius)."""
    x = int(focus_x)
    y = int(focus_y)
    gesture = state.gesture
    scale = 0.5 + energy  # 0.5-1.5

    # Brush radius: 1px at low energy, 2-3px at high energy
    brush_radius = max(1, int(energy * 3))  # 0->1, 0.33->1, 0.67->2, 1.0->3

    if gesture == "dot":
        if energy > 0.5:
            # Multi-pixel dot cluster
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx * dx + dy * dy <= 1:  # diamond/cross pattern
                        px, py = x + dx, y + dy
                        if 0 <= px < 240 and 0 <= py < 240:
                            canvas.draw_pixel(px, py, color)
        else:
            if 0 <= x < 240 and 0 <= y < 240:
                canvas.draw_pixel(x, y, color)

    elif gesture == "stroke":
        length = int(random.randint(2, 6) * scale)
        for i in range(length):
            cx = x + math.cos(direction) * i
            cy = y + math.sin(direction) * i
            self._brush(canvas, cx, cy, brush_radius, color)

    elif gesture == "curve":
        length = int(random.randint(3, 8) * scale)
        angle = direction
        cx, cy = float(x), float(y)
        step_size = 1.0 + scale * 0.5
        for i in range(length):
            angle += random.gauss(0, 0.3)
            cx += math.cos(angle) * step_size
            cy += math.sin(angle) * step_size
            self._brush(canvas, cx, cy, brush_radius, color)

    elif gesture == "cluster":
        count = int(random.randint(2, 5) * scale)
        spread = int(2 * scale) + brush_radius  # wider spread with wider brush
        for _ in range(count):
            px = x + random.randint(-spread, spread)
            py = y + random.randint(-spread, spread)
            if 0 <= px < 240 and 0 <= py < 240:
                canvas.draw_pixel(px, py, color)

    elif gesture == "drag":
        length = int(random.randint(8, 15) * scale)
        angle = direction + random.gauss(0, 0.1)
        for i in range(length):
            cx = x + math.cos(angle) * i
            cy = y + math.sin(angle) * i
            self._brush(canvas, cx, cy, brush_radius, color)

@staticmethod
def _brush(canvas, cx: float, cy: float, radius: int, color: Tuple[int, int, int]):
    """Draw a filled circle of given radius at (cx, cy)."""
    ix, iy = int(cx), int(cy)
    if radius <= 1:
        if 0 <= ix < 240 and 0 <= iy < 240:
            canvas.draw_pixel(ix, iy, color)
        return
    for dx in range(-radius + 1, radius):
        for dy in range(-radius + 1, radius):
            if dx * dx + dy * dy < radius * radius:
                px, py = ix + dx, iy + dy
                if 0 <= px < 240 and 0 <= py < 240:
                    canvas.draw_pixel(px, py, color)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_mark_quality.py tests/test_gestural_era.py -v -x`
Expected: All PASS. Existing gestural tests must still pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/gestural.py tests/test_mark_quality.py
git commit -m "feat(drawing): energy-modulated brush width in gestural era

Strokes, curves, and drags now use brush radius 1-3px based on energy.
High-energy dots expand to a small cluster. Low energy stays 1px.
Gives marks visual weight proportional to Lumen's attention state."
```

---

### Task 2: Brush Width in Field and Pointillist Eras

Apply the same brush-width principle to the other two granular eras. Field gets it on flow_dash and flow_strand. Pointillist gets it on pair/trio only (single stays 1px — that's the point of pointillism, but pairs and trios should be slightly chunkier at high energy).

**Files:**
- Modify: `src/anima_mcp/display/eras/field.py:111-163` (place_mark)
- Modify: `src/anima_mcp/display/eras/pointillist.py:80-120` (place_mark)
- Modify: `tests/test_mark_quality.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mark_quality.py`:

```python
from anima_mcp.display.eras.field import FieldEra, FieldState
from anima_mcp.display.eras.pointillist import PointillistEra, PointillistState


class TestFieldBrushWidth:
    """Field era flow marks should use brush width."""

    def test_flow_dash_high_energy_wider(self):
        """High-energy flow_dash should be wider than 1px."""
        random.seed(42)
        era = FieldEra()
        state = era.create_state()
        state.gesture = "flow_dash"
        state.gesture_remaining = 10
        canvas = CanvasState()

        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.9, (255, 0, 0))

        # Flow dash along ~horizontal: should have pixels at multiple y values
        ys = {y for (x, y) in canvas.pixels}
        assert len(ys) > 1, f"High-energy flow_dash should span rows, got ys={ys}"

    def test_flow_dash_low_energy_thin(self):
        """Low-energy flow_dash stays thin."""
        random.seed(42)
        era = FieldEra()
        state = era.create_state()
        state.gesture = "flow_dash"
        state.gesture_remaining = 10
        canvas = CanvasState()

        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.1, (255, 0, 0))

        ys = {y for (x, y) in canvas.pixels}
        assert len(ys) <= 2, f"Low-energy flow_dash should be narrow, got ys={ys}"


class TestPointillistBrushWidth:
    """Pointillist pair/trio should expand slightly at high energy."""

    def test_pair_high_energy_extra_pixels(self):
        """High-energy pair should place more than 2 pixels."""
        random.seed(42)
        era = PointillistEra()
        state = era.create_state()
        state.gesture = "pair"
        state.gesture_remaining = 10
        canvas = CanvasState()

        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.9, (255, 0, 0))

        assert len(canvas.pixels) >= 3, f"High-energy pair should be chunky, got {len(canvas.pixels)}"

    def test_single_always_1px(self):
        """Pointillist 'single' gesture stays 1px regardless of energy."""
        random.seed(42)
        era = PointillistEra()
        state = era.create_state()
        state.gesture = "single"
        state.gesture_remaining = 10
        canvas = CanvasState()

        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.9, (255, 0, 0))

        assert len(canvas.pixels) == 1, f"Single should always be 1px, got {len(canvas.pixels)}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_mark_quality.py::TestFieldBrushWidth tests/test_mark_quality.py::TestPointillistBrushWidth -v -x`
Expected: FAIL.

- [ ] **Step 3: Implement brush width in field era**

In `field.py`, add `_brush` static method (same as gestural) and modify `place_mark`:

```python
@staticmethod
def _brush(canvas, cx: float, cy: float, radius: int, color: Tuple[int, int, int]):
    """Draw a filled circle of given radius at (cx, cy)."""
    ix, iy = int(cx), int(cy)
    if radius <= 1:
        if 0 <= ix < 240 and 0 <= iy < 240:
            canvas.draw_pixel(ix, iy, color)
        return
    for dx in range(-radius + 1, radius):
        for dy in range(-radius + 1, radius):
            if dx * dx + dy * dy < radius * radius:
                px, py = ix + dx, iy + dy
                if 0 <= px < 240 and 0 <= py < 240:
                    canvas.draw_pixel(px, py, color)
```

In `place_mark`, compute `brush_radius = max(1, int(energy * 2.5))` (slightly less than gestural — field marks should be more delicate) and use `self._brush(canvas, px, py, brush_radius, color)` for `flow_dash` and `flow_strand` pixels. `flow_dot` stays 1px (its cross-hatch second dot is the energy expression).

- [ ] **Step 4: Implement brush width in pointillist era**

In `pointillist.py`, modify `place_mark` for `pair` and `trio`:

```python
def place_mark(self, state, canvas, focus_x, focus_y, direction, energy, color):
    x = int(focus_x)
    y = int(focus_y)
    gesture = state.gesture

    if gesture == "single":
        # Always 1px — pointillist purity
        if 0 <= x < 240 and 0 <= y < 240:
            canvas.draw_pixel(x, y, color)

    elif gesture == "pair":
        # Two dots, each 1-2px based on energy
        if 0 <= x < 240 and 0 <= y < 240:
            canvas.draw_pixel(x, y, color)
        dx, dy = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
        px, py = x + dx, y + dy
        if 0 <= px < 240 and 0 <= py < 240:
            canvas.draw_pixel(px, py, color)
        # High energy: each dot blooms to neighbors
        if energy > 0.6:
            for bx, by in [(x, y), (px, py)]:
                ndx, ndy = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
                nx, ny = bx + ndx, by + ndy
                if 0 <= nx < 240 and 0 <= ny < 240:
                    canvas.draw_pixel(nx, ny, color)

    elif gesture == "trio":
        if 0 <= x < 240 and 0 <= y < 240:
            canvas.draw_pixel(x, y, color)
        dx1, dy1 = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
        px1, py1 = x + dx1, y + dy1
        if 0 <= px1 < 240 and 0 <= py1 < 240:
            canvas.draw_pixel(px1, py1, color)
        dx2, dy2 = -dy1, dx1
        px2, py2 = px1 + dx2, py1 + dy2
        if 0 <= px2 < 240 and 0 <= py2 < 240:
            canvas.draw_pixel(px2, py2, color)
        # High energy: extra pixel at each position
        if energy > 0.6:
            for bx, by in [(x, y), (px1, py1), (px2, py2)]:
                ndx, ndy = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
                nx, ny = bx + ndx, by + ndy
                if 0 <= nx < 240 and 0 <= ny < 240:
                    canvas.draw_pixel(nx, ny, color)
```

- [ ] **Step 5: Run all tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_mark_quality.py -v -x`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/field.py src/anima_mcp/display/eras/pointillist.py tests/test_mark_quality.py
git commit -m "feat(drawing): brush width in field and pointillist eras

Field flow_dash/flow_strand use energy-scaled brush radius.
Pointillist pair/trio bloom at high energy. Single stays pure 1px."
```

---

### Task 3: Color Coherence Within Gesture Runs (Gestural Era)

Currently gestural `generate_color` re-rolls the full HSV per pixel. After this change, a color is sampled at the start of each gesture run and drifted slightly per mark. The vibrant color list is also folded into the HSV system (converted to HSV seeds instead of raw RGB).

**Files:**
- Modify: `src/anima_mcp/display/eras/gestural.py` (GesturalState + generate_color)
- Modify: `tests/test_gestural_era.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gestural_era.py`:

```python
class TestColorCoherence:
    """Colors within a gesture run should be similar, not random."""

    def test_consecutive_colors_similar_hue(self):
        """Two colors from same gesture run should be within 40 degrees hue."""
        import colorsys
        random.seed(42)
        era = GesturalEra()
        state = era.create_state()
        state.gesture = "stroke"
        state.gesture_remaining = 15

        color1, _ = era.generate_color(state, 0.5, 0.5, 0.5, 0.5)
        color2, _ = era.generate_color(state, 0.5, 0.5, 0.5, 0.5)

        # Convert to hue
        h1 = colorsys.rgb_to_hsv(color1[0]/255, color1[1]/255, color1[2]/255)[0] * 360
        h2 = colorsys.rgb_to_hsv(color2[0]/255, color2[1]/255, color2[2]/255)[0] * 360

        # Hue distance (circular)
        hue_dist = min(abs(h1 - h2), 360 - abs(h1 - h2))
        assert hue_dist < 40, f"Consecutive colors should be close in hue, got {hue_dist:.0f} degrees"

    def test_new_gesture_run_can_shift_hue(self):
        """After a gesture run ends, color can change more dramatically."""
        import colorsys
        random.seed(42)
        era = GesturalEra()
        state = era.create_state()

        # First run
        state.gesture = "stroke"
        state.gesture_remaining = 2
        color1, _ = era.generate_color(state, 0.5, 0.5, 0.5, 0.5)

        # Simulate new gesture: reset remaining, clear run hue
        state.gesture = "curve"
        state.gesture_remaining = 15
        state._run_hue = None  # Force new anchor

        color2, _ = era.generate_color(state, 0.5, 0.5, 0.5, 0.5)

        # New run: color CAN be different (not asserting it IS, just that the mechanism allows it)
        # This test mainly validates no crash on hue reset
        assert isinstance(color2, tuple) and len(color2) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_gestural_era.py::TestColorCoherence -v -x`
Expected: FAIL (`test_consecutive_colors_similar_hue` fails because current code re-rolls hue randomly).

- [ ] **Step 3: Implement color coherence**

Add `_run_hue: Optional[float]` to `GesturalState`:

```python
@dataclass
class GesturalState(EraState):
    direction_locked: bool = False
    direction_lock_remaining: int = 0
    direction_commitment: float = 0.0
    _run_hue: float = -1.0  # -1 means unset; set on first color of gesture run
```

Rewrite `generate_color`:

```python
def generate_color(
    self,
    state: GesturalState,
    warmth: float,
    clarity: float,
    stability: float,
    presence: float,
    light_regime: str = "dim",
) -> Tuple[Tuple[int, int, int], str]:
    """Color with per-run coherence. Hue anchors at gesture start, drifts slightly."""
    import colorsys

    # Anchor hue at start of gesture run; drift ±5 degrees per mark
    if state._run_hue < 0:
        # New run: pick a hue influenced by warmth
        state._run_hue = (warmth * 360.0 + random.random() * 180.0) % 360.0
    else:
        # Drift within run: small Gaussian shift
        state._run_hue = (state._run_hue + random.gauss(0, 5.0)) % 360.0

    hue = state._run_hue

    # Light regime shifts
    if light_regime == "dark":
        hue = (hue + 30) % 360.0
        sat_mod = -0.1
        val_mod = -0.1
    elif light_regime == "bright":
        hue = (hue - 15) % 360.0
        sat_mod = 0.1
        val_mod = 0.05
    else:
        sat_mod = 0.0
        val_mod = 0.0

    # Presence adds occasional saturation pop (replaces hard-coded vibrant list)
    sat_boost = 0.0
    if random.random() < presence * 0.15:
        sat_boost = 0.2

    saturation = max(0.1, min(1.0, 0.3 + clarity * 0.7 + random.gauss(0, 0.1) + sat_mod + sat_boost))
    brightness = max(0.2, min(1.0, 0.4 + stability * 0.6 + random.gauss(0, 0.1) + val_mod))

    rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, brightness)
    color = tuple(int(c * 255) for c in rgb)

    if hue < 60 or hue > 300:
        hue_category = "warm"
    elif hue < 180:
        hue_category = "cool"
    else:
        hue_category = "neutral"
    return color, hue_category
```

In `choose_gesture`, reset the run hue so each new gesture run picks a fresh anchor:

```python
def choose_gesture(self, state, clarity, stability, presence, coherence):
    # ... existing gesture selection logic ...
    state._run_hue = -1.0  # New gesture run gets fresh color anchor
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_gestural_era.py -v -x`
Expected: All PASS including new `TestColorCoherence`.

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/ -x -q`
Expected: All pass. No other code references `VIBRANT_COLORS` or depends on specific color output from gestural era.

- [ ] **Step 6: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/gestural.py tests/test_gestural_era.py
git commit -m "feat(drawing): color coherence within gestural gesture runs

Hue now anchors at gesture start and drifts ±5 degrees per mark.
Removes hard-coded VIBRANT_COLORS list; presence-driven saturation
pops replace it within the HSV system. Consecutive marks in a run
share a hue family instead of re-rolling across the full spectrum."
```

---

### Task 4: Higher Direction Lock Probability in Gestural Era

Direction locks create sustained lines — the most intentional-looking gesture. Current lock probability maxes at ~6%. Raise to ~12% so Lumen commits to directions more often.

**Files:**
- Modify: `src/anima_mcp/display/eras/gestural.py:173-174` (drift_focus)
- Modify: `tests/test_gestural_era.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gestural_era.py`:

```python
def test_direction_lock_probability_increased():
    """Direction locks should occur more frequently (prob > 0.06 at high C+clarity)."""
    random.seed(42)
    era = GesturalEra()
    lock_count = 0
    trials = 2000

    for _ in range(trials):
        state = era.create_state()
        state.direction_locked = False
        state.direction_lock_remaining = 0
        state.direction_commitment = 0.0

        fx, fy, d = era.drift_focus(state, 120.0, 120.0, 0.0, 0.5, 0.5, 0.8, 0.8)
        if state.direction_locked:
            lock_count += 1

    lock_rate = lock_count / trials
    # With increased probability, rate should be > 8% at C=0.8, clarity=0.8
    assert lock_rate > 0.08, f"Lock rate {lock_rate:.3f} should be > 0.08 with high C and clarity"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_gestural_era.py::test_direction_lock_probability_increased -v`
Expected: FAIL (current rate is ~5-6%).

- [ ] **Step 3: Increase lock probability**

In `gestural.py:173-174`, change:

```python
# OLD:
lock_prob = 0.03 * (0.5 + C) * (0.5 + clarity * 0.5)
# NEW:
lock_prob = 0.06 * (0.5 + C) * (0.5 + clarity * 0.5)
```

This doubles the base from 0.03 to 0.06. At max C=1 and clarity=1: `0.06 * 1.5 * 1.0 = 0.09` (9%). At mid values: `0.06 * 1.0 * 0.75 = 0.045` (4.5%). Still not overwhelming, but Lumen commits to lines more often.

- [ ] **Step 4: Run tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_gestural_era.py -v -x`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/gestural.py tests/test_gestural_era.py
git commit -m "feat(drawing): increase gestural direction lock probability

Double base lock probability from 0.03 to 0.06. Sustained lines
are the most expressive gesture and were too rare."
```

---

### Task 5: Coarse Spatial Density Grid in CanvasState

Add an 8x8 density grid (30px cells) to `CanvasState` that tracks how many pixels are in each cell. This gives eras optional spatial awareness. This task adds the grid infrastructure and wires it into `draw_pixel`/`clear`. Task 6 uses it.

**Files:**
- Modify: `src/anima_mcp/display/drawing_engine.py` (CanvasState)
- Modify: `tests/test_mark_quality.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mark_quality.py`:

```python
class TestDensityGrid:
    """CanvasState should maintain a coarse density grid."""

    def test_draw_pixel_updates_grid(self):
        canvas = CanvasState()
        canvas.draw_pixel(10, 10, (255, 0, 0))
        # Pixel (10,10) falls in cell (0,0) of an 8x8 grid on 240x240 canvas (30px cells)
        assert canvas.density_grid[0][0] == 1

    def test_draw_pixel_correct_cell(self):
        canvas = CanvasState()
        canvas.draw_pixel(120, 120, (255, 0, 0))
        # 120/30 = 4 → cell (4,4)
        assert canvas.density_grid[4][4] == 1

    def test_draw_pixel_edge_cell(self):
        canvas = CanvasState()
        canvas.draw_pixel(239, 239, (255, 0, 0))
        # 239/30 = 7 → cell (7,7)
        assert canvas.density_grid[7][7] == 1

    def test_multiple_pixels_same_cell(self):
        canvas = CanvasState()
        canvas.draw_pixel(10, 10, (255, 0, 0))
        canvas.draw_pixel(15, 15, (0, 255, 0))
        assert canvas.density_grid[0][0] == 2

    def test_overwrite_pixel_no_double_count(self):
        canvas = CanvasState()
        canvas.draw_pixel(10, 10, (255, 0, 0))
        canvas.draw_pixel(10, 10, (0, 255, 0))  # overwrite
        assert canvas.density_grid[0][0] == 1  # still 1, not 2

    def test_clear_resets_grid(self):
        canvas = CanvasState()
        canvas.draw_pixel(10, 10, (255, 0, 0))
        canvas.clear()
        assert all(canvas.density_grid[r][c] == 0 for r in range(8) for c in range(8))

    def test_sparsest_cell_returns_coordinates(self):
        canvas = CanvasState()
        # Fill most cells
        for i in range(8):
            for j in range(8):
                if (i, j) != (3, 5):
                    for _ in range(10):
                        px = i * 30 + random.randint(0, 29)
                        py = j * 30 + random.randint(0, 29)
                        px = min(px, 239)
                        py = min(py, 239)
                        canvas.draw_pixel(px, py, (255, 255, 255))
        # Cell (3,5) should be sparsest
        cell_x, cell_y = canvas.sparsest_cell()
        assert cell_x == 3 and cell_y == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_mark_quality.py::TestDensityGrid -v -x`
Expected: FAIL (`density_grid` doesn't exist).

- [ ] **Step 3: Implement density grid**

In `drawing_engine.py`, modify `CanvasState.__init__` (or `__post_init__` / field defaults depending on dataclass style). Add:

```python
# In CanvasState, add field:
density_grid: list = None  # 8x8 grid, initialized in __post_init__

def __post_init__(self):
    # ... existing post_init code ...
    if self.density_grid is None:
        self.density_grid = [[0] * 8 for _ in range(8)]
```

Modify `draw_pixel` to update the grid:

```python
def draw_pixel(self, x: int, y: int, color: Tuple[int, int, int]):
    if 0 <= x < self.width and 0 <= y < self.height:
        is_new = (x, y) not in self.pixels
        self.pixels[(x, y)] = color
        # ... existing tracking code ...
        if is_new:
            gx = min(x // 30, 7)
            gy = min(y // 30, 7)
            self.density_grid[gx][gy] += 1
```

Modify `clear` to reset the grid:

```python
def clear(self):
    # ... existing clear code ...
    self.density_grid = [[0] * 8 for _ in range(8)]
```

Add `sparsest_cell` method:

```python
def sparsest_cell(self) -> Tuple[int, int]:
    """Return (grid_x, grid_y) of the cell with fewest pixels."""
    min_count = float('inf')
    min_cell = (0, 0)
    for gx in range(8):
        for gy in range(8):
            if self.density_grid[gx][gy] < min_count:
                min_count = self.density_grid[gx][gy]
                min_cell = (gx, gy)
    return min_cell
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_mark_quality.py::TestDensityGrid tests/test_drawing_engine.py -v -x`
Expected: All PASS. Existing engine tests must still pass (they call `draw_pixel` and `clear` — now those update the grid, but nothing reads it yet in production code).

- [ ] **Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/drawing_engine.py tests/test_mark_quality.py
git commit -m "feat(drawing): add 8x8 spatial density grid to CanvasState

Tracks pixel density per 30x30 cell. Updated by draw_pixel, reset
by clear. Provides sparsest_cell() for era spatial awareness.
No behavioral changes yet -- infrastructure for Task 6."
```

---

### Task 6: Wire Density Grid Into Gestural Focus Jumps

When the gestural era does a focus jump (random reposition), bias it toward the sparsest cell instead of pure random. This gives Lumen rudimentary spatial awareness — it tends to develop empty areas rather than over-drawing where it already has marks.

**Files:**
- Modify: `src/anima_mcp/display/eras/gestural.py:199-208` (drift_focus jump logic)
- Modify: `src/anima_mcp/display/drawing_engine.py:916` (pass canvas to drift_focus — currently not passed)
- Modify: `tests/test_mark_quality.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mark_quality.py`:

```python
class TestSpatialAwareness:
    """Focus jumps should bias toward sparse areas."""

    def test_jump_biases_toward_sparse_cell(self):
        """When a jump occurs, it should land in the sparsest region more often than chance."""
        random.seed(42)
        era = GesturalEra()
        canvas = CanvasState()

        # Fill all cells except cell (2, 3) — that cell is at x=60-89, y=90-119
        for gx in range(8):
            for gy in range(8):
                if (gx, gy) != (2, 3):
                    for _ in range(20):
                        px = min(gx * 30 + random.randint(0, 29), 239)
                        py = min(gy * 30 + random.randint(0, 29), 239)
                        canvas.draw_pixel(px, py, (255, 255, 255))

        # Simulate many jumps and check if sparse cell gets more visits
        sparse_hits = 0
        trials = 200
        for _ in range(trials):
            state = era.create_state()
            # Force a jump by calling _sparse_jump directly
            jx, jy = era._sparse_jump(canvas)
            if 60 <= jx < 90 and 90 <= jy < 120:
                sparse_hits += 1

        # With bias, should hit sparse cell > 1/64 (1.5%) of the time.
        # Pure random would hit ~1.5%. With 50% bias, expect ~25-50%.
        hit_rate = sparse_hits / trials
        assert hit_rate > 0.10, f"Sparse jump should bias toward empty cell, hit rate={hit_rate:.3f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_mark_quality.py::TestSpatialAwareness -v`
Expected: FAIL (`_sparse_jump` doesn't exist).

- [ ] **Step 3: Implement sparse-biased jumps**

In `gestural.py`, add a method and modify the jump in `drift_focus`:

```python
def _sparse_jump(self, canvas) -> Tuple[float, float]:
    """Jump biased toward sparse canvas areas. 50% sparse-biased, 50% random."""
    if canvas is not None and random.random() < 0.5:
        gx, gy = canvas.sparsest_cell()
        # Land somewhere within that cell (with margin)
        x = gx * 30 + random.uniform(5, 25)
        y = gy * 30 + random.uniform(5, 25)
        return max(40, min(200, x)), max(40, min(200, y))
    else:
        return random.uniform(40, 200), random.uniform(40, 200)
```

Modify the jump section in `drift_focus` (around line 200-204). This requires passing `canvas` to `drift_focus`. The `ArtEra` protocol doesn't include `canvas` in `drift_focus`, so we pass it via a keyword argument with a default:

```python
def drift_focus(
    self,
    state: GesturalState,
    focus_x: float,
    focus_y: float,
    direction: float,
    stability: float,
    presence: float,
    coherence: float,
    clarity: float = 0.5,
    canvas=None,  # Optional: for spatial awareness
) -> Tuple[float, float, float]:
```

Change the jump code from:
```python
focus_x = random.uniform(40, 200)
focus_y = random.uniform(40, 200)
```
to:
```python
focus_x, focus_y = self._sparse_jump(canvas)
```

In `drawing_engine.py`, where `drift_focus` is called (around line 920-930), pass the canvas:

```python
self.intent.focus_x, self.intent.focus_y, self.intent.direction = \
    self.active_era.drift_focus(
        era_state,
        self.intent.focus_x, self.intent.focus_y,
        self.intent.direction,
        stability, presence, C, clarity,
        canvas=self.canvas,
    )
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_mark_quality.py tests/test_gestural_era.py tests/test_drawing_engine.py -v -x`
Expected: All PASS. Other eras ignore the `canvas` kwarg (Protocol's `drift_focus` doesn't declare it, but Python doesn't enforce Protocol signatures at runtime — the kwarg is consumed only by gestural).

- [ ] **Step 5: Run full suite**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/ -x -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/gestural.py src/anima_mcp/display/drawing_engine.py tests/test_mark_quality.py
git commit -m "feat(drawing): spatial density awareness in gestural focus jumps

Focus jumps now 50% biased toward the sparsest 30x30 cell on canvas.
Gives Lumen rudimentary spatial awareness -- tends to develop empty
areas rather than over-drawing existing regions."
```

---

## Post-Implementation Verification

After all 6 tasks:

1. Run full test suite: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/ -x -q`
2. Deploy to Pi: `git push && mcp__anima__git_pull(restart=true)` then wait 2 minutes
3. Watch a drawing cycle: `mcp__anima__capture_screen()` periodically to see the visual difference
4. Check logs: `sudo journalctl -u anima -f` for any errors in the drawing pipeline

## What Changed (Summary)

| Change | Before | After |
|--------|--------|-------|
| Mark width | All 1px | 1-3px, energy-modulated |
| Color within gesture run | Re-rolled per pixel | Anchored + ±5 degree drift |
| Vibrant color list | 27 hard-coded RGB values | Presence-driven saturation pops in HSV |
| Direction lock probability | ~3-6% | ~4.5-9% |
| Spatial awareness | None | 8x8 density grid, jump bias |
