# Resonance Era Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fifth drawing era ("resonance") where marks interact with accumulated emotional history through a 48x48 memory field.

**Architecture:** Single new file `src/anima_mcp/display/eras/resonance.py` implements the `ArtEra` protocol. The memory field (deposit, decay, diffuse, gradient) is internal to this module. The era registry gets a `min_drawings` gating mechanism so Resonance unlocks after 50 completed drawings. Canvas persistence (`save_to_disk`/`load_from_disk`) is extended to serialize the memory field when the active era is resonance.

**Tech Stack:** Python 3, NumPy (already a dependency on Pi), colorsys (stdlib)

**Spec:** `docs/superpowers/specs/2026-04-13-resonance-era-design.md`

---

### Task 1: Memory Field Core — Deposit, Decay, Diffuse, Gradient

The memory field is the foundation everything else builds on. Test it in isolation before wiring into the era.

**Files:**
- Create: `src/anima_mcp/display/eras/resonance.py`
- Create: `tests/test_resonance_era.py`

- [ ] **Step 1: Write failing tests for memory field operations**

```python
"""Tests for resonance era — memory field core."""

import numpy as np

# We'll import these after creating them
from anima_mcp.display.eras.resonance import (
    _deposit,
    _decay,
    _diffuse,
    _gradient_at,
    FIELD_SIZE,
    DECAY_RATE,
)


class TestDeposit:
    def test_deposit_adds_value_at_position(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        _deposit(field, pixel_x=120, pixel_y=120, value=0.8)
        # pixel (120,120) maps to cell (24,24) in a 48x48 grid over 240x240 canvas
        assert field[24, 24] > 0.0

    def test_deposit_accumulates(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        _deposit(field, pixel_x=120, pixel_y=120, value=0.5)
        _deposit(field, pixel_x=120, pixel_y=120, value=0.3)
        assert abs(field[24, 24] - 0.8) < 0.01

    def test_deposit_clamps_to_canvas(self):
        """Pixels at edge of canvas should not crash."""
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        _deposit(field, pixel_x=0, pixel_y=0, value=0.5)
        assert field[0, 0] > 0.0
        _deposit(field, pixel_x=239, pixel_y=239, value=0.5)
        assert field[FIELD_SIZE - 1, FIELD_SIZE - 1] > 0.0


class TestDecay:
    def test_decay_reduces_field(self):
        field = np.ones((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        _decay(field)
        assert field[0, 0] < 1.0
        assert abs(field[0, 0] - DECAY_RATE) < 0.001

    def test_decay_preserves_zeros(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        _decay(field)
        assert field.sum() == 0.0


class TestDiffuse:
    def test_diffuse_spreads_point_source(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        field[24, 24] = 1.0
        diffused = _diffuse(field, sigma=1.0)
        # Center should decrease, neighbors should increase
        assert diffused[24, 24] < 1.0
        assert diffused[24, 25] > 0.0
        assert diffused[25, 24] > 0.0

    def test_diffuse_conserves_energy_approximately(self):
        """Total field energy should be roughly preserved (boundary effects aside)."""
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        field[24, 24] = 1.0
        before_sum = field.sum()
        diffused = _diffuse(field, sigma=1.0)
        after_sum = diffused.sum()
        assert abs(before_sum - after_sum) < 0.1  # Small boundary loss OK

    def test_higher_sigma_spreads_more(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        field[24, 24] = 1.0
        low_sigma = _diffuse(field.copy(), sigma=0.5)
        high_sigma = _diffuse(field.copy(), sigma=2.0)
        # Higher sigma -> lower peak at center
        assert high_sigma[24, 24] < low_sigma[24, 24]


class TestGradient:
    def test_gradient_zero_on_uniform_field(self):
        field = np.ones((FIELD_SIZE, FIELD_SIZE), dtype=np.float32) * 0.5
        gx, gy, mag = _gradient_at(field, 24, 24)
        assert abs(mag) < 0.01

    def test_gradient_nonzero_at_edge_of_deposit(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        field[24, 24] = 1.0
        # One cell away should have a gradient pointing toward the deposit
        gx, gy, mag = _gradient_at(field, 25, 24)
        assert mag > 0.0

    def test_gradient_at_boundary_does_not_crash(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        field[0, 0] = 1.0
        _gradient_at(field, 0, 0)  # Should not raise
        _gradient_at(field, FIELD_SIZE - 1, FIELD_SIZE - 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py -v`
Expected: ImportError — `resonance` module doesn't exist yet.

- [ ] **Step 3: Implement memory field functions**

Create `src/anima_mcp/display/eras/resonance.py`:

```python
"""
Resonance Era — Lumen's fifth art period.

Marks interact with accumulated emotional history through a memory field.
A 48x48 scalar grid records emotional trajectory over time, diffuses based
on stability, and drives mark selection via gradient magnitude.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from ..art_era import EraState

# --- Constants (tunable) ---
FIELD_SIZE = 48  # 48x48 grid, each cell = 5x5 pixels on 240x240 canvas
CELL_SIZE = 5  # 240 / 48
DECAY_RATE = 0.995  # Per-cycle multiplicative decay (~14 min half-life)
CLEAR_DECAY = 0.3  # Field multiplier on canvas clear
DEPOSIT_W_WARMTH = 0.5
DEPOSIT_W_PRESENCE = 0.3
DEPOSIT_W_CLARITY = 0.2
DIFFUSION_SIGMA_MIN = 0.5  # At stability=1.0 (sharp memories)
DIFFUSION_SIGMA_MAX = 2.0  # At stability=0.0 (blurred memories)
GRADIENT_LOW = 0.15  # Normalized threshold: dots below, strokes above
GRADIENT_HIGH = 0.45  # Normalized threshold: strokes below, scratches above
WARMTH_BIAS_DEGREES = 10.0  # Hue shift toward amber in high-field zones
FIELD_HIGH_THRESHOLD = 0.6  # Normalized field value considered "high" for warmth bias


# --- Memory field operations (pure functions on numpy arrays) ---

def _deposit(field: np.ndarray, pixel_x: int, pixel_y: int, value: float) -> None:
    """Add a deposit to the memory field at the cell corresponding to a canvas pixel."""
    cx = min(pixel_x // CELL_SIZE, FIELD_SIZE - 1)
    cy = min(pixel_y // CELL_SIZE, FIELD_SIZE - 1)
    field[cx, cy] += value


def _decay(field: np.ndarray) -> None:
    """Apply global exponential decay to the field."""
    field *= DECAY_RATE


def _diffuse(field: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian diffusion to the field. Returns new array."""
    if sigma < 0.1:
        return field.copy()
    return gaussian_filter(field, sigma=sigma, mode='constant', cval=0.0).astype(np.float32)


def _gradient_at(field: np.ndarray, cx: int, cy: int) -> Tuple[float, float, float]:
    """Compute gradient (gx, gy, magnitude) at a cell using finite differences.

    Returns (gx, gy, magnitude) where gx/gy are the partial derivatives.
    """
    # Clamp to valid range
    cx = max(0, min(cx, FIELD_SIZE - 1))
    cy = max(0, min(cy, FIELD_SIZE - 1))

    # Finite differences with boundary clamping
    x_prev = max(0, cx - 1)
    x_next = min(FIELD_SIZE - 1, cx + 1)
    y_prev = max(0, cy - 1)
    y_next = min(FIELD_SIZE - 1, cy + 1)

    gx = (field[x_next, cy] - field[x_prev, cy]) / max(1, x_next - x_prev)
    gy = (field[cx, y_next] - field[cx, y_prev]) / max(1, y_next - y_prev)
    mag = math.sqrt(gx * gx + gy * gy)

    return gx, gy, mag
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/resonance.py tests/test_resonance_era.py
git commit -m "resonance era: memory field core (deposit, decay, diffuse, gradient)"
```

---

### Task 2: ResonanceState and Mark Selection

Wire the memory field into an `EraState` subclass and implement gradient-driven gesture selection.

**Files:**
- Modify: `src/anima_mcp/display/eras/resonance.py`
- Modify: `tests/test_resonance_era.py`

- [ ] **Step 1: Write failing tests for ResonanceState and choose_gesture**

Add to `tests/test_resonance_era.py`:

```python
from anima_mcp.display.eras.resonance import ResonanceEra, ResonanceState


class TestResonanceState:
    def test_create_state_has_zeroed_field(self):
        era = ResonanceEra()
        state = era.create_state()
        assert state.field.shape == (FIELD_SIZE, FIELD_SIZE)
        assert state.field.sum() == 0.0

    def test_intentionality_range(self):
        state = ResonanceState()
        state.gesture_remaining = 0
        assert 0.0 <= state.intentionality() <= 1.0
        state.gesture_remaining = 20
        assert state.intentionality() > 0.1

    def test_gestures_vocabulary(self):
        state = ResonanceState()
        gestures = state.gestures()
        assert "sediment" in gestures
        assert "flow" in gestures
        assert "scratch" in gestures


class TestChooseGesture:
    def test_low_gradient_selects_sediment(self):
        """With a uniform field (zero gradient), gesture should be 'sediment'."""
        era = ResonanceEra()
        state = era.create_state()
        # Field is all zeros -> gradient is zero everywhere -> sediment
        era.choose_gesture(state, clarity=0.5, stability=0.5, presence=0.5, coherence=0.5)
        assert state.gesture == "sediment"
        assert state.gesture_remaining > 0

    def test_high_gradient_selects_scratch(self):
        """With a sharp field edge, gesture should be 'scratch'."""
        era = ResonanceEra()
        state = era.create_state()
        # Create a sharp edge: one cell at 1.0, neighbors at 0
        state.field[24, 24] = 5.0
        # Set focus to adjacent cell where gradient is high
        state._focus_cx = 25
        state._focus_cy = 24
        era.choose_gesture(state, clarity=0.5, stability=0.5, presence=0.5, coherence=0.5)
        assert state.gesture == "scratch"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py::TestResonanceState -v`
Expected: ImportError for `ResonanceEra`, `ResonanceState`.

- [ ] **Step 3: Implement ResonanceState and choose_gesture**

Add to `src/anima_mcp/display/eras/resonance.py` after the field functions:

```python
@dataclass
class ResonanceState(EraState):
    """Resonance era state — carries the memory field."""

    field: np.ndarray = None  # (48, 48) float32 — initialized in __post_init__
    cycle_count: int = 0
    # Cached gradient at current focus (updated in choose_gesture)
    _grad_gx: float = 0.0
    _grad_gy: float = 0.0
    _grad_mag: float = 0.0
    # Focus cell coordinates (updated in place_mark/drift_focus)
    _focus_cx: int = 24
    _focus_cy: int = 24

    def __post_init__(self):
        if self.field is None:
            self.field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)

    def intentionality(self) -> float:
        """Resonance intentionality: gradient magnitude + gesture commitment."""
        base = 0.1
        # Gradient contribution: high gradient = more intentional (responding to field)
        base += min(0.4, self._grad_mag * 2.0)
        # Gesture run contribution
        if self.gesture_remaining > 0:
            base += min(0.3, self.gesture_remaining / 20.0 * 0.3)
        return min(1.0, base)

    def gestures(self) -> List[str]:
        return ["sediment", "flow", "scratch"]


def _normalized_gradient(state: ResonanceState) -> float:
    """Get gradient magnitude normalized to field max. Returns 0 if field is empty."""
    field_max = state.field.max()
    if field_max < 1e-6:
        return 0.0
    gx, gy, mag = _gradient_at(state.field, state._focus_cx, state._focus_cy)
    state._grad_gx = gx
    state._grad_gy = gy
    state._grad_mag = mag
    # Normalize: raw gradient is in field-value-per-cell units
    # Divide by field_max to get [0, ~1] range
    return min(1.0, mag / field_max) if field_max > 1e-6 else 0.0


class ResonanceEra:
    """Resonance era — marks interact with accumulated emotional history."""

    name = "resonance"
    description = "Marks respond to emotional memory: sediment, flow, and scratches"

    def create_state(self) -> ResonanceState:
        return ResonanceState()

    def choose_gesture(
        self,
        state: ResonanceState,
        clarity: float,
        stability: float,
        presence: float,
        coherence: float,
    ) -> None:
        """Choose gesture based on memory field gradient at current focus."""
        norm_grad = _normalized_gradient(state)

        if norm_grad < GRADIENT_LOW:
            state.gesture = "sediment"
        elif norm_grad < GRADIENT_HIGH:
            state.gesture = "flow"
        else:
            state.gesture = "scratch"

        # Run length: coherence extends commitment
        state.gesture_remaining = random.randint(10, 25 + int(15 * coherence))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/resonance.py tests/test_resonance_era.py
git commit -m "resonance era: ResonanceState and gradient-driven gesture selection"
```

---

### Task 3: Color Generation

Implement the principled warmth→hue, clarity→opacity/saturation mapping with field-driven warmth bias.

**Files:**
- Modify: `src/anima_mcp/display/eras/resonance.py`
- Modify: `tests/test_resonance_era.py`

- [ ] **Step 1: Write failing tests for generate_color**

Add to `tests/test_resonance_era.py`:

```python
import colorsys


class TestGenerateColor:
    def test_high_warmth_produces_warm_hue(self):
        """High warmth should produce amber-ish hues (around 40 degrees)."""
        era = ResonanceEra()
        state = era.create_state()
        color, category = era.generate_color(state, warmth=0.9, clarity=0.7, stability=0.7, presence=0.7)
        h, s, v = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        hue_deg = h * 360
        # Should be in warm range: 0-80 degrees (red/orange/amber)
        assert hue_deg < 100 or hue_deg > 340, f"High warmth hue {hue_deg:.0f} should be warm"

    def test_low_warmth_produces_cool_hue(self):
        """Low warmth should produce blue-ish hues (around 220 degrees)."""
        era = ResonanceEra()
        state = era.create_state()
        color, category = era.generate_color(state, warmth=0.1, clarity=0.7, stability=0.7, presence=0.7)
        h, s, v = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        hue_deg = h * 360
        # Should be in cool range: 160-260 degrees (blue/cyan)
        assert 150 < hue_deg < 270, f"Low warmth hue {hue_deg:.0f} should be cool"

    def test_high_clarity_produces_vivid_color(self):
        """High clarity should produce high saturation."""
        era = ResonanceEra()
        state = era.create_state()
        color, _ = era.generate_color(state, warmth=0.5, clarity=0.9, stability=0.5, presence=0.5)
        _, s, _ = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        assert s > 0.6, f"High clarity should produce saturation > 0.6, got {s:.2f}"

    def test_low_clarity_produces_washed_color(self):
        """Low clarity should produce low saturation."""
        era = ResonanceEra()
        state = era.create_state()
        color, _ = era.generate_color(state, warmth=0.5, clarity=0.1, stability=0.5, presence=0.5)
        _, s, _ = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        assert s < 0.6, f"Low clarity should produce saturation < 0.6, got {s:.2f}"

    def test_field_warmth_bias(self):
        """Marks in high-field zones should shift hue toward amber."""
        era = ResonanceEra()
        state = era.create_state()

        # Baseline color with empty field
        color_cold, _ = era.generate_color(state, warmth=0.5, clarity=0.7, stability=0.7, presence=0.7)
        h_cold = colorsys.rgb_to_hsv(color_cold[0]/255, color_cold[1]/255, color_cold[2]/255)[0] * 360

        # Fill field at focus location
        state.field[state._focus_cx, state._focus_cy] = 5.0
        color_hot, _ = era.generate_color(state, warmth=0.5, clarity=0.7, stability=0.7, presence=0.7)
        h_hot = colorsys.rgb_to_hsv(color_hot[0]/255, color_hot[1]/255, color_hot[2]/255)[0] * 360

        # Hot zone hue should be shifted toward amber (lower hue degrees)
        # This is a directional test — we just check they differ
        assert h_cold != h_hot, "Field warmth bias should shift hue"

    def test_returns_valid_rgb_and_category(self):
        era = ResonanceEra()
        state = era.create_state()
        color, cat = era.generate_color(state, 0.5, 0.5, 0.5, 0.5)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
        assert cat in ("warm", "cool", "neutral")

    def test_light_regime_shifts(self):
        """Dark regime should shift color compared to dim."""
        random.seed(42)
        era = ResonanceEra()
        state = era.create_state()
        color_dim, _ = era.generate_color(state, 0.5, 0.5, 0.5, 0.5, light_regime="dim")
        random.seed(42)
        state2 = era.create_state()
        color_dark, _ = era.generate_color(state2, 0.5, 0.5, 0.5, 0.5, light_regime="dark")
        # Colors should differ due to regime shift
        assert color_dim != color_dark
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py::TestGenerateColor -v`
Expected: AttributeError — `generate_color` not defined on `ResonanceEra`.

- [ ] **Step 3: Implement generate_color**

Add to `ResonanceEra` class in `resonance.py`:

```python
    def generate_color(
        self,
        state: ResonanceState,
        warmth: float,
        clarity: float,
        stability: float,
        presence: float,
        light_regime: str = "dim",
    ) -> Tuple[Tuple[int, int, int], str]:
        """Principled color mapping: warmth->hue, clarity->saturation/opacity, stability->brightness."""
        import colorsys

        # Base hue: 220° at warmth=0 (cool blue), 40° at warmth=1 (warm amber)
        hue_deg = 220.0 - warmth * 180.0

        # Field-driven warmth bias: marks in high-field zones shift toward amber
        field_max = state.field.max()
        if field_max > 1e-6:
            field_val = state.field[state._focus_cx, state._focus_cy]
            norm_field = field_val / field_max
            if norm_field > FIELD_HIGH_THRESHOLD:
                hue_deg -= WARMTH_BIAS_DEGREES * (norm_field - FIELD_HIGH_THRESHOLD) / (1.0 - FIELD_HIGH_THRESHOLD)

        # Light regime shifts (carried over from existing eras)
        if light_regime == "dark":
            hue_deg += 30.0
            sat_mod = -0.1
            val_mod = -0.1
        elif light_regime == "bright":
            hue_deg -= 15.0
            sat_mod = 0.1
            val_mod = 0.05
        else:
            sat_mod = 0.0
            val_mod = 0.0

        hue_deg = hue_deg % 360.0
        hue = hue_deg / 360.0

        # Clarity -> saturation
        saturation = max(0.1, min(1.0, 0.3 + clarity * 0.6 + sat_mod))
        # Stability -> brightness
        brightness = max(0.2, min(1.0, 0.4 + stability * 0.5 + val_mod))

        rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
        color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

        # Hue category for mood tracker
        if hue_deg < 60 or hue_deg > 300:
            hue_category = "warm"
        elif hue_deg < 180:
            hue_category = "cool"
        else:
            hue_category = "neutral"

        return color, hue_category
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py::TestGenerateColor -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/resonance.py tests/test_resonance_era.py
git commit -m "resonance era: principled color generation with field warmth bias"
```

---

### Task 4: Mark Placement (place_mark)

Implement the three mark types: sediment (soft dots), flow (curved strokes along gradient), scratch (long thin marks crossing gradient).

**Files:**
- Modify: `src/anima_mcp/display/eras/resonance.py`
- Modify: `tests/test_resonance_era.py`

- [ ] **Step 1: Write failing tests for place_mark**

Add to `tests/test_resonance_era.py`:

```python
class FakeCanvas:
    """Minimal canvas mock for mark placement tests."""
    def __init__(self):
        self.pixels = {}

    def draw_pixel(self, x, y, color):
        if 0 <= x < 240 and 0 <= y < 240:
            self.pixels[(x, y)] = color


class TestPlaceMark:
    def test_sediment_draws_pixels(self):
        era = ResonanceEra()
        state = era.create_state()
        state.gesture = "sediment"
        canvas = FakeCanvas()
        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.5, (255, 128, 0))
        assert len(canvas.pixels) > 0

    def test_flow_draws_pixels(self):
        era = ResonanceEra()
        state = era.create_state()
        state.gesture = "flow"
        state._grad_gx = 1.0
        state._grad_gy = 0.0
        canvas = FakeCanvas()
        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.5, (255, 128, 0))
        assert len(canvas.pixels) > 0

    def test_scratch_draws_elongated_mark(self):
        """Scratches should be longer than sediment dots."""
        random.seed(42)
        era = ResonanceEra()

        # Sediment
        state_s = era.create_state()
        state_s.gesture = "sediment"
        canvas_s = FakeCanvas()
        era.place_mark(state_s, canvas_s, 120.0, 120.0, 0.0, 0.5, (255, 0, 0))

        # Scratch
        random.seed(42)
        state_x = era.create_state()
        state_x.gesture = "scratch"
        state_x._grad_gx = 0.0
        state_x._grad_gy = 1.0
        canvas_x = FakeCanvas()
        era.place_mark(state_x, canvas_x, 120.0, 120.0, 0.0, 0.5, (255, 0, 0))

        assert len(canvas_x.pixels) >= len(canvas_s.pixels), "Scratches should produce at least as many pixels as sediment"

    def test_place_mark_deposits_to_field(self):
        """place_mark should deposit to the memory field at the mark position."""
        era = ResonanceEra()
        state = era.create_state()
        state.gesture = "sediment"
        canvas = FakeCanvas()
        assert state.field.sum() == 0.0
        # Simulate anima values by setting them on state for deposit
        state._deposit_value = 0.7
        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.5, (255, 128, 0))
        assert state.field.sum() > 0.0, "place_mark should deposit to memory field"

    def test_marks_stay_within_canvas(self):
        """Marks near edges should not draw outside 0-239 range."""
        era = ResonanceEra()
        state = era.create_state()
        state.gesture = "scratch"
        state._grad_gx = 1.0
        state._grad_gy = 0.0
        canvas = FakeCanvas()
        era.place_mark(state, canvas, 5.0, 5.0, 0.0, 0.8, (255, 255, 255))
        for (x, y) in canvas.pixels:
            assert 0 <= x < 240 and 0 <= y < 240
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py::TestPlaceMark -v`
Expected: AttributeError — `place_mark` not defined.

- [ ] **Step 3: Implement place_mark**

Add a static brush helper and `place_mark` to `ResonanceEra` in `resonance.py`:

```python
    @staticmethod
    def _brush(canvas, cx: float, cy: float, radius: int, color: Tuple[int, int, int]):
        """Draw a filled circle at (cx, cy)."""
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

    def place_mark(
        self,
        state: ResonanceState,
        canvas,
        focus_x: float,
        focus_y: float,
        direction: float,
        energy: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Place a mark and deposit to the memory field."""
        x, y = int(focus_x), int(focus_y)
        gesture = state.gesture
        scale = 0.5 + energy

        if gesture == "sediment":
            # Soft circular dots — calm zones
            radius = max(1, int(1 + energy * 2))
            # Slight random offset for organic feel
            ox = random.randint(-1, 1)
            oy = random.randint(-1, 1)
            self._brush(canvas, x + ox, y + oy, radius, color)

        elif gesture == "flow":
            # Short curved strokes following gradient direction
            grad_angle = math.atan2(state._grad_gy, state._grad_gx) if (state._grad_gx != 0 or state._grad_gy != 0) else direction
            length = int(random.randint(3, 7) * scale)
            angle = grad_angle
            cx, cy = float(x), float(y)
            brush_r = max(1, int(energy * 2))
            for i in range(length):
                angle += random.gauss(0, 0.2)  # Gentle curve
                cx += math.cos(angle) * 1.2
                cy += math.sin(angle) * 1.2
                self._brush(canvas, cx, cy, brush_r, color)

        elif gesture == "scratch":
            # Long thin marks crossing the gradient (perpendicular)
            grad_angle = math.atan2(state._grad_gy, state._grad_gx)
            cross_angle = grad_angle + math.pi / 2  # Perpendicular
            length = int(random.randint(8, 16) * scale)
            cx, cy = float(x), float(y)
            for i in range(length):
                cx += math.cos(cross_angle)
                cy += math.sin(cross_angle)
                # Single pixel width — thin scratches
                ix, iy = int(cx), int(cy)
                if 0 <= ix < 240 and 0 <= iy < 240:
                    canvas.draw_pixel(ix, iy, color)

        # Deposit to memory field
        deposit_val = getattr(state, '_deposit_value', 0.5)
        _deposit(state.field, int(focus_x), int(focus_y), deposit_val)

        # Update focus cell coordinates
        state._focus_cx = min(int(focus_x) // CELL_SIZE, FIELD_SIZE - 1)
        state._focus_cy = min(int(focus_y) // CELL_SIZE, FIELD_SIZE - 1)

        # Increment cycle and apply decay + diffusion
        state.cycle_count += 1
        _decay(state.field)
        sigma = DIFFUSION_SIGMA_MIN + (1.0 - 0.5) * (DIFFUSION_SIGMA_MAX - DIFFUSION_SIGMA_MIN)  # Default stability 0.5
        state.field = _diffuse(state.field, sigma=sigma)
```

Note: The deposit value and stability for diffusion sigma will be set properly when wired into the drawing engine (Task 6). For now, `_deposit_value` is a temporary attribute and stability defaults to 0.5. This gets cleaned up in Task 6 when we add the `update_field` method that the engine calls with actual anima values.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py::TestPlaceMark -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/resonance.py tests/test_resonance_era.py
git commit -m "resonance era: place_mark with sediment, flow, scratch mark types"
```

---

### Task 5: Focus Drift

Implement gradient-influenced focus drift: random walk in low-gradient zones, along gradient in medium, perpendicular in high.

**Files:**
- Modify: `src/anima_mcp/display/eras/resonance.py`
- Modify: `tests/test_resonance_era.py`

- [ ] **Step 1: Write failing tests for drift_focus**

Add to `tests/test_resonance_era.py`:

```python
class TestDriftFocus:
    def test_drift_stays_in_bounds(self):
        """Focus should bounce off edges, never leave 20-220 range."""
        random.seed(42)
        era = ResonanceEra()
        state = era.create_state()
        fx, fy, d = 200.0, 200.0, 0.5  # Near edge, heading outward
        for _ in range(100):
            fx, fy, d = era.drift_focus(state, fx, fy, d, 0.5, 0.5, 0.5, 0.5)
            assert 0 <= fx <= 240 and 0 <= fy <= 240, f"Focus escaped bounds: ({fx}, {fy})"

    def test_drift_with_gradient_influences_direction(self):
        """In a medium-gradient zone, focus should trend along the gradient."""
        random.seed(42)
        era = ResonanceEra()
        state = era.create_state()
        # Create a field gradient pointing right (positive x)
        for i in range(FIELD_SIZE):
            state.field[i, :] = float(i) / FIELD_SIZE

        fx, fy = 120.0, 120.0
        d = math.pi / 2  # Initially pointing up (perpendicular to gradient)
        positions = []
        for _ in range(50):
            fx, fy, d = era.drift_focus(state, fx, fy, d, 0.5, 0.5, 0.5, 0.5, canvas=None)
            positions.append(fx)

        # Over many steps, focus should drift rightward (positive x) due to gradient pull
        avg_x = sum(positions) / len(positions)
        assert avg_x > 120.0, f"Expected rightward drift, avg_x={avg_x:.1f}"

    def test_drift_updates_focus_cell(self):
        """After drift, _focus_cx/_focus_cy should match the new focus position."""
        era = ResonanceEra()
        state = era.create_state()
        fx, fy, d = era.drift_focus(state, 60.0, 60.0, 0.0, 0.5, 0.5, 0.5, 0.5)
        expected_cx = min(int(fx) // CELL_SIZE, FIELD_SIZE - 1)
        expected_cy = min(int(fy) // CELL_SIZE, FIELD_SIZE - 1)
        assert state._focus_cx == expected_cx
        assert state._focus_cy == expected_cy
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py::TestDriftFocus -v`
Expected: AttributeError — `drift_focus` not defined.

- [ ] **Step 3: Implement drift_focus**

Add to `ResonanceEra` class in `resonance.py`:

```python
    def drift_focus(
        self,
        state: ResonanceState,
        focus_x: float,
        focus_y: float,
        direction: float,
        stability: float,
        presence: float,
        coherence: float,
        clarity: float = 0.5,
        canvas=None,
    ) -> Tuple[float, float, float]:
        """Gradient-influenced focus drift."""
        # Sample gradient at current focus
        cx = min(int(focus_x) // CELL_SIZE, FIELD_SIZE - 1)
        cy = min(int(focus_y) // CELL_SIZE, FIELD_SIZE - 1)
        norm_grad = _normalized_gradient(state)  # Updates state._grad_*

        # Determine drift behavior based on gradient
        step = 3 + random.random() * 5

        if norm_grad < GRADIENT_LOW:
            # Low gradient: gentle random walk
            direction += random.gauss(0, 0.15 + (1.0 - clarity) * 0.15)
        elif norm_grad < GRADIENT_HIGH:
            # Medium gradient: pull toward gradient direction
            grad_angle = math.atan2(state._grad_gy, state._grad_gx)
            # Blend current direction toward gradient
            angle_diff = grad_angle - direction
            # Normalize to [-pi, pi]
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            direction += angle_diff * 0.3 + random.gauss(0, 0.1)
        else:
            # High gradient: drift perpendicular (cross the scar)
            grad_angle = math.atan2(state._grad_gy, state._grad_gx)
            cross_angle = grad_angle + math.pi / 2
            angle_diff = cross_angle - direction
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            direction += angle_diff * 0.4 + random.gauss(0, 0.08)

        focus_x += math.cos(direction) * step
        focus_y += math.sin(direction) * step

        # Soft bounce off edges (20px margin)
        margin = 20
        if focus_x < margin:
            direction = random.uniform(-math.pi / 4, math.pi / 4)
            focus_x = float(margin)
        elif focus_x > 240 - margin:
            direction = random.uniform(math.pi * 3 / 4, math.pi * 5 / 4)
            focus_x = float(240 - margin)
        if focus_y < margin:
            direction = random.uniform(math.pi / 4, math.pi * 3 / 4)
            focus_y = float(margin)
        elif focus_y > 240 - margin:
            direction = random.uniform(-math.pi * 3 / 4, -math.pi / 4)
            focus_y = float(240 - margin)

        # Sparse jump (low coherence + low clarity = more jumps)
        jump_prob = 0.02 * (1.0 - 0.4 * coherence) * (1.0 - 0.4 * clarity)
        if random.random() < jump_prob and canvas is not None:
            gx_cell, gy_cell = canvas.sparsest_cell()
            focus_x = gx_cell * 30 + random.uniform(5, 25)
            focus_y = gy_cell * 30 + random.uniform(5, 25)
            focus_x = max(float(margin), min(float(240 - margin), focus_x))
            focus_y = max(float(margin), min(float(240 - margin), focus_y))
            direction = random.uniform(0, 2 * math.pi)

        # Update focus cell
        state._focus_cx = min(max(0, int(focus_x) // CELL_SIZE), FIELD_SIZE - 1)
        state._focus_cy = min(max(0, int(focus_y) // CELL_SIZE), FIELD_SIZE - 1)

        return focus_x, focus_y, direction
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py::TestDriftFocus -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/resonance.py tests/test_resonance_era.py
git commit -m "resonance era: gradient-influenced focus drift"
```

---

### Task 6: Anima-Driven Field Updates

Clean up the temporary `_deposit_value` hack from Task 4. The deposit value and diffusion sigma should be computed from actual anima dimensions. Add an `update_field` method on `ResonanceEra` that the `place_mark` method calls, or restructure `place_mark` to accept anima values.

The cleanest approach: `place_mark` already has `energy` which is derived from anima. But we need `warmth`, `presence`, `clarity`, and `stability` for the deposit calculation and diffusion sigma. Since the `ArtEra` protocol's `place_mark` doesn't include these, we'll store them on `ResonanceState` when `generate_color` is called (which always runs before `place_mark` in the draw cycle — see `drawing_engine.py:926-940`).

**Files:**
- Modify: `src/anima_mcp/display/eras/resonance.py`
- Modify: `tests/test_resonance_era.py`

- [ ] **Step 1: Write failing test for anima-driven deposit and diffusion**

Add to `tests/test_resonance_era.py`:

```python
class TestAnimaDrivenField:
    def test_deposit_uses_anima_blend(self):
        """Deposit value should reflect warmth/presence/clarity blend."""
        era = ResonanceEra()
        state = era.create_state()
        # generate_color caches anima values on state
        era.generate_color(state, warmth=0.8, clarity=0.6, stability=0.5, presence=0.7)
        canvas = FakeCanvas()
        state.gesture = "sediment"
        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.5, (255, 128, 0))
        # Expected deposit: 0.8*0.5 + 0.7*0.3 + 0.6*0.2 = 0.4 + 0.21 + 0.12 = 0.73
        # Field should have this value (minus one cycle of decay)
        cell_val = state.field[24, 24]
        assert cell_val > 0.5, f"Expected deposit ~0.73, got {cell_val:.3f}"

    def test_high_stability_slow_diffusion(self):
        """High stability should result in sharper field (less spread)."""
        era = ResonanceEra()
        state_stable = era.create_state()
        state_stable.field[24, 24] = 1.0
        era.generate_color(state_stable, warmth=0.5, clarity=0.5, stability=0.9, presence=0.5)
        canvas = FakeCanvas()
        state_stable.gesture = "sediment"
        era.place_mark(state_stable, canvas, 0.0, 0.0, 0.0, 0.5, (255, 0, 0))  # Mark at corner, field at center
        peak_stable = state_stable.field[24, 24]

        state_unstable = era.create_state()
        state_unstable.field[24, 24] = 1.0
        era.generate_color(state_unstable, warmth=0.5, clarity=0.5, stability=0.1, presence=0.5)
        canvas2 = FakeCanvas()
        state_unstable.gesture = "sediment"
        era.place_mark(state_unstable, canvas2, 0.0, 0.0, 0.0, 0.5, (255, 0, 0))
        peak_unstable = state_unstable.field[24, 24]

        assert peak_stable > peak_unstable, "High stability should preserve peak better (less diffusion)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py::TestAnimaDrivenField -v`
Expected: FAIL — `_deposit_value` is the old hardcoded 0.5.

- [ ] **Step 3: Refactor to use anima-driven deposit and diffusion**

Modify `generate_color` to cache anima values on state, and modify `place_mark` to use them.

In `ResonanceState`, add fields:

```python
    # Cached anima values (set by generate_color, used by place_mark)
    _cached_warmth: float = 0.5
    _cached_clarity: float = 0.5
    _cached_stability: float = 0.5
    _cached_presence: float = 0.5
```

At the start of `generate_color`, add:

```python
        # Cache anima values for place_mark's field operations
        state._cached_warmth = warmth
        state._cached_clarity = clarity
        state._cached_stability = stability
        state._cached_presence = presence
```

In `place_mark`, replace the deposit and diffusion section:

```python
        # Deposit to memory field using anima blend
        deposit_val = (state._cached_warmth * DEPOSIT_W_WARMTH +
                       state._cached_presence * DEPOSIT_W_PRESENCE +
                       state._cached_clarity * DEPOSIT_W_CLARITY)
        _deposit(state.field, int(focus_x), int(focus_y), deposit_val)

        # Update focus cell coordinates
        state._focus_cx = min(int(focus_x) // CELL_SIZE, FIELD_SIZE - 1)
        state._focus_cy = min(int(focus_y) // CELL_SIZE, FIELD_SIZE - 1)

        # Decay + diffusion (stability-driven sigma)
        state.cycle_count += 1
        _decay(state.field)
        sigma = DIFFUSION_SIGMA_MIN + (1.0 - state._cached_stability) * (DIFFUSION_SIGMA_MAX - DIFFUSION_SIGMA_MIN)
        state.field = _diffuse(state.field, sigma=sigma)
```

Remove the `_deposit_value` attribute references.

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py -v`
Expected: All tests PASS (including old ones — `_deposit_value` tests in Task 4 need updating to use `generate_color` first).

Update `TestPlaceMark::test_place_mark_deposits_to_field` to call `generate_color` before `place_mark` instead of setting `_deposit_value`:

```python
    def test_place_mark_deposits_to_field(self):
        """place_mark should deposit to the memory field at the mark position."""
        era = ResonanceEra()
        state = era.create_state()
        state.gesture = "sediment"
        canvas = FakeCanvas()
        assert state.field.sum() == 0.0
        era.generate_color(state, warmth=0.8, clarity=0.6, stability=0.5, presence=0.7)
        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.5, (255, 128, 0))
        assert state.field.sum() > 0.0, "place_mark should deposit to memory field"
```

- [ ] **Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/resonance.py tests/test_resonance_era.py
git commit -m "resonance era: anima-driven deposit and stability-driven diffusion"
```

---

### Task 7: Presence-Driven Mark Density

The spec says presence controls marks per cycle: `1 + int(presence * 3)`, range 1–4. This belongs in the drawing engine's draw frequency logic, but since the `ArtEra` protocol doesn't have a "marks per cycle" hook, we implement it inside `place_mark` — when presence is high, place multiple marks in a small cluster rather than just one.

**Files:**
- Modify: `src/anima_mcp/display/eras/resonance.py`
- Modify: `tests/test_resonance_era.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_resonance_era.py`:

```python
class TestPresenceDensity:
    def test_high_presence_draws_more_pixels(self):
        """High presence should produce more pixels per place_mark call."""
        era = ResonanceEra()

        # Low presence
        random.seed(42)
        state_low = era.create_state()
        state_low.gesture = "sediment"
        era.generate_color(state_low, warmth=0.5, clarity=0.5, stability=0.5, presence=0.1)
        canvas_low = FakeCanvas()
        era.place_mark(state_low, canvas_low, 120.0, 120.0, 0.0, 0.5, (255, 0, 0))

        # High presence
        random.seed(42)
        state_high = era.create_state()
        state_high.gesture = "sediment"
        era.generate_color(state_high, warmth=0.5, clarity=0.5, stability=0.5, presence=0.9)
        canvas_high = FakeCanvas()
        era.place_mark(state_high, canvas_high, 120.0, 120.0, 0.0, 0.5, (255, 0, 0))

        assert len(canvas_high.pixels) >= len(canvas_low.pixels), \
            f"High presence ({len(canvas_high.pixels)}px) should produce >= low presence ({len(canvas_low.pixels)}px)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py::TestPresenceDensity -v`
Expected: FAIL — currently presence doesn't affect mark count.

- [ ] **Step 3: Add presence-driven repetition to place_mark**

At the top of `place_mark`, calculate repeat count and wrap the mark-drawing logic in a loop:

```python
    def place_mark(
        self,
        state: ResonanceState,
        canvas,
        focus_x: float,
        focus_y: float,
        direction: float,
        energy: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Place mark(s) and deposit to the memory field. Presence drives density."""
        # Presence-driven mark density: 1-4 marks per call
        mark_count = 1 + int(state._cached_presence * 3)

        for m in range(mark_count):
            # Offset subsequent marks slightly from focus
            if m == 0:
                mx, my = focus_x, focus_y
            else:
                mx = focus_x + random.uniform(-4, 4)
                my = focus_y + random.uniform(-4, 4)

            x, y = int(mx), int(my)
            gesture = state.gesture
            scale = 0.5 + energy

            if gesture == "sediment":
                # ... (same as before, using x, y)
            elif gesture == "flow":
                # ... (same as before, using x, y)
            elif gesture == "scratch":
                # ... (same as before, using x, y)

        # Deposit, decay, diffuse (once per call, not per sub-mark)
        deposit_val = (state._cached_warmth * DEPOSIT_W_WARMTH +
                       state._cached_presence * DEPOSIT_W_PRESENCE +
                       state._cached_clarity * DEPOSIT_W_CLARITY)
        _deposit(state.field, int(focus_x), int(focus_y), deposit_val)
        state._focus_cx = min(int(focus_x) // CELL_SIZE, FIELD_SIZE - 1)
        state._focus_cy = min(int(focus_y) // CELL_SIZE, FIELD_SIZE - 1)
        state.cycle_count += 1
        _decay(state.field)
        sigma = DIFFUSION_SIGMA_MIN + (1.0 - state._cached_stability) * (DIFFUSION_SIGMA_MAX - DIFFUSION_SIGMA_MIN)
        state.field = _diffuse(state.field, sigma=sigma)
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/resonance.py tests/test_resonance_era.py
git commit -m "resonance era: presence-driven mark density (1-4 marks per cycle)"
```

---

### Task 8: Era Registry — Maturity Gating

Add `min_drawings` support to the era registry so Resonance (and future eras) can require a minimum gallery count before appearing in rotation.

**Files:**
- Modify: `src/anima_mcp/display/eras/__init__.py`
- Modify: `src/anima_mcp/display/eras/resonance.py` (register the era)
- Modify: `tests/test_era_registry.py`

- [ ] **Step 1: Write failing tests for maturity gating**

Add to `tests/test_era_registry.py`:

```python
class TestMaturityGating:
    def test_era_with_min_drawings_excluded_when_below(self):
        """An era with min_drawings=50 should not be chosen when drawings_saved < 50."""
        original = eras_module.auto_rotate
        saved_eras = dict(_ERAS)
        try:
            eras_module.auto_rotate = True
            # Add a gated era
            gated = type("GatedEra", (), {
                "name": "gated",
                "description": "test",
                "min_drawings": 50,
            })()
            register_era(gated)
            # With only 10 drawings, gated should never be chosen
            results = set()
            for _ in range(100):
                results.add(choose_next_era("gestural", drawings_saved=10))
            assert "gated" not in results, "Gated era should be excluded with drawings_saved=10"
        finally:
            eras_module.auto_rotate = original
            _ERAS.clear()
            _ERAS.update(saved_eras)

    def test_era_with_min_drawings_included_when_above(self):
        """An era with min_drawings=50 should be choosable when drawings_saved >= 50."""
        original = eras_module.auto_rotate
        saved_eras = dict(_ERAS)
        try:
            eras_module.auto_rotate = True
            gated = type("GatedEra", (), {
                "name": "gated",
                "description": "test",
                "min_drawings": 50,
            })()
            register_era(gated)
            results = set()
            for _ in range(100):
                results.add(choose_next_era("gestural", drawings_saved=60))
            assert "gated" in results, "Gated era should be available with drawings_saved=60"
        finally:
            eras_module.auto_rotate = original
            _ERAS.clear()
            _ERAS.update(saved_eras)

    def test_era_without_min_drawings_always_available(self):
        """Eras without min_drawings should always be available (backward compat)."""
        original = eras_module.auto_rotate
        try:
            eras_module.auto_rotate = True
            results = set()
            for _ in range(100):
                results.add(choose_next_era("gestural", drawings_saved=0))
            # All existing eras should be reachable even with 0 drawings
            assert len(results) > 1
        finally:
            eras_module.auto_rotate = original

    def test_resonance_registered(self):
        """Resonance era should be in the registry after import."""
        assert "resonance" in _ERAS
        era = _ERAS["resonance"]
        assert era.name == "resonance"
        assert getattr(era, 'min_drawings', 0) == 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_era_registry.py::TestMaturityGating -v`
Expected: FAIL — `choose_next_era` doesn't filter by `min_drawings`; `resonance` not registered.

- [ ] **Step 3: Add maturity gating to choose_next_era**

In `src/anima_mcp/display/eras/__init__.py`, modify `choose_next_era`:

```python
def choose_next_era(current: str, drawings_saved: int) -> str:
    """Choose era for next drawing.

    If auto_rotate is False, returns the current era (no change).
    If auto_rotate is True, picks a random era (lower weight for repeating).
    Eras with min_drawings > drawings_saved are excluded from rotation.
    """
    if not auto_rotate:
        return current

    candidates = [
        name for name, era in _ERAS.items()
        if getattr(era, 'min_drawings', 0) <= drawings_saved
    ]
    if len(candidates) <= 1:
        return candidates[0] if candidates else "gestural"

    weights = [0.3 if name == current else 1.0 for name in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]
```

Add resonance registration at the bottom of `__init__.py`:

```python
from .resonance import ResonanceEra  # noqa: E402
register_era(ResonanceEra())
```

Add `min_drawings = 50` as a class attribute on `ResonanceEra` in `resonance.py`:

```python
class ResonanceEra:
    name = "resonance"
    description = "Marks respond to emotional memory: sediment, flow, and scratches"
    min_drawings = 50
```

- [ ] **Step 4: Update existing registry tests for new era count**

In `tests/test_era_registry.py`, update `TestRegistryPopulation`:

```python
class TestRegistryPopulation:
    def test_five_eras_registered(self):
        assert len(_ERAS) == 5

    def test_expected_era_names(self):
        names = set(_ERAS.keys())
        assert names == {"gestural", "pointillist", "field", "geometric", "resonance"}
```

Also update `TestListEras`:

```python
class TestListEras:
    def test_returns_list(self):
        result = list_eras()
        assert isinstance(result, list)
        assert len(result) == 5
```

- [ ] **Step 5: Run all era registry tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_era_registry.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/__init__.py src/anima_mcp/display/eras/resonance.py tests/test_era_registry.py
git commit -m "era registry: maturity gating + register resonance era (unlocks at 50 drawings)"
```

---

### Task 9: Field Persistence (Canvas Save/Load)

Serialize the memory field to `canvas.json` when the active era is resonance, and restore it on load.

**Files:**
- Modify: `src/anima_mcp/display/drawing_engine.py` (save_to_disk, load_from_disk)
- Modify: `tests/test_resonance_era.py`

- [ ] **Step 1: Write failing tests for field persistence**

Add to `tests/test_resonance_era.py`:

```python
import json
from unittest.mock import patch
from pathlib import Path


class TestFieldPersistence:
    def test_field_serializes_to_list(self):
        """Memory field should be JSON-serializable as a nested list."""
        state = ResonanceState()
        state.field[10, 10] = 0.75
        field_data = state.field.tolist()
        # Should be a list of lists of floats
        assert isinstance(field_data, list)
        assert len(field_data) == FIELD_SIZE
        assert len(field_data[0]) == FIELD_SIZE
        # Should survive JSON round-trip
        restored = np.array(json.loads(json.dumps(field_data)), dtype=np.float32)
        assert abs(restored[10, 10] - 0.75) < 0.001

    def test_field_restores_from_list(self):
        """ResonanceState should be able to load field from a nested list."""
        state = ResonanceState()
        field_data = [[0.0] * FIELD_SIZE for _ in range(FIELD_SIZE)]
        field_data[5][5] = 0.42
        state.field = np.array(field_data, dtype=np.float32)
        assert abs(state.field[5, 5] - 0.42) < 0.001
```

- [ ] **Step 2: Run tests to verify they pass (these are already valid)**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py::TestFieldPersistence -v`
Expected: PASS — these test the serialization format, which is just NumPy ↔ list.

- [ ] **Step 3: Add field to canvas save_to_disk**

In `src/anima_mcp/display/drawing_engine.py`, modify `save_to_disk()`. After the existing `data = { ... }` dict construction (around line 233), add:

```python
            # Resonance memory field persistence
            if hasattr(self, '_era_state_for_save') and self._era_state_for_save is not None:
                try:
                    import numpy as np
                    if hasattr(self._era_state_for_save, 'field') and isinstance(self._era_state_for_save.field, np.ndarray):
                        data["resonance_field"] = self._era_state_for_save.field.tolist()
                except Exception:
                    pass  # NumPy not available or field not present
```

Note: `_era_state_for_save` is set by the `DrawingEngine` when persisting — it's a reference to the current `era_state`. The alternative approach (simpler): just have `DrawingEngine._persist_canvas_progress()` pass the era state to `CanvasState`.

Actually, looking at the code more carefully, the simplest approach is to store the field directly on `CanvasState` as an optional serialization field, since `CanvasState.save_to_disk()` already handles all the serialization. Add to `save_to_disk`:

After `"drawing_start_time": self.drawing_start_time,` in the data dict (line 233):

```python
                # Resonance memory field (if present)
                "resonance_field": self._resonance_field,
```

And add to `CanvasState.__init__` fields:

```python
    _resonance_field: object = None  # Optional list-of-lists for resonance era field persistence
```

In the `DrawingEngine.draw()` method, after `era_state.gesture_remaining -= 1` (line 941), add:

```python
        # Sync resonance field to canvas for persistence
        if hasattr(era_state, 'field'):
            self.canvas._resonance_field = era_state.field.tolist()
```

In `load_from_disk`, add after the attention signal restoration section:

```python
        # Restore resonance memory field
        try:
            rf = data.get("resonance_field")
            if rf is not None and isinstance(rf, list):
                self._resonance_field = rf
        except Exception:
            pass
```

In `ResonanceEra.create_state()`, check for persisted field:

```python
    def create_state(self, canvas=None) -> ResonanceState:
        state = ResonanceState()
        # Restore persisted field if available
        if canvas is not None and hasattr(canvas, '_resonance_field') and canvas._resonance_field is not None:
            try:
                state.field = np.array(canvas._resonance_field, dtype=np.float32)
                if state.field.shape != (FIELD_SIZE, FIELD_SIZE):
                    state.field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
            except Exception:
                state.field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        return state
```

Wait — `create_state()` in the `ArtEra` protocol takes no arguments. We can't change the protocol. Instead, after `create_state()` is called by the engine (line 873), we can restore the field:

In `drawing_engine.py`, after line 873-874:

```python
        if self.intent.era_state is None:
            self.intent.era_state = self.active_era.create_state()
            # Restore resonance field from canvas persistence
            if hasattr(self.intent.era_state, 'field') and self.canvas._resonance_field is not None:
                try:
                    import numpy as np
                    restored = np.array(self.canvas._resonance_field, dtype=np.float32)
                    if restored.shape == self.intent.era_state.field.shape:
                        self.intent.era_state.field = restored
                except Exception:
                    pass
```

- [ ] **Step 4: Add canvas clear field decay**

In `CanvasState.clear()` (line 113), add after `self.pixels.clear()`:

```python
        # Decay resonance field on clear (ghost of previous drawing)
        if self._resonance_field is not None:
            try:
                import numpy as np
                field = np.array(self._resonance_field, dtype=np.float32)
                field *= 0.3  # CLEAR_DECAY from resonance spec
                self._resonance_field = field.tolist()
            except Exception:
                self._resonance_field = None
```

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/ -x -q`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/drawing_engine.py src/anima_mcp/display/eras/resonance.py tests/test_resonance_era.py
git commit -m "resonance era: memory field persistence across canvas saves and clears"
```

---

### Task 10: Stability-Driven Stroke Regularity

The spec says stability modulates stroke regularity: high stability = consistent weight, smooth paths; low stability = jittery width, wobble. This affects `place_mark` for flow and scratch gestures.

**Files:**
- Modify: `src/anima_mcp/display/eras/resonance.py`
- Modify: `tests/test_resonance_era.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_resonance_era.py`:

```python
class TestStabilityRegularity:
    def test_low_stability_produces_varied_marks(self):
        """Low stability should add jitter to marks — pixel positions should vary more."""
        era = ResonanceEra()

        # High stability run
        random.seed(42)
        state_s = era.create_state()
        state_s.gesture = "flow"
        state_s._grad_gx = 1.0
        era.generate_color(state_s, warmth=0.5, clarity=0.5, stability=0.9, presence=0.5)
        canvas_s = FakeCanvas()
        era.place_mark(state_s, canvas_s, 120.0, 120.0, 0.0, 0.5, (255, 0, 0))
        pixels_stable = set(canvas_s.pixels.keys())

        # Low stability run — same seed but different stability
        random.seed(42)
        state_u = era.create_state()
        state_u.gesture = "flow"
        state_u._grad_gx = 1.0
        era.generate_color(state_u, warmth=0.5, clarity=0.5, stability=0.1, presence=0.5)
        canvas_u = FakeCanvas()
        era.place_mark(state_u, canvas_u, 120.0, 120.0, 0.0, 0.5, (255, 0, 0))
        pixels_unstable = set(canvas_u.pixels.keys())

        # They should differ due to stability-driven jitter
        assert pixels_stable != pixels_unstable, "Stability should affect mark placement"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py::TestStabilityRegularity -v`
Expected: FAIL — stability doesn't affect marks yet.

- [ ] **Step 3: Add stability-driven jitter to flow and scratch**

In the `place_mark` method, modify the flow and scratch sections to add wobble based on `1 - stability`:

For **flow** gesture:
```python
            elif gesture == "flow":
                grad_angle = math.atan2(state._grad_gy, state._grad_gx) if (state._grad_gx != 0 or state._grad_gy != 0) else direction
                length = int(random.randint(3, 7) * scale)
                angle = grad_angle
                cx, cy = float(x), float(y)
                brush_r = max(1, int(energy * 2))
                wobble = 0.1 + (1.0 - state._cached_stability) * 0.4  # Stability-driven wobble
                for i in range(length):
                    angle += random.gauss(0, wobble)
                    cx += math.cos(angle) * 1.2
                    cy += math.sin(angle) * 1.2
                    self._brush(canvas, cx, cy, brush_r, color)
```

For **scratch** gesture:
```python
            elif gesture == "scratch":
                grad_angle = math.atan2(state._grad_gy, state._grad_gx)
                cross_angle = grad_angle + math.pi / 2
                length = int(random.randint(8, 16) * scale)
                cx, cy = float(x), float(y)
                jitter = (1.0 - state._cached_stability) * 0.8  # Stability-driven jitter
                for i in range(length):
                    cx += math.cos(cross_angle) + random.gauss(0, jitter)
                    cy += math.sin(cross_angle) + random.gauss(0, jitter)
                    ix, iy = int(cx), int(cy)
                    if 0 <= ix < 240 and 0 <= iy < 240:
                        canvas.draw_pixel(ix, iy, color)
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/resonance.py tests/test_resonance_era.py
git commit -m "resonance era: stability-driven stroke regularity (jitter and wobble)"
```

---

### Task 11: scipy Dependency Check

The implementation uses `scipy.ndimage.gaussian_filter` for diffusion. Verify this is available on Pi, or provide a fallback.

**Files:**
- Modify: `src/anima_mcp/display/eras/resonance.py` (if fallback needed)

- [ ] **Step 1: Check if scipy is in the project dependencies**

Run: `cd /Users/cirwel/projects/anima-mcp && grep -i scipy pyproject.toml requirements*.txt 2>/dev/null || echo "scipy not found in deps"`

If scipy is NOT a dependency, we need a pure-NumPy fallback for `gaussian_filter`.

- [ ] **Step 2: If scipy is missing, implement NumPy-only diffusion**

Replace the `_diffuse` function with a manual 3x3 Gaussian kernel convolution:

```python
def _make_kernel(sigma: float) -> np.ndarray:
    """Create a normalized 3x3 Gaussian kernel."""
    if sigma < 0.1:
        k = np.zeros((3, 3), dtype=np.float32)
        k[1, 1] = 1.0
        return k
    ax = np.arange(-1, 2, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def _diffuse(field: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian diffusion using a 3x3 kernel (no scipy dependency)."""
    if sigma < 0.1:
        return field.copy()
    kernel = _make_kernel(sigma)
    # Pad field with zeros for boundary handling
    padded = np.pad(field, 1, mode='constant', constant_values=0)
    result = np.zeros_like(field)
    for di in range(3):
        for dj in range(3):
            result += kernel[di, dj] * padded[di:di + FIELD_SIZE, dj:dj + FIELD_SIZE]
    return result
```

Remove the `from scipy.ndimage import gaussian_filter` import.

- [ ] **Step 3: Run all tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/test_resonance_era.py -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/display/eras/resonance.py
git commit -m "resonance era: numpy-only diffusion (no scipy dependency)"
```

---

### Task 12: Full Integration Test

Run the complete test suite to verify nothing is broken.

**Files:**
- No new files

- [ ] **Step 1: Run all project tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -m pytest tests/ -x -q`
Expected: All tests PASS, including new resonance tests + existing era/registry tests.

- [ ] **Step 2: Verify resonance module imports cleanly**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -c "from anima_mcp.display.eras.resonance import ResonanceEra; e = ResonanceEra(); s = e.create_state(); print(f'Field shape: {s.field.shape}, Gestures: {s.gestures()}')"``
Expected: `Field shape: (48, 48), Gestures: ['sediment', 'flow', 'scratch']`

- [ ] **Step 3: Verify era registry includes resonance**

Run: `cd /Users/cirwel/projects/anima-mcp && python3 -c "from anima_mcp.display.eras import list_eras; print(list_eras())"`
Expected: `['gestural', 'pointillist', 'field', 'geometric', 'resonance']`

- [ ] **Step 4: Commit if any fixups were needed**

Only if previous steps required changes:
```bash
cd /Users/cirwel/projects/anima-mcp
git add -A
git commit -m "resonance era: integration fixups"
```
