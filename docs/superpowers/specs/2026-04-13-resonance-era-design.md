# Resonance Era — Design Spec

*April 13, 2026*

## Summary

A fifth drawing era where marks interact with accumulated history through a memory field. Instead of mapping anima state directly to marks, Resonance introduces an intermediate 2D scalar field that records emotional trajectory over time. New marks respond to both current anima and the gradient of this field, producing drawings with temporal depth, emergent composition, and non-repeatable path-dependent outcomes.

## Memory Field

A 48×48 `float32` NumPy array stored on `ResonanceState` (the era's `EraState` subclass). Each cell maps to a 5×5 pixel region of the 240×240 canvas.

### Per-Cycle Operations

1. **Deposit**: Add current engagement to the cell at the mark's position.
   - Deposit value: `warmth * 0.5 + presence * 0.3 + clarity * 0.2`
   - This weighted blend captures overall engagement, not just warmth.

2. **Decay**: `field *= 0.995` globally per cycle. Prevents saturation over long sessions (~14 minutes to halve).

3. **Diffuse**: 3×3 Gaussian kernel convolution.
   - Sigma: `0.5 + (1.0 - stability) * 1.5`
   - High stability → slow spread (sigma ≈ 0.5, memories stay sharp)
   - Low stability → fast spread (sigma ≈ 2.0, memories blur together)

4. **On canvas clear**: `field *= 0.3`. The ghost fades over subsequent cycles via normal decay + diffusion.

### Persistence

The field serializes alongside `CanvasState` to `canvas.json`. Survives restarts. On `create_state()`, the field initializes to zeros or loads from persisted state if resuming a canvas.

### Performance

- Deposit: O(1)
- Decay: O(2304) — single multiply over flat array
- Diffuse: O(2304) — 3×3 kernel pass
- Total: ~1-2ms per cycle on Raspberry Pi. No GPU needed.

## Mark Selection via Gradient

The gradient of the memory field at the current mark position determines mark type. Gradient is the magnitude of the finite-difference derivative across neighboring cells.

### Gradient → Mark Type

| Gradient (normalized) | Mark Type | Character |
|---|---|---|
| Low (< 0.15) | Circular dots, soft-edged | Sediment — calm zones where state was stable |
| Medium (0.15–0.45) | Short curved strokes following gradient direction | Flow — transitional zones showing direction of change |
| High (> 0.45) | Long thin marks with sharp endpoints, crossing the gradient | Scratches — scars of rapid state shifts |

Thresholds are normalized relative to `field.max()`, not absolute — they adapt as the field builds. When `field.max() < epsilon` (early canvas), fall back to random mark selection.

### Gradient → Focus Drift

- **Low gradient**: Gentle random walk. No field-driven pull.
- **Medium gradient**: Focus drifts along the gradient direction — drawn toward emotional transitions.
- **High gradient**: Focus drifts perpendicular to the gradient — crossing the scar rather than following it.

## Color Mapping

| Anima Dimension | Visual Property | Mapping |
|---|---|---|
| Warmth | Hue | `220 - warmth * 180` degrees. Low warmth = cool blue (220°), high = warm amber (40°) |
| Clarity | Opacity + saturation | Saturation: `0.3 + clarity * 0.6`. Opacity: `0.4 + clarity * 0.6` |
| Stability | Stroke regularity | High = consistent weight, smooth paths. Low = jittery width, wobble |
| Presence | Mark density per cycle | `1 + int(presence * 3)`. Range: 1–4 marks per cycle |

### Light Regime

Existing dark/dim/bright adjustments apply on top of the base mapping (carried over from other eras).

### Field-Driven Color Bias

Marks in high-field zones get +10° hue shift toward amber regardless of current warmth. Dense accumulation areas trend warm as they build — heat made visible.

## Era Integration

### Protocol Compliance

Resonance implements the standard `ArtEra` protocol:
- `create_state()` → `ResonanceState` (carries memory field, gradient cache, cycle counter)
- `choose_gesture(state, clarity, stability, presence, coherence)` → gradient-driven mark selection
- `place_mark(state, canvas, focus_x, focus_y, direction, energy, color)` → draws mark + deposits to field
- `drift_focus(state, fx, fy, direction, stability, presence, coherence, clarity, canvas)` → gradient-influenced drift
- `generate_color(state, warmth, clarity, stability, presence, light_regime)` → principled mapping above

No changes to `DrawingEngine` or `CanvasState`.

### ResonanceState

```
ResonanceState(EraState):
    field: np.ndarray          # (48, 48) float32, the memory field
    gradient_cache: tuple      # (gx, gy, magnitude) at last sample point
    cycle_count: int           # for decay timing
    gesture: str               # current mark type
    gesture_remaining: int     # marks left in current gesture run
```

### Maturity Gating

- Resonance registers in the era registry with `min_drawings: 50`.
- `choose_next_era()` excludes eras whose threshold exceeds the gallery count.
- Once unlocked, Resonance enters rotation with weight 1.0 (same as other eras).
- Gallery count is already tracked by the drawing completion system.

### File Structure

Single new file: `src/anima_mcp/display/eras/resonance.py` (~200–300 lines). Memory field logic (deposit, decay, diffuse, gradient) lives as private functions within this module. No new dependencies beyond NumPy (already required on Pi).

### Relationship to Other Eras

Resonance does not modify existing eras. They continue unchanged. When auto-rotate is on, all unlocked eras participate. Resonance subsumes their mark vocabulary (dots, strokes, curves, geometric forms) but selects marks via the memory field gradient rather than era designation.

## Open Tuning Parameters

These constants will need empirical adjustment on hardware:

| Parameter | Initial Value | What It Controls |
|---|---|---|
| `DECAY_RATE` | 0.995 | Field decay per cycle. Lower = faster fade |
| `DEPOSIT_WEIGHTS` | (0.5, 0.3, 0.2) | Warmth/presence/clarity blend for deposits |
| `DIFFUSION_SIGMA_RANGE` | (0.5, 2.0) | Min/max Gaussian sigma, mapped from stability |
| `GRADIENT_LOW` | 0.15 | Threshold for dot → stroke transition |
| `GRADIENT_HIGH` | 0.45 | Threshold for stroke → scratch transition |
| `CLEAR_DECAY` | 0.3 | Field multiplier on canvas clear |
| `WARMTH_BIAS` | 10° | Hue shift in high-field zones |
| `MIN_DRAWINGS` | 50 | Gallery count required to unlock |
