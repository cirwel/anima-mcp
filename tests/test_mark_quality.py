"""Tests for mark quality improvements across eras."""

import random
from anima_mcp.display.drawing_engine import CanvasState
from anima_mcp.display.eras.gestural import GesturalEra


class TestGesturalBrushWidth:
    """Gestural strokes, curves, drags are single-pixel for fine detail."""

    def test_stroke_is_single_pixel_wide(self):
        """Strokes should be 1px wide regardless of energy (fine-grained character)."""
        random.seed(42)
        era = GesturalEra()
        state = era.create_state()
        state.gesture = "stroke"
        state.gesture_remaining = 10
        canvas = CanvasState()

        # Horizontal stroke at y=120
        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.9, (255, 0, 0))

        ys = {y for (x, y) in canvas.pixels}
        assert len(ys) <= 2, f"Stroke should be 1px wide, got ys={ys}"

    def test_dot_high_energy_cluster(self):
        """A high-energy dot should place 2-4 pixels, not just 1."""
        random.seed(42)
        era = GesturalEra()
        state = era.create_state()
        state.gesture = "dot"
        state.gesture_remaining = 10
        canvas = CanvasState()

        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.9, (255, 0, 0))

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


from anima_mcp.display.eras.field import FieldEra  # noqa: E402
from anima_mcp.display.eras.pointillist import PointillistEra  # noqa: E402


class TestFieldSinglePixel:
    """Field era marks are single-pixel for fine detail."""

    def test_flow_dash_is_single_pixel_wide(self):
        """flow_dash should be a thin 1px line along the field."""
        random.seed(42)
        era = FieldEra()
        state = era.create_state()
        state.gesture = "flow_dash"
        state.gesture_remaining = 10
        canvas = CanvasState()

        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.9, (255, 0, 0))

        # 3-6 pixels in a line
        assert len(canvas.pixels) <= 6, f"flow_dash should be thin, got {len(canvas.pixels)}px"


class TestPointillistSinglePixel:
    """Pointillist marks are strictly single-pixel dots."""

    def test_pair_is_exactly_2px(self):
        """Pair gesture places exactly 2 pixels regardless of energy."""
        random.seed(42)
        era = PointillistEra()
        state = era.create_state()
        state.gesture = "pair"
        state.gesture_remaining = 10
        canvas = CanvasState()

        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.9, (255, 0, 0))

        assert len(canvas.pixels) == 2, f"Pair should be exactly 2px, got {len(canvas.pixels)}"

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


class TestDensityGrid:
    """CanvasState should maintain a coarse density grid."""

    def test_draw_pixel_updates_grid(self):
        canvas = CanvasState()
        canvas.draw_pixel(10, 10, (255, 0, 0))
        # Pixel (10,10) falls in cell (0,0) of 8x8 grid (30px cells)
        assert canvas.density_grid[0][0] == 1

    def test_draw_pixel_correct_cell(self):
        canvas = CanvasState()
        canvas.draw_pixel(120, 120, (255, 0, 0))
        # 120/30 = 4 -> cell (4,4)
        assert canvas.density_grid[4][4] == 1

    def test_draw_pixel_edge_cell(self):
        canvas = CanvasState()
        canvas.draw_pixel(239, 239, (255, 0, 0))
        # 239/30 = 7 -> cell (7,7)
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

    def test_sparsest_cell_returns_correct(self):
        canvas = CanvasState()
        # Fill most cells except (3,5)
        for i in range(8):
            for j in range(8):
                if (i, j) != (3, 5):
                    for k in range(10):
                        px = min(i * 30 + k, 239)
                        py = min(j * 30 + k, 239)
                        canvas.draw_pixel(px, py, (255, 255, 255))
        cell_x, cell_y = canvas.sparsest_cell()
        assert cell_x == 3 and cell_y == 5


from anima_mcp.display.eras.geometric import GeometricEra  # noqa: E402


class TestEraCompletionTuning:
    """Each era should expose fatigue_rate and min_marks_for_completion."""

    def test_geometric_fatigues_faster(self):
        era = GeometricEra()
        assert era.fatigue_rate == 2.0
        assert era.min_marks_for_completion == 3

    def test_pointillist_fatigues_slower(self):
        era = PointillistEra()
        assert era.fatigue_rate == 0.5
        assert era.min_marks_for_completion == 80

    def test_field_moderate_fatigue(self):
        era = FieldEra()
        assert era.fatigue_rate == 0.7
        assert era.min_marks_for_completion == 30

    def test_gestural_defaults(self):
        """Gestural has no explicit attributes — engine uses defaults (1.0, 5)."""
        era = GesturalEra()
        assert getattr(era, 'fatigue_rate', 1.0) == 1.0
        assert getattr(era, 'min_marks_for_completion', 5) == 5

    def test_fatigue_rate_applied_in_engine(self):
        """Engine should scale fatigue by era's fatigue_rate."""
        from anima_mcp.display.drawing_engine import DrawingEngine

        # Geometric era (fatigue_rate=2.0)
        engine_geo = DrawingEngine()
        engine_geo.set_era("geometric", force_immediate=True)
        engine_geo.intent.state.engagement = 0.5
        engine_geo.intent.state.fatigue = 0.0
        engine_geo._update_attention(0.5, 0.3, 0.5, True)  # gesture_switch=True
        geo_fatigue = engine_geo.intent.state.fatigue

        # Pointillist era (fatigue_rate=0.5)
        engine_pt = DrawingEngine()
        engine_pt.set_era("pointillist", force_immediate=True)
        engine_pt.intent.state.engagement = 0.5
        engine_pt.intent.state.fatigue = 0.0
        engine_pt._update_attention(0.5, 0.3, 0.5, True)

        pt_fatigue = engine_pt.intent.state.fatigue

        assert geo_fatigue > pt_fatigue * 2, (
            f"Geometric fatigue ({geo_fatigue:.4f}) should be > 2x pointillist ({pt_fatigue:.4f})"
        )
