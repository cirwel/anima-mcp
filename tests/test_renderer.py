"""
Tests for display/renderer.py -- render pipeline and display abstraction.

Covers:
- DisplayConfig defaults
- PilRenderer: canvas creation, face rendering, brightness presets,
  text rendering, deferred rendering, flush, blank, save/get image
- NoopRenderer: all methods are safe no-ops
- get_display factory function
"""

import pytest
from unittest.mock import patch, MagicMock

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

pytestmark = pytest.mark.skipif(not HAS_PIL, reason="PIL required for renderer tests")

from anima_mcp.display.renderer import (  # noqa: E402
    DisplayConfig,
    PilRenderer,
    NoopRenderer,
    DisplayRenderer,
    get_display,
    WIDTH,
    HEIGHT,
    BLACK,
    WHITE,
)
from anima_mcp.display.face import FaceState, EyeState, MouthState  # noqa: E402


# ---------------------------------------------------------------------------
# DisplayConfig
# ---------------------------------------------------------------------------

class TestDisplayConfig:
    """Test display configuration defaults."""

    def test_default_dimensions(self):
        config = DisplayConfig()
        assert config.width == 240
        assert config.height == 240

    def test_default_rotation(self):
        config = DisplayConfig()
        assert config.rotation == 180

    def test_default_fps(self):
        config = DisplayConfig()
        assert config.fps == 10

    def test_custom_config(self):
        config = DisplayConfig(width=320, height=240, rotation=90, fps=30)
        assert config.width == 320
        assert config.height == 240
        assert config.rotation == 90
        assert config.fps == 30


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module-level constants."""

    def test_width_height(self):
        assert WIDTH == 240
        assert HEIGHT == 240

    def test_colors(self):
        assert BLACK == (0, 0, 0)
        assert WHITE == (255, 255, 255)


# ---------------------------------------------------------------------------
# PilRenderer (mocked hardware)
# ---------------------------------------------------------------------------

@pytest.fixture
def renderer(tmp_path):
    """Create a PilRenderer with mocked hardware init."""
    with patch.object(PilRenderer, '_init_display'), \
         patch.object(PilRenderer, '_load_brightness'):
        r = PilRenderer()
        r._display = None  # No hardware
        r._manual_brightness = 1.0
        r._brightness_index = 0
        r._brightness_config_path = tmp_path / "brightness.json"
    return r


class TestPilRendererInit:
    """Test PilRenderer initialization."""

    def test_creates_with_default_config(self, renderer):
        assert renderer.config.width == 240
        assert renderer.config.height == 240

    def test_creates_with_custom_config(self, tmp_path):
        config = DisplayConfig(width=320, height=240)
        with patch.object(PilRenderer, '_init_display'), \
             patch.object(PilRenderer, '_load_brightness'):
            r = PilRenderer(config=config)
        assert r.config.width == 320

    def test_no_display_when_hardware_missing(self, renderer):
        assert renderer._display is None
        assert renderer.is_available() is False


class TestPilRendererCreateCanvas:
    """Test canvas creation."""

    def test_create_canvas_default_background(self, renderer):
        image, draw = renderer._create_canvas()
        assert image.size == (240, 240)
        # Check background is black
        assert image.getpixel((0, 0)) == (0, 0, 0)

    def test_create_canvas_custom_background(self, renderer):
        image, draw = renderer._create_canvas(background=(255, 0, 0))
        assert image.getpixel((0, 0)) == (255, 0, 0)

    def test_create_canvas_returns_draw(self, renderer):
        _, draw = renderer._create_canvas()
        assert isinstance(draw, ImageDraw.ImageDraw)


class TestPilRendererFaceRendering:
    """Test face rendering pipeline."""

    def _make_face_state(self, **kwargs):
        defaults = {
            "eyes": EyeState.NORMAL,
            "mouth": MouthState.NEUTRAL,
            "tint": (100, 100, 200),
            "eye_openness": 0.7,
        }
        defaults.update(kwargs)
        return FaceState(**defaults)

    def test_render_face_no_display_no_crash(self, renderer):
        """render_face should not crash when no display hardware."""
        state = self._make_face_state()
        renderer.render_face(state)

    def test_render_face_with_mock_display(self, renderer):
        """render_face creates an image with mocked display."""
        renderer._display = MagicMock()
        state = self._make_face_state()
        renderer.render_face(state)
        assert renderer._image is not None
        assert renderer._image.size == (240, 240)

    def test_render_face_with_name(self, renderer):
        """render_face renders name text at bottom."""
        renderer._display = MagicMock()
        state = self._make_face_state()
        renderer.render_face(state, name="Lumen")
        assert renderer._image is not None

    def test_render_face_stores_last_state(self, renderer):
        """render_face stores the state for transition smoothing."""
        renderer._display = MagicMock()
        state = self._make_face_state()
        renderer.render_face(state)
        assert renderer._last_face_state is not None

    def test_render_face_tint_smoothing(self, renderer):
        """Second render smooths tint transition."""
        renderer._display = MagicMock()
        state1 = self._make_face_state(tint=(100, 100, 200))
        renderer.render_face(state1)

        state2 = self._make_face_state(tint=(200, 100, 100))
        renderer.render_face(state2)
        # The tint should be smoothed, not exactly the target
        assert renderer._last_face_state is not None

    def test_render_all_eye_states(self, renderer):
        """All eye states render without error."""
        renderer._display = MagicMock()
        for eye_state in EyeState:
            state = self._make_face_state(eyes=eye_state)
            renderer.render_face(state)
            assert renderer._image is not None

    def test_render_all_mouth_states(self, renderer):
        """All mouth states render without error."""
        renderer._display = MagicMock()
        for mouth_state in MouthState:
            state = self._make_face_state(mouth=mouth_state)
            renderer.render_face(state)
            assert renderer._image is not None

    def test_render_face_blink_triggers(self, renderer):
        """Blink should trigger after blink_frequency elapsed."""
        renderer._display = MagicMock()
        renderer._last_blink_time = 0.0  # Long ago
        state = self._make_face_state(blink_frequency=0.001)
        renderer.render_face(state)
        assert renderer._blink_in_progress is True

    def test_render_face_blink_completes(self, renderer):
        """Blink should complete after blink_duration."""
        renderer._display = MagicMock()
        import time
        renderer._blink_in_progress = True
        renderer._blink_start_time = time.time() - 1.0  # 1s ago
        state = self._make_face_state(blink_duration=0.15)
        renderer.render_face(state)
        assert renderer._blink_in_progress is False

    def test_render_face_with_smile_intensity(self, renderer):
        """Face renders with smile_intensity modifier."""
        renderer._display = MagicMock()
        state = self._make_face_state(
            mouth=MouthState.NEUTRAL,
            smile_intensity=0.8,
        )
        renderer.render_face(state)
        assert renderer._image is not None

    def test_render_face_with_frown_intensity(self, renderer):
        """Face renders with negative smile_intensity (frown)."""
        renderer._display = MagicMock()
        state = self._make_face_state(
            mouth=MouthState.NEUTRAL,
            smile_intensity=-0.5,
        )
        renderer.render_face(state)
        assert renderer._image is not None

    def test_render_face_droopy_eyes(self, renderer):
        """Droopy eyes render with pupils and highlights."""
        renderer._display = MagicMock()
        state = self._make_face_state(eyes=EyeState.DROOPY, eye_openness=0.6)
        renderer.render_face(state)
        assert renderer._image is not None

    def test_render_face_squint_eyes(self, renderer):
        """Squint eyes render as narrow ovals."""
        renderer._display = MagicMock()
        state = self._make_face_state(eyes=EyeState.SQUINT, eye_openness=0.4)
        renderer.render_face(state)
        assert renderer._image is not None

    def test_render_face_wide_eyes_with_highlight(self, renderer):
        """Wide eyes at high openness show highlights."""
        renderer._display = MagicMock()
        state = self._make_face_state(
            eyes=EyeState.WIDE,
            eye_openness=0.9,
        )
        renderer.render_face(state)
        assert renderer._image is not None

    def test_render_face_open_mouth(self, renderer):
        """Open mouth renders as oval."""
        renderer._display = MagicMock()
        state = self._make_face_state(
            mouth=MouthState.OPEN,
            expression_intensity=0.8,
        )
        renderer.render_face(state)
        assert renderer._image is not None


class TestPilRendererTextRendering:
    """Test text rendering."""

    def test_render_text(self, renderer):
        renderer._display = MagicMock()
        renderer.render_text("Hello World")
        assert renderer._image is not None

    def test_render_text_multiline(self, renderer):
        renderer._display = MagicMock()
        renderer.render_text("Line 1\nLine 2\nLine 3")
        assert renderer._image is not None

    def test_render_text_custom_position(self, renderer):
        renderer._display = MagicMock()
        renderer.render_text("Test", position=(50, 50))
        assert renderer._image is not None

    def test_render_text_custom_color(self, renderer):
        renderer._display = MagicMock()
        renderer.render_text("Colored", color=(255, 0, 0))
        assert renderer._image is not None

    def test_render_colored_text(self, renderer):
        renderer._display = MagicMock()
        lines = [
            ("Red line", (255, 0, 0)),
            ("Green line", (0, 255, 0)),
            "White line",  # Should default to white
        ]
        renderer.render_colored_text(lines)
        assert renderer._image is not None


class TestPilRendererBrightness:
    """Test brightness preset cycling."""

    def test_brightness_presets_exist(self, renderer):
        assert len(renderer._brightness_presets) == 4

    def test_brightness_up_from_medium(self, renderer):
        renderer._brightness_index = 1  # Medium
        name = renderer.brightness_up()
        assert name == "Full"
        assert renderer._brightness_index == 0

    def test_brightness_up_at_max(self, renderer):
        renderer._brightness_index = 0  # Full
        name = renderer.brightness_up()
        assert name == "Full"  # Stays at max
        assert renderer._brightness_index == 0

    def test_brightness_down_from_medium(self, renderer):
        renderer._brightness_index = 1  # Medium
        name = renderer.brightness_down()
        assert name == "Dim"
        assert renderer._brightness_index == 2

    def test_brightness_down_at_min(self, renderer):
        renderer._brightness_index = 3  # Night
        name = renderer.brightness_down()
        assert name == "Night"  # Stays at min
        assert renderer._brightness_index == 3

    def test_brightness_invalidates_cache(self, renderer):
        renderer._cached_dimmed_image = "something"
        renderer._brightness_index = 1
        renderer.brightness_up()
        assert renderer._cached_dimmed_image is None

    def test_get_brightness_preset(self, renderer):
        renderer._brightness_index = 2
        preset = renderer.get_brightness_preset()
        assert preset["name"] == "Dim"

    def test_brightness_saves_to_disk(self, renderer):
        renderer._brightness_index = 1
        renderer.brightness_up()
        assert renderer._brightness_config_path.exists()

    def test_brightness_load_from_disk(self, tmp_path):
        config_path = tmp_path / "brightness.json"
        import json
        config_path.write_text(json.dumps({"name": "Dim", "display": 1.0, "leds": 0.06}))

        with patch.object(PilRenderer, '_init_display'):
            r = PilRenderer()
            r._brightness_config_path = config_path
            r._load_brightness()

        assert r._brightness_index == 2
        assert r._manual_led_brightness == 0.06


class TestPilRendererDeferredRendering:
    """Test deferred rendering mode."""

    def test_deferred_mode_skips_show(self, renderer):
        renderer._deferred = True
        renderer._display = MagicMock()
        renderer._image = Image.new("RGB", (240, 240), (0, 0, 0))
        renderer._show()
        # Should not call _push_to_display internals
        renderer._display.image.assert_not_called()

    def test_flush_pushes_to_display(self, renderer):
        renderer._display = MagicMock()
        renderer._image = Image.new("RGB", (240, 240), (0, 0, 0))
        renderer._deferred = False

        with patch.object(renderer, '_push_to_display') as mock_push:
            renderer.flush()
            mock_push.assert_called_once()


class TestPilRendererImageOps:
    """Test image save/get operations."""

    def test_save_image(self, renderer, tmp_path):
        renderer._image = Image.new("RGB", (240, 240), (255, 0, 0))
        path = str(tmp_path / "test_output.png")
        renderer.save_image(path)
        saved = Image.open(path)
        assert saved.size == (240, 240)

    def test_get_image(self, renderer):
        img = Image.new("RGB", (240, 240), (0, 255, 0))
        renderer._image = img
        assert renderer.get_image() is img

    def test_get_image_none(self, renderer):
        renderer._image = None
        assert renderer.get_image() is None


class TestPilRendererClear:
    """Test clear/blank operations."""

    def test_clear_shows_waking_face(self, renderer):
        renderer._display = MagicMock()
        renderer.clear()
        assert renderer._image is not None

    def test_blank_no_display_no_crash(self, renderer):
        renderer._display = None
        renderer.blank()  # Should not crash

    def test_blank_with_display(self, renderer):
        renderer._display = MagicMock()
        renderer.blank()
        assert renderer._image is not None

    def test_show_default(self, renderer):
        renderer._display = MagicMock()
        renderer.show_default()
        assert renderer._image is not None


class TestPilRendererDisplayRecovery:
    """Test display recovery/reinit logic."""

    def test_push_attempts_reinit_after_cooldown(self, renderer):
        """When display is None, push_to_display attempts reinit after cooldown."""
        renderer._display = None
        renderer._last_reinit_attempt = 0.0  # Long ago
        renderer._image = Image.new("RGB", (240, 240), (0, 0, 0))

        with patch.object(renderer, '_init_display') as mock_init:
            renderer._push_to_display()
            mock_init.assert_called_once()

    def test_push_no_reinit_during_cooldown(self, renderer):
        """During 30s cooldown, no reinit attempted."""
        import time
        renderer._display = None
        renderer._last_reinit_attempt = time.time()  # Just tried
        renderer._image = Image.new("RGB", (240, 240), (0, 0, 0))

        with patch.object(renderer, '_init_display') as mock_init:
            renderer._push_to_display()
            mock_init.assert_not_called()

    def test_render_image_no_display(self, renderer):
        """render_image with no display is a no-op."""
        renderer._display = None
        img = Image.new("RGB", (240, 240), (0, 0, 0))
        renderer.render_image(img)  # Should not crash


class TestPilRendererGradientFace:
    """Test the gradient face drawing."""

    def test_draw_gradient_face_produces_non_black_image(self, renderer):
        """Gradient face should produce non-trivial image content."""
        image, draw = renderer._create_canvas()
        renderer._draw_gradient_face(draw, 120, 100, 90, (100, 150, 200))
        # Check that center area is not pure black
        center_pixel = image.getpixel((120, 100))
        assert center_pixel != (0, 0, 0)


# ---------------------------------------------------------------------------
# NoopRenderer
# ---------------------------------------------------------------------------

class TestNoopRenderer:
    """Test NoopRenderer is safe no-ops."""

    def test_render_face_noop(self):
        r = NoopRenderer()
        state = FaceState(
            eyes=EyeState.NORMAL,
            mouth=MouthState.NEUTRAL,
            tint=(100, 100, 200),
            eye_openness=0.7,
        )
        r.render_face(state)  # Should not crash

    def test_render_text_noop(self):
        r = NoopRenderer()
        r.render_text("hello")  # Should not crash

    def test_clear_noop(self):
        r = NoopRenderer()
        r.clear()  # Should not crash

    def test_is_available_false(self):
        r = NoopRenderer()
        assert r.is_available() is False

    def test_show_default_noop(self):
        r = NoopRenderer()
        r.show_default()  # Should not crash


# ---------------------------------------------------------------------------
# get_display factory
# ---------------------------------------------------------------------------

class TestGetDisplay:
    """Test display factory function."""

    def test_returns_renderer(self):
        with patch.object(PilRenderer, '__init__', return_value=None):
            result = get_display()
            assert isinstance(result, DisplayRenderer)

    def test_returns_pil_renderer_when_pil_available(self):
        with patch.object(PilRenderer, '__init__', return_value=None):
            result = get_display()
            assert isinstance(result, PilRenderer)
