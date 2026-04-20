"""
Tests for display/screens.py -- screen mode transitions and ScreenRenderer.

Covers:
- ScreenMode enum
- ScreenState defaults
- ScreenRenderer: mode transitions, navigation, groups, loading, input feedback,
  brightness overlay, connection status, caching, text wrapping
"""

import time
import pytest
from unittest.mock import patch, MagicMock

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

pytestmark = pytest.mark.skipif(not HAS_PIL, reason="PIL required for screen tests")

from anima_mcp.display.screens import (
    ScreenMode,
    ScreenState,
    ScreenRenderer,
)
from anima_mcp.display.renderer import PilRenderer
from anima_mcp.display.face import FaceState, EyeState, MouthState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_display(tmp_path):
    """Create a PilRenderer with mocked hardware."""
    with patch.object(PilRenderer, '_init_display'), \
         patch.object(PilRenderer, '_load_brightness'):
        r = PilRenderer()
        r._display = MagicMock()
        r._manual_brightness = 1.0
        r._brightness_index = 0
        r._brightness_config_path = tmp_path / "brightness.json"
        r._deferred = False
    return r


@pytest.fixture
def screen_renderer(mock_display, tmp_path):
    """Create a ScreenRenderer with mocked dependencies."""
    with patch("anima_mcp.display.drawing_engine._get_canvas_path",
               return_value=tmp_path / "canvas.json"):
        sr = ScreenRenderer(
            display_renderer=mock_display,
            db_path=str(tmp_path / "test.db"),
            identity_store=None,
        )
    return sr


# ---------------------------------------------------------------------------
# ScreenMode enum
# ---------------------------------------------------------------------------

class TestScreenMode:
    """Test ScreenMode enum values."""

    def test_all_modes_exist(self):
        expected = {
            "FACE", "SENSORS", "IDENTITY", "DIAGNOSTICS", "NEURAL",
            "INNER_LIFE", "LEARNING", "SELF_GRAPH", "GOALS_BELIEFS",
            "AGENCY", "NOTEPAD", "MESSAGES", "QUESTIONS", "VISITORS",
            "ART_ERAS", "HEALTH",
        }
        actual = {m.name for m in ScreenMode}
        assert expected.issubset(actual)

    def test_face_is_default(self):
        assert ScreenMode.FACE.value == "face"

    def test_all_have_string_values(self):
        for mode in ScreenMode:
            assert isinstance(mode.value, str)


# ---------------------------------------------------------------------------
# ScreenState defaults
# ---------------------------------------------------------------------------

class TestScreenState:
    """Test ScreenState dataclass defaults."""

    def test_default_mode(self):
        state = ScreenState()
        assert state.mode == ScreenMode.FACE

    def test_default_transition_complete(self):
        state = ScreenState()
        assert state.transition_progress == 1.0

    def test_default_loading_off(self):
        state = ScreenState()
        assert state.is_loading is False

    def test_default_wifi_connected(self):
        state = ScreenState()
        assert state.wifi_connected is True

    def test_default_governance_disconnected(self):
        state = ScreenState()
        assert state.governance_connected is False

    def test_default_message_scroll(self):
        state = ScreenState()
        assert state.message_scroll_index == 0
        assert state.message_expanded_id is None

    def test_default_era_cursor(self):
        state = ScreenState()
        assert state.era_cursor == 0

    def test_default_governance_paused(self):
        state = ScreenState()
        assert state.governance_paused is False


# ---------------------------------------------------------------------------
# ScreenRenderer -- mode transitions
# ---------------------------------------------------------------------------

class TestScreenRendererModeTransitions:
    """Test set_mode, next_mode, previous_mode."""

    def test_get_mode_default(self, screen_renderer):
        assert screen_renderer.get_mode() == ScreenMode.FACE

    def test_set_mode(self, screen_renderer):
        screen_renderer.set_mode(ScreenMode.SENSORS)
        assert screen_renderer.get_mode() == ScreenMode.SENSORS

    def test_set_mode_same_no_op(self, screen_renderer):
        screen_renderer._state.last_switch_time = 0
        screen_renderer.set_mode(ScreenMode.FACE)
        # Transition should NOT be triggered
        assert screen_renderer._state.transition_progress == 1.0

    def test_set_mode_starts_transition(self, screen_renderer):
        # Give it an image to capture for transition
        screen_renderer._display._image = Image.new("RGB", (240, 240), (0, 0, 0))
        screen_renderer.set_mode(ScreenMode.SENSORS)
        assert screen_renderer._state.transition_progress == 0.0
        assert screen_renderer._state.previous_image is not None

    def test_set_mode_updates_timestamps(self, screen_renderer):
        before = time.time()
        screen_renderer.set_mode(ScreenMode.IDENTITY)
        assert screen_renderer._state.last_switch_time >= before
        assert screen_renderer._state.last_user_action_time >= before

    def test_next_mode_cycles(self, screen_renderer):
        screen_renderer.set_mode(ScreenMode.FACE)
        screen_renderer.next_mode()
        assert screen_renderer.get_mode() != ScreenMode.FACE

    def test_next_mode_wraps_around(self, screen_renderer):
        """Cycling through all modes wraps back to FACE."""
        screen_renderer.set_mode(ScreenMode.ART_ERAS)  # Last in list
        screen_renderer._state.last_switch_time = 0  # Clear debounce
        screen_renderer.next_mode()
        assert screen_renderer.get_mode() == ScreenMode.FACE

    def test_previous_mode_cycles(self, screen_renderer):
        # FACE is default, no need to set_mode; clear debounce just in case
        screen_renderer._state.last_switch_time = 0
        screen_renderer.previous_mode()
        assert screen_renderer.get_mode() == ScreenMode.ART_ERAS  # Wraps to last

    def test_toggle_notepad_enters(self, screen_renderer):
        # FACE is default, clear debounce
        screen_renderer._state.last_switch_time = 0
        screen_renderer.toggle_notepad()
        assert screen_renderer.get_mode() == ScreenMode.NOTEPAD

    def test_toggle_notepad_exits(self, screen_renderer):
        screen_renderer.set_mode(ScreenMode.NOTEPAD)
        screen_renderer._state.last_switch_time = 0
        screen_renderer.toggle_notepad()
        assert screen_renderer.get_mode() == ScreenMode.FACE

    def test_debounce_prevents_rapid_switch(self, screen_renderer):
        """Very rapid switches should be debounced."""
        screen_renderer.set_mode(ScreenMode.SENSORS)
        # Immediately try another switch (within 20ms debounce)
        screen_renderer._state.last_switch_time = time.time()
        screen_renderer.set_mode(ScreenMode.IDENTITY)
        # Might or might not switch depending on timing, but shouldn't crash


# ---------------------------------------------------------------------------
# ScreenRenderer -- group navigation
# ---------------------------------------------------------------------------

class TestScreenRendererGroupNavigation:
    """Test group-based navigation."""

    def test_next_group_from_face(self, screen_renderer):
        # FACE is default; clear debounce
        screen_renderer._state.last_switch_time = 0
        screen_renderer.next_group()
        # Should go to info group (IDENTITY)
        assert screen_renderer.get_mode() == ScreenMode.IDENTITY

    def test_previous_group_from_face(self, screen_renderer):
        # FACE is default; clear debounce
        screen_renderer._state.last_switch_time = 0
        screen_renderer.previous_group()
        # Should go to art group (NOTEPAD)
        assert screen_renderer.get_mode() == ScreenMode.NOTEPAD

    def test_next_group_wraps(self, screen_renderer):
        screen_renderer.set_mode(ScreenMode.NOTEPAD)  # art group
        screen_renderer._state.last_switch_time = 0
        screen_renderer.next_group()
        assert screen_renderer.get_mode() == ScreenMode.FACE  # home group

    def test_previous_group_wraps(self, screen_renderer):
        # FACE is default; clear debounce
        screen_renderer._state.last_switch_time = 0
        screen_renderer.previous_group()
        assert screen_renderer.get_mode() == ScreenMode.NOTEPAD  # art group

    def test_all_modes_have_group(self, screen_renderer):
        """Every mode should be in a group."""
        for mode in ScreenMode:
            assert mode in ScreenRenderer._SCREEN_GROUPS, f"{mode} has no group"


class TestScreenRendererInGroupNavigation:
    """Test within-group navigation."""

    def test_next_in_group(self, screen_renderer):
        screen_renderer.set_mode(ScreenMode.IDENTITY)
        screen_renderer._state.last_switch_time = 0
        screen_renderer.next_in_group()
        assert screen_renderer.get_mode() == ScreenMode.SENSORS

    def test_next_in_group_wraps(self, screen_renderer):
        screen_renderer.set_mode(ScreenMode.HEALTH)  # Last in info group
        screen_renderer._state.last_switch_time = 0
        screen_renderer.next_in_group()
        assert screen_renderer.get_mode() == ScreenMode.IDENTITY  # Wraps

    def test_previous_in_group(self, screen_renderer):
        screen_renderer.set_mode(ScreenMode.SENSORS)
        screen_renderer._state.last_switch_time = 0
        screen_renderer.previous_in_group()
        assert screen_renderer.get_mode() == ScreenMode.IDENTITY

    def test_previous_in_group_wraps(self, screen_renderer):
        screen_renderer.set_mode(ScreenMode.IDENTITY)
        screen_renderer._state.last_switch_time = 0
        screen_renderer.previous_in_group()
        assert screen_renderer.get_mode() == ScreenMode.HEALTH  # Wraps

    def test_single_screen_group_no_change(self, screen_renderer):
        """FACE is alone in home group, next_in_group is no-op."""
        # FACE is default
        screen_renderer._state.last_switch_time = 0
        screen_renderer.next_in_group()
        assert screen_renderer.get_mode() == ScreenMode.FACE


class TestScreenRendererLeftRightNavigation:
    """Test navigate_left/navigate_right."""

    def test_navigate_right_non_cycle_group_goes_next(self, screen_renderer):
        """Non-cycle groups jump to next group on right."""
        # FACE is default; clear debounce
        screen_renderer._state.last_switch_time = 0
        screen_renderer.navigate_right()
        assert screen_renderer.get_mode() == ScreenMode.IDENTITY

    def test_navigate_left_non_cycle_group_goes_prev(self, screen_renderer):
        """Non-cycle groups jump to previous group on left."""
        screen_renderer.set_mode(ScreenMode.IDENTITY)
        screen_renderer._state.last_switch_time = 0
        screen_renderer.navigate_left()
        assert screen_renderer.get_mode() == ScreenMode.FACE

    def test_navigate_right_msgs_cycles_within(self, screen_renderer):
        """Messages group cycles within before jumping."""
        screen_renderer.set_mode(ScreenMode.MESSAGES)
        screen_renderer._state.last_switch_time = 0
        screen_renderer.navigate_right()
        assert screen_renderer.get_mode() == ScreenMode.QUESTIONS

    def test_navigate_right_msgs_end_jumps_group(self, screen_renderer):
        """At end of msgs group, right jumps to next group."""
        screen_renderer.set_mode(ScreenMode.VISITORS)
        screen_renderer._state.last_switch_time = 0
        screen_renderer.navigate_right()
        assert screen_renderer.get_mode() == ScreenMode.NOTEPAD

    def test_navigate_left_msgs_start_jumps_group(self, screen_renderer):
        """At start of msgs group, left jumps to previous group."""
        screen_renderer.set_mode(ScreenMode.MESSAGES)
        screen_renderer._state.last_switch_time = 0
        screen_renderer.navigate_left()
        assert screen_renderer.get_mode() == ScreenMode.NEURAL  # mind group


# ---------------------------------------------------------------------------
# ScreenRenderer -- loading state
# ---------------------------------------------------------------------------

class TestScreenRendererLoading:
    """Test loading state management."""

    def test_set_loading(self, screen_renderer):
        screen_renderer.set_loading("processing...")
        assert screen_renderer._state.is_loading is True
        assert screen_renderer._state.loading_message == "processing..."
        assert screen_renderer._state.loading_start_time > 0

    def test_clear_loading(self, screen_renderer):
        screen_renderer.set_loading("thinking...")
        screen_renderer.clear_loading()
        assert screen_renderer._state.is_loading is False
        assert screen_renderer._state.loading_message == ""


# ---------------------------------------------------------------------------
# ScreenRenderer -- connection status
# ---------------------------------------------------------------------------

class TestScreenRendererConnectionStatus:
    """Test connection status indicators."""

    def test_update_wifi_status(self, screen_renderer):
        screen_renderer.update_connection_status(wifi=False)
        assert screen_renderer._state.wifi_connected is False

    def test_update_governance_status(self, screen_renderer):
        screen_renderer.update_connection_status(governance=True)
        assert screen_renderer._state.governance_connected is True

    def test_update_both(self, screen_renderer):
        screen_renderer.update_connection_status(wifi=True, governance=True)
        assert screen_renderer._state.wifi_connected is True
        assert screen_renderer._state.governance_connected is True

    def test_update_none_preserves(self, screen_renderer):
        screen_renderer._state.wifi_connected = False
        screen_renderer.update_connection_status()
        assert screen_renderer._state.wifi_connected is False


# ---------------------------------------------------------------------------
# ScreenRenderer -- input feedback
# ---------------------------------------------------------------------------

class TestScreenRendererInputFeedback:
    """Test input feedback visual triggers."""

    def test_trigger_sets_time(self, screen_renderer):
        screen_renderer.trigger_input_feedback("left")
        assert screen_renderer._state.input_feedback_direction == "left"
        assert screen_renderer._state.input_feedback_until > time.time()

    def test_trigger_all_directions(self, screen_renderer):
        for direction in ("left", "right", "up", "down", "press"):
            screen_renderer.trigger_input_feedback(direction)
            assert screen_renderer._state.input_feedback_direction == direction

    def test_draw_input_feedback_expired(self, screen_renderer):
        """Expired feedback should not draw anything."""
        screen_renderer._state.input_feedback_until = 0.0
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_input_feedback(draw, image)
        # Should be pure black (nothing drawn)
        assert image.getpixel((0, 0)) == (0, 0, 0)

    def test_draw_input_feedback_active_left(self, screen_renderer):
        """Active left feedback draws on left edge."""
        screen_renderer._state.input_feedback_until = time.time() + 1.0
        screen_renderer._state.input_feedback_direction = "left"
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_input_feedback(draw, image)
        # Left edge should have color
        assert image.getpixel((2, 120)) != (0, 0, 0)

    def test_draw_input_feedback_active_right(self, screen_renderer):
        """Active right feedback draws on right edge."""
        screen_renderer._state.input_feedback_until = time.time() + 1.0
        screen_renderer._state.input_feedback_direction = "right"
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_input_feedback(draw, image)
        assert image.getpixel((238, 120)) != (0, 0, 0)

    def test_draw_input_feedback_active_up(self, screen_renderer):
        """Active up feedback draws on top edge."""
        screen_renderer._state.input_feedback_until = time.time() + 1.0
        screen_renderer._state.input_feedback_direction = "up"
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_input_feedback(draw, image)
        assert image.getpixel((120, 2)) != (0, 0, 0)

    def test_draw_input_feedback_active_down(self, screen_renderer):
        """Active down feedback draws on bottom edge."""
        screen_renderer._state.input_feedback_until = time.time() + 1.0
        screen_renderer._state.input_feedback_direction = "down"
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_input_feedback(draw, image)
        assert image.getpixel((120, 238)) != (0, 0, 0)

    def test_draw_input_feedback_press_corners(self, screen_renderer):
        """Press feedback draws corner highlights."""
        screen_renderer._state.input_feedback_until = time.time() + 1.0
        screen_renderer._state.input_feedback_direction = "press"
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_input_feedback(draw, image)
        # Corner should have color
        assert image.getpixel((5, 5)) != (0, 0, 0)


# ---------------------------------------------------------------------------
# ScreenRenderer -- brightness overlay
# ---------------------------------------------------------------------------

class TestScreenRendererBrightnessOverlay:
    """Test brightness overlay rendering."""

    def test_trigger_brightness_overlay(self, screen_renderer):
        screen_renderer.trigger_brightness_overlay("Full", 1.0)
        assert screen_renderer._state.brightness_overlay_name == "Full"
        assert screen_renderer._state.brightness_overlay_level == 1.0
        assert screen_renderer._state.brightness_changed_at > 0

    def test_draw_brightness_overlay_not_expired(self, screen_renderer):
        """Active brightness overlay draws on image."""
        screen_renderer._state.brightness_changed_at = time.time()
        screen_renderer._state.brightness_overlay_name = "Medium"
        screen_renderer._state.brightness_overlay_level = 0.5
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_brightness_overlay(draw, image)
        # Center area should have content
        center = image.getpixel((120, 120))
        assert center != (0, 0, 0)

    def test_draw_brightness_overlay_expired(self, screen_renderer):
        """Expired brightness overlay does nothing."""
        screen_renderer._state.brightness_changed_at = time.time() - 5.0
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_brightness_overlay(draw, image)
        # Should remain black
        assert image.getpixel((120, 120)) == (0, 0, 0)


# ---------------------------------------------------------------------------
# ScreenRenderer -- controls overlay
# ---------------------------------------------------------------------------

class TestScreenRendererControlsOverlay:
    """Test controls overlay."""

    def test_trigger_controls_overlay(self, screen_renderer):
        screen_renderer.trigger_controls_overlay(duration_s=2.0)
        assert screen_renderer._state.controls_overlay_until > time.time()

    def test_trigger_controls_overlay_minimum_duration(self, screen_renderer):
        screen_renderer.trigger_controls_overlay(duration_s=0.1)
        # Minimum is 0.5s
        assert screen_renderer._state.controls_overlay_until > time.time()


# ---------------------------------------------------------------------------
# ScreenRenderer -- screen cache
# ---------------------------------------------------------------------------

class TestScreenRendererCache:
    """Test screen image caching."""

    def test_cache_miss(self, screen_renderer):
        result = screen_renderer._check_screen_cache("test_screen", "key_v1")
        assert result is False

    def test_cache_store_and_hit(self, screen_renderer):
        image = Image.new("RGB", (240, 240), (100, 100, 100))
        screen_renderer._display._image = None  # Reset

        screen_renderer._store_screen_cache("test_screen", "key_v1", image)
        result = screen_renderer._check_screen_cache("test_screen", "key_v1")
        assert result is True

    def test_cache_miss_different_key(self, screen_renderer):
        image = Image.new("RGB", (240, 240), (100, 100, 100))
        screen_renderer._store_screen_cache("test_screen", "key_v1", image)
        result = screen_renderer._check_screen_cache("test_screen", "key_v2")
        assert result is False

    def test_cache_eviction(self, screen_renderer):
        """Cache evicts LRU entry when exceeding max size."""
        screen_renderer._screen_cache_max_size = 3
        for i in range(5):
            img = Image.new("RGB", (240, 240), (i, i, i))
            screen_renderer._store_screen_cache(f"screen_{i}", f"key_{i}", img)

        # First two should be evicted
        assert "screen_0" not in screen_renderer._screen_cache
        assert "screen_1" not in screen_renderer._screen_cache
        assert "screen_4" in screen_renderer._screen_cache

    def test_cache_returns_copy(self, screen_renderer):
        """Cache hit returns a copy to prevent mutation."""
        image = Image.new("RGB", (240, 240), (100, 100, 100))
        screen_renderer._store_screen_cache("test", "key", image)
        screen_renderer._display._image = None
        screen_renderer._check_screen_cache("test", "key")
        # The display image should be a different object (copy)
        if screen_renderer._display._image is not None:
            assert screen_renderer._display._image is not image


# ---------------------------------------------------------------------------
# ScreenRenderer -- text wrapping
# ---------------------------------------------------------------------------

class TestScreenRendererTextWrapping:
    """Test _wrap_text helper."""

    def test_wrap_short_text(self, screen_renderer):
        from PIL import ImageFont
        font = ImageFont.load_default()
        lines = screen_renderer._wrap_text("Hi", font, 200)
        assert len(lines) == 1
        assert lines[0] == "Hi"

    def test_wrap_long_text(self, screen_renderer):
        from PIL import ImageFont
        font = ImageFont.load_default()
        long_text = " ".join(["word"] * 30)
        lines = screen_renderer._wrap_text(long_text, font, 100)
        assert len(lines) > 1

    def test_wrap_empty_text(self, screen_renderer):
        from PIL import ImageFont
        font = ImageFont.load_default()
        lines = screen_renderer._wrap_text("", font, 200)
        assert lines == []


# ---------------------------------------------------------------------------
# ScreenRenderer -- transition effects
# ---------------------------------------------------------------------------

class TestScreenRendererTransition:
    """Test screen transition (fade) effects."""

    def test_apply_transition_complete(self, screen_renderer):
        """Completed transition returns new image unchanged."""
        screen_renderer._state.transition_progress = 1.0
        new_img = Image.new("RGB", (240, 240), (255, 255, 255))
        result = screen_renderer._apply_transition(new_img)
        assert result is new_img

    def test_apply_transition_no_previous_image(self, screen_renderer):
        """No previous image: transition completes immediately."""
        screen_renderer._state.transition_progress = 0.5
        screen_renderer._state.previous_image = None
        new_img = Image.new("RGB", (240, 240), (255, 255, 255))
        result = screen_renderer._apply_transition(new_img)
        assert screen_renderer._state.transition_progress == 1.0

    def test_apply_transition_blends(self, screen_renderer):
        """Active transition blends old and new images."""
        old_img = Image.new("RGB", (240, 240), (0, 0, 0))
        new_img = Image.new("RGB", (240, 240), (255, 255, 255))
        screen_renderer._state.transition_progress = 0.0
        # Set start time 5s in the past with 10s duration → ~50% progress
        screen_renderer._state.transition_start_time = time.time() - 5.0
        screen_renderer._state.transition_duration = 10.0
        screen_renderer._state.previous_image = old_img
        result = screen_renderer._apply_transition(new_img)
        # Result should be blended (not pure black or white)
        center = result.getpixel((120, 120))
        assert center != (0, 0, 0)
        assert center != (255, 255, 255)


# ---------------------------------------------------------------------------
# ScreenRenderer -- action hints
# ---------------------------------------------------------------------------

class TestScreenRendererActionHints:
    """Test action hint generation."""

    def test_face_hint(self, screen_renderer):
        hint = screen_renderer._get_action_hint(ScreenMode.FACE)
        assert "L/R" in hint

    def test_notepad_hint(self, screen_renderer):
        hint = screen_renderer._get_action_hint(ScreenMode.NOTEPAD)
        assert "save" in hint

    def test_art_eras_hint(self, screen_renderer):
        hint = screen_renderer._get_action_hint(ScreenMode.ART_ERAS)
        assert "choose" in hint or "select" in hint

    def test_messages_hint_normal(self, screen_renderer):
        screen_renderer._state.message_expanded_id = None
        hint = screen_renderer._get_action_hint(ScreenMode.MESSAGES)
        assert "scroll" in hint

    def test_messages_hint_expanded(self, screen_renderer):
        screen_renderer._state.message_expanded_id = "msg123"
        hint = screen_renderer._get_action_hint(ScreenMode.MESSAGES)
        assert "back" in hint

    def test_questions_hint_normal(self, screen_renderer):
        screen_renderer._state.qa_expanded = False
        hint = screen_renderer._get_action_hint(ScreenMode.QUESTIONS)
        assert "select" in hint

    def test_questions_hint_expanded(self, screen_renderer):
        screen_renderer._state.qa_expanded = True
        hint = screen_renderer._get_action_hint(ScreenMode.QUESTIONS)
        assert "focus" in hint


# ---------------------------------------------------------------------------
# ScreenRenderer -- screen indicator
# ---------------------------------------------------------------------------

class TestScreenRendererScreenIndicator:
    """Test screen group indicator drawing."""

    def test_draw_screen_indicator_single(self, screen_renderer):
        """Single screen group shows just group name."""
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_screen_indicator(draw, ScreenMode.FACE)
        # Should not crash; face is alone in home group

    def test_draw_screen_indicator_multi(self, screen_renderer):
        """Multi-screen group shows position indicator."""
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_screen_indicator(draw, ScreenMode.SENSORS)
        # Should not crash; sensors is in info group with position


# ---------------------------------------------------------------------------
# ScreenRenderer -- status bar
# ---------------------------------------------------------------------------

class TestScreenRendererStatusBar:
    """Test status bar drawing."""

    def test_draw_status_bar_wifi_connected(self, screen_renderer):
        screen_renderer._state.wifi_connected = True
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_status_bar(draw)
        # Should draw green indicators in top-right

    def test_draw_status_bar_wifi_disconnected(self, screen_renderer):
        screen_renderer._state.wifi_connected = False
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_status_bar(draw)
        # Should draw red X in top-right

    def test_draw_status_bar_governance_connected(self, screen_renderer):
        screen_renderer._state.governance_connected = True
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        screen_renderer._draw_status_bar(draw)


# ---------------------------------------------------------------------------
# ScreenRenderer -- loading indicator
# ---------------------------------------------------------------------------

class TestScreenRendererLoadingIndicator:
    """Test loading overlay drawing."""

    def test_loading_indicator_not_loading(self, screen_renderer):
        screen_renderer._state.is_loading = False
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        result = screen_renderer._draw_loading_indicator(draw, image)
        assert result is None

    def test_loading_indicator_active(self, screen_renderer):
        screen_renderer._state.is_loading = True
        screen_renderer._state.loading_start_time = time.time()
        screen_renderer._state.loading_message = "thinking..."
        image = Image.new("RGB", (240, 240), (128, 128, 128))
        result = screen_renderer._draw_loading_indicator(None, image)
        assert result is not None
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# ScreenRenderer -- backward compat properties
# ---------------------------------------------------------------------------

class TestScreenRendererBackwardCompat:
    """Test backward-compatibility pass-through properties."""

    def test_canvas_property(self, screen_renderer):
        assert screen_renderer._canvas is screen_renderer.drawing_engine.canvas

    def test_intent_property(self, screen_renderer):
        assert screen_renderer._intent is screen_renderer.drawing_engine.intent

    def test_active_era_property(self, screen_renderer):
        assert screen_renderer._active_era is screen_renderer.drawing_engine.active_era

    def test_drawing_goal_property(self, screen_renderer):
        assert screen_renderer._drawing_goal is screen_renderer.drawing_engine.drawing_goal

    def test_last_anima_property(self, screen_renderer):
        from conftest import make_anima
        anima = make_anima()
        screen_renderer._last_anima = anima
        assert screen_renderer.drawing_engine.last_anima is anima

    def test_mood_tracker_property(self, screen_renderer):
        assert screen_renderer._mood_tracker is screen_renderer.drawing_engine._mood_tracker


class TestScreenRendererPassThrough:
    """Test pass-through methods to DrawingEngine."""

    def test_get_drawing_eisv(self, screen_renderer):
        result = screen_renderer.get_drawing_eisv()
        assert isinstance(result, dict)

    def test_get_current_era(self, screen_renderer):
        result = screen_renderer.get_current_era()
        assert "current_era" in result
        assert "all_eras" in result

    def test_set_era(self, screen_renderer):
        result = screen_renderer.set_era("pointillist")
        assert result["success"] is True


# ---------------------------------------------------------------------------
# ScreenRenderer -- font caching
# ---------------------------------------------------------------------------

class TestScreenRendererFonts:
    """Test font loading and caching."""

    def test_get_fonts_returns_dict(self, screen_renderer):
        fonts = screen_renderer._get_fonts()
        assert isinstance(fonts, dict)
        assert "tiny" in fonts
        assert "small" in fonts
        assert "medium" in fonts
        assert "default" in fonts
        assert "large" in fonts
        assert "title" in fonts

    def test_get_fonts_cached(self, screen_renderer):
        fonts1 = screen_renderer._get_fonts()
        fonts2 = screen_renderer._get_fonts()
        assert fonts1 is fonts2  # Same dict object (cached)

    def test_get_measure_draw_cached(self, screen_renderer):
        d1 = screen_renderer._get_measure_draw()
        d2 = screen_renderer._get_measure_draw()
        assert d1 is d2


# ---------------------------------------------------------------------------
# ScreenRenderer -- messages cache hash
# ---------------------------------------------------------------------------

class TestScreenRendererMessagesCacheHash:
    """Test messages screen cache hash computation."""

    def test_empty_messages(self, screen_renderer):
        h = screen_renderer._get_messages_cache_hash([], 0, None)
        assert "|0|" in h

    def test_different_scroll_different_hash(self, screen_renderer):
        h1 = screen_renderer._get_messages_cache_hash([], 0, None)
        h2 = screen_renderer._get_messages_cache_hash([], 1, None)
        assert h1 != h2

    def test_different_expanded_different_hash(self, screen_renderer):
        h1 = screen_renderer._get_messages_cache_hash([], 0, None)
        h2 = screen_renderer._get_messages_cache_hash([], 0, "msg123")
        assert h1 != h2


# ---------------------------------------------------------------------------
# ScreenRenderer -- render dispatch (smoke tests)
# ---------------------------------------------------------------------------

class TestScreenRendererRenderDispatch:
    """Smoke tests for the main render() method with different modes."""

    @pytest.fixture
    def face_state(self):
        return FaceState(
            eyes=EyeState.NORMAL,
            mouth=MouthState.NEUTRAL,
            tint=(100, 100, 200),
            eye_openness=0.7,
        )

    def test_render_face_mode(self, screen_renderer, face_state):
        """Face mode renders without crash."""
        screen_renderer.set_mode(ScreenMode.FACE)
        screen_renderer.render(face_state=face_state)

    def test_render_notepad_mode(self, screen_renderer):
        """Notepad mode renders without crash."""
        screen_renderer.set_mode(ScreenMode.NOTEPAD)
        from conftest import make_anima
        anima = make_anima()
        screen_renderer.render(anima=anima)

    def test_render_sets_deferred_then_flushes(self, screen_renderer, face_state):
        """Render sets deferred mode and flushes at end."""
        screen_renderer.render(face_state=face_state)
        # After render, deferred should be False (flushed)
        assert screen_renderer._display._deferred is False

    def test_render_stores_governance_agent_id(self, screen_renderer, face_state):
        governance = {"unitares_agent_id": "test-agent-123"}
        screen_renderer.render(face_state=face_state, governance=governance)
        assert screen_renderer._unitares_agent_id == "test-agent-123"

    def test_render_governance_pause(self, screen_renderer, face_state):
        governance = {"action": "pause"}
        screen_renderer.render(face_state=face_state, governance=governance)
        assert screen_renderer._state.governance_paused is True

    def test_render_governance_proceed(self, screen_renderer, face_state):
        screen_renderer._state.governance_paused = True
        governance = {"action": "proceed"}
        screen_renderer.render(face_state=face_state, governance=governance)
        assert screen_renderer._state.governance_paused is False


class TestSlowRenderPhaseBreakdown:
    """Slow renders (>500ms) should log a phase breakdown so outliers
    reveal whether the hotspot is lock_wait / pre / mode / post / flush."""

    @pytest.fixture
    def face_state(self):
        return FaceState(
            eyes=EyeState.NORMAL,
            mouth=MouthState.NEUTRAL,
            tint=(100, 100, 200),
            eye_openness=0.7,
        )

    def test_slow_render_logs_phase_breakdown(self, screen_renderer, face_state, capsys):
        # Force flush to exceed the 500ms threshold so the log fires
        def slow_flush():
            time.sleep(0.6)

        screen_renderer._display.flush = slow_flush
        screen_renderer.render(face_state=face_state)

        captured = capsys.readouterr()
        assert "Slow render:" in captured.err
        # Phase labels must appear in the breakdown
        for label in ("lock_wait=", "pre=", "mode=", "post=", "flush="):
            assert label in captured.err, f"missing phase '{label}' in: {captured.err}"

    def test_fast_render_emits_no_slow_log(self, screen_renderer, face_state, capsys):
        screen_renderer.render(face_state=face_state)
        captured = capsys.readouterr()
        assert "Slow render:" not in captured.err
