"""Tests for ScreenRenderer navigation methods.

Covers navigate_left, navigate_right, next_group, previous_group,
next_in_group, previous_in_group, and verifies msgs group cycling behavior.
"""

import pytest
from unittest.mock import MagicMock, patch

from anima_mcp.display.screens import ScreenRenderer, ScreenMode


@pytest.fixture
def renderer(tmp_path):
    """Create a ScreenRenderer with mocked display and tmp db path."""
    display = MagicMock()
    display._deferred = False
    # Patch canvas path to tmp_path so no real disk I/O
    with patch("anima_mcp.display.drawing_state._get_canvas_path", return_value=tmp_path / "canvas.json"):
        r = ScreenRenderer(display, db_path=str(tmp_path / "test.db"))
    return r


class TestNavigateRight:
    """navigate_right: within msgs group cycles, other groups jump."""

    def test_face_jumps_to_identity(self, renderer):
        renderer._state.mode = ScreenMode.FACE
        renderer.navigate_right()
        assert renderer._state.mode == ScreenMode.IDENTITY

    def test_messages_to_questions(self, renderer):
        renderer._state.mode = ScreenMode.MESSAGES
        renderer.navigate_right()
        assert renderer._state.mode == ScreenMode.QUESTIONS

    def test_questions_to_visitors(self, renderer):
        renderer._state.mode = ScreenMode.QUESTIONS
        renderer.navigate_right()
        assert renderer._state.mode == ScreenMode.VISITORS

    def test_visitors_jumps_to_next_group(self, renderer):
        renderer._state.mode = ScreenMode.VISITORS
        renderer.navigate_right()
        # Next group after msgs is art, default is NOTEPAD
        assert renderer._state.mode == ScreenMode.NOTEPAD


class TestNavigateLeft:
    """navigate_left: within msgs group cycles, other groups jump."""

    def test_face_wraps_to_notepad(self, renderer):
        renderer._state.mode = ScreenMode.FACE
        renderer.navigate_left()
        # Previous group from home is art, default is NOTEPAD
        assert renderer._state.mode == ScreenMode.NOTEPAD

    def test_visitors_to_questions(self, renderer):
        renderer._state.mode = ScreenMode.VISITORS
        renderer.navigate_left()
        assert renderer._state.mode == ScreenMode.QUESTIONS

    def test_questions_to_messages(self, renderer):
        renderer._state.mode = ScreenMode.QUESTIONS
        renderer.navigate_left()
        assert renderer._state.mode == ScreenMode.MESSAGES

    def test_messages_jumps_to_previous_group(self, renderer):
        renderer._state.mode = ScreenMode.MESSAGES
        renderer.navigate_left()
        # Previous group from msgs is mind, default is NEURAL
        assert renderer._state.mode == ScreenMode.NEURAL


class TestNextGroup:
    """next_group: jumps to next group's default screen."""

    def test_home_to_info(self, renderer):
        renderer._state.mode = ScreenMode.FACE
        renderer.next_group()
        assert renderer._state.mode == ScreenMode.IDENTITY

    def test_info_to_mind(self, renderer):
        renderer._state.mode = ScreenMode.SENSORS
        renderer.next_group()
        assert renderer._state.mode == ScreenMode.NEURAL

    def test_mind_to_msgs(self, renderer):
        renderer._state.mode = ScreenMode.SELF_GRAPH
        renderer.next_group()
        assert renderer._state.mode == ScreenMode.MESSAGES

    def test_msgs_to_art(self, renderer):
        renderer._state.mode = ScreenMode.QUESTIONS
        renderer.next_group()
        assert renderer._state.mode == ScreenMode.NOTEPAD

    def test_art_to_home(self, renderer):
        renderer._state.mode = ScreenMode.ART_ERAS
        renderer.next_group()
        assert renderer._state.mode == ScreenMode.FACE

    def test_full_cycle(self, renderer):
        """home→info→mind→msgs→art→home."""
        expected = [ScreenMode.IDENTITY, ScreenMode.NEURAL, ScreenMode.MESSAGES, ScreenMode.NOTEPAD, ScreenMode.FACE]
        renderer._state.mode = ScreenMode.FACE
        for exp in expected:
            renderer._state.last_switch_time = 0  # bypass debounce
            renderer.next_group()
            assert renderer._state.mode == exp

    def test_unknown_mode_defaults_to_face(self, renderer):
        renderer._state.mode = MagicMock()  # Something not in _SCREEN_GROUPS
        renderer.next_group()
        assert renderer._state.mode == ScreenMode.FACE


class TestPreviousGroup:
    """previous_group: jumps to previous group's default screen."""

    def test_home_to_art(self, renderer):
        renderer._state.mode = ScreenMode.FACE
        renderer.previous_group()
        assert renderer._state.mode == ScreenMode.NOTEPAD

    def test_info_to_home(self, renderer):
        renderer._state.mode = ScreenMode.HEALTH
        renderer.previous_group()
        assert renderer._state.mode == ScreenMode.FACE

    def test_mind_to_info(self, renderer):
        renderer._state.mode = ScreenMode.LEARNING
        renderer.previous_group()
        assert renderer._state.mode == ScreenMode.IDENTITY

    def test_msgs_to_mind(self, renderer):
        renderer._state.mode = ScreenMode.VISITORS
        renderer.previous_group()
        assert renderer._state.mode == ScreenMode.NEURAL

    def test_art_to_msgs(self, renderer):
        renderer._state.mode = ScreenMode.NOTEPAD
        renderer.previous_group()
        assert renderer._state.mode == ScreenMode.MESSAGES


class TestNextInGroup:
    """next_in_group: cycles within group, wraps around."""

    def test_msgs_cycle(self, renderer):
        renderer._state.mode = ScreenMode.MESSAGES
        renderer.next_in_group()
        assert renderer._state.mode == ScreenMode.QUESTIONS
        renderer._state.last_switch_time = 0
        renderer.next_in_group()
        assert renderer._state.mode == ScreenMode.VISITORS
        renderer._state.last_switch_time = 0
        renderer.next_in_group()
        assert renderer._state.mode == ScreenMode.MESSAGES

    def test_info_cycle(self, renderer):
        renderer._state.mode = ScreenMode.IDENTITY
        renderer.next_in_group()
        assert renderer._state.mode == ScreenMode.SENSORS
        renderer._state.last_switch_time = 0
        renderer.next_in_group()
        assert renderer._state.mode == ScreenMode.DIAGNOSTICS
        renderer._state.last_switch_time = 0
        renderer.next_in_group()
        assert renderer._state.mode == ScreenMode.HEALTH
        renderer._state.last_switch_time = 0
        renderer.next_in_group()
        assert renderer._state.mode == ScreenMode.IDENTITY

    def test_single_screen_group_noop(self, renderer):
        renderer._state.mode = ScreenMode.FACE
        renderer.next_in_group()
        assert renderer._state.mode == ScreenMode.FACE


class TestPreviousInGroup:
    """previous_in_group: cycles backward within group."""

    def test_msgs_reverse_cycle(self, renderer):
        renderer._state.mode = ScreenMode.MESSAGES
        renderer.previous_in_group()
        assert renderer._state.mode == ScreenMode.VISITORS
        renderer._state.last_switch_time = 0
        renderer.previous_in_group()
        assert renderer._state.mode == ScreenMode.QUESTIONS
        renderer._state.last_switch_time = 0
        renderer.previous_in_group()
        assert renderer._state.mode == ScreenMode.MESSAGES

    def test_single_screen_group_noop(self, renderer):
        renderer._state.mode = ScreenMode.FACE
        renderer.previous_in_group()
        assert renderer._state.mode == ScreenMode.FACE


class TestNonCycleGroups:
    """Info, mind, and art groups jump directly on navigate_left/right (no within-group cycling)."""

    def test_info_right_jumps_group(self, renderer):
        renderer._state.mode = ScreenMode.IDENTITY
        renderer.navigate_right()
        # Info is not a cycle group, so it jumps to next group (mind)
        assert renderer._state.mode == ScreenMode.NEURAL

    def test_mind_right_jumps_group(self, renderer):
        renderer._state.mode = ScreenMode.NEURAL
        renderer.navigate_right()
        # Mind jumps to msgs
        assert renderer._state.mode == ScreenMode.MESSAGES

    def test_art_right_jumps_group(self, renderer):
        renderer._state.mode = ScreenMode.NOTEPAD
        renderer.navigate_right()
        # Art jumps to home
        assert renderer._state.mode == ScreenMode.FACE

    def test_info_left_jumps_group(self, renderer):
        renderer._state.mode = ScreenMode.SENSORS
        renderer.navigate_left()
        # Info jumps to previous group (home)
        assert renderer._state.mode == ScreenMode.FACE

    def test_mind_left_jumps_group(self, renderer):
        renderer._state.mode = ScreenMode.INNER_LIFE
        renderer.navigate_left()
        # Mind jumps to info
        assert renderer._state.mode == ScreenMode.IDENTITY

    def test_art_left_jumps_group(self, renderer):
        renderer._state.mode = ScreenMode.ART_ERAS
        renderer.navigate_left()
        # Art jumps to msgs
        assert renderer._state.mode == ScreenMode.MESSAGES
