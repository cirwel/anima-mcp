"""Tests for next_steps_advocate.py — state reporting and drive expression."""


from conftest import make_anima, make_readings

from anima_mcp.next_steps_advocate import (
    NextStep, NextStepsAdvocate, Priority, StepCategory, get_advocate,
)
from anima_mcp.eisv_mapper import EISVMetrics


# ---------------------------------------------------------------------------
# Dataclass & enum basics
# ---------------------------------------------------------------------------

class TestNextStepDataclass:
    def test_defaults_for_blockers_and_related_files(self):
        step = NextStep(
            feeling="f", desire="d", action="a",
            priority=Priority.HIGH, category=StepCategory.HARDWARE, reason="r",
        )
        assert step.blockers == []
        assert step.related_files == []

    def test_to_dict_has_all_keys(self):
        step = NextStep(
            feeling="warm", desire="explore", action="go",
            priority=Priority.LOW, category=StepCategory.TESTING, reason="curious",
        )
        d = step.to_dict()
        assert set(d.keys()) == {
            "feeling", "desire", "action", "priority", "category",
            "reason", "blockers", "estimated_time", "related_files",
        }

    def test_to_dict_priority_is_string(self):
        step = NextStep(
            feeling="f", desire="d", action="a",
            priority=Priority.CRITICAL, category=StepCategory.HARDWARE, reason="r",
        )
        assert step.to_dict()["priority"] == "critical"

    def test_to_dict_category_is_string(self):
        step = NextStep(
            feeling="f", desire="d", action="a",
            priority=Priority.LOW, category=StepCategory.OPTIMIZATION, reason="r",
        )
        assert step.to_dict()["category"] == "optimization"


class TestEnums:
    def test_priority_values(self):
        assert Priority.CRITICAL.value == "critical"
        assert Priority.HIGH.value == "high"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.LOW.value == "low"

    def test_category_values(self):
        assert StepCategory.HARDWARE.value == "hardware"
        assert StepCategory.SOFTWARE.value == "software"
        assert StepCategory.INTEGRATION.value == "integration"
        assert StepCategory.TESTING.value == "testing"
        assert StepCategory.DOCUMENTATION.value == "documentation"
        assert StepCategory.OPTIMIZATION.value == "optimization"


# ---------------------------------------------------------------------------
# analyze_current_state — display branch
# ---------------------------------------------------------------------------

class TestDisplayBranch:
    def test_no_args_returns_display_and_unitares_steps(self):
        """Defaults: display_available=False, unitares_connected=False → 2 steps."""
        adv = NextStepsAdvocate()
        steps = adv.analyze_current_state()
        assert len(steps) >= 2
        cats = {s.category for s in steps}
        assert StepCategory.HARDWARE in cats
        assert StepCategory.INTEGRATION in cats

    def test_all_good_no_issues_returns_empty(self):
        """display + unitares on, no anima → nothing to suggest."""
        adv = NextStepsAdvocate()
        steps = adv.analyze_current_state(display_available=True, unitares_connected=True)
        assert steps == []

    def test_display_unavailable_adds_high_hardware_step(self):
        adv = NextStepsAdvocate()
        steps = adv.analyze_current_state(display_available=False)
        assert len(steps) >= 1
        hw = [s for s in steps if s.category == StepCategory.HARDWARE]
        assert len(hw) >= 1
        assert hw[0].priority == Priority.HIGH

    def test_display_unavailable_feeling_is_factual(self):
        adv = NextStepsAdvocate()
        steps = adv.analyze_current_state(display_available=False)
        hw = [s for s in steps if s.category == StepCategory.HARDWARE]
        assert any("display" in s.feeling.lower() for s in hw)


# ---------------------------------------------------------------------------
# analyze_current_state — unitares branch
# ---------------------------------------------------------------------------

class TestUnitaresBranch:
    def test_unitares_disconnected_adds_integration_step(self):
        adv = NextStepsAdvocate()
        steps = adv.analyze_current_state(unitares_connected=False)
        integration = [s for s in steps if s.category == StepCategory.INTEGRATION]
        assert len(integration) >= 1
        assert integration[0].priority == Priority.MEDIUM

    def test_unitares_connected_no_integration_step(self):
        adv = NextStepsAdvocate()
        steps = adv.analyze_current_state(unitares_connected=True)
        integration = [s for s in steps if s.category == StepCategory.INTEGRATION]
        assert len(integration) == 0


# ---------------------------------------------------------------------------
# analyze_current_state — proprioception branches
# ---------------------------------------------------------------------------

class TestProprioceptionBranches:
    def test_low_clarity_adds_high_step(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.5, clarity=0.2, stability=0.5, presence=0.5)
        readings = make_readings()
        steps = adv.analyze_current_state(anima=anima, readings=readings)
        clarity_steps = [s for s in steps if "clarity" in s.feeling.lower()]
        assert len(clarity_steps) >= 1
        assert clarity_steps[0].priority == Priority.HIGH

    def test_normal_clarity_no_clarity_step(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5)
        readings = make_readings()
        steps = adv.analyze_current_state(anima=anima, readings=readings)
        clarity_steps = [s for s in steps if "clarity" in s.feeling.lower()]
        assert len(clarity_steps) == 0

    def test_high_entropy_adds_critical_step(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5)
        readings = make_readings()
        eisv = EISVMetrics(energy=0.5, integrity=0.5, entropy=0.7, valence=0.1)
        steps = adv.analyze_current_state(anima=anima, readings=readings, eisv=eisv)
        entropy_steps = [s for s in steps if "entropy" in s.feeling.lower()]
        assert len(entropy_steps) >= 1
        assert entropy_steps[0].priority == Priority.CRITICAL

    def test_low_entropy_no_chaos_step(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5)
        readings = make_readings()
        eisv = EISVMetrics(energy=0.5, integrity=0.5, entropy=0.3, valence=0.1)
        steps = adv.analyze_current_state(anima=anima, readings=readings, eisv=eisv)
        entropy_steps = [s for s in steps if "entropy" in s.feeling.lower()]
        assert len(entropy_steps) == 0

    def test_low_stability_adds_high_step(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.5, clarity=0.5, stability=0.3, presence=0.5)
        readings = make_readings()
        steps = adv.analyze_current_state(anima=anima, readings=readings)
        stability_steps = [s for s in steps if "stability" in s.feeling.lower()]
        assert len(stability_steps) >= 1
        assert stability_steps[0].priority == Priority.HIGH

    def test_normal_stability_no_stability_step(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5)
        readings = make_readings()
        steps = adv.analyze_current_state(anima=anima, readings=readings)
        stability_steps = [s for s in steps if "stability" in s.feeling.lower()]
        assert len(stability_steps) == 0

    def test_low_warmth_adds_medium_step(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.2, clarity=0.5, stability=0.5, presence=0.5)
        readings = make_readings()
        steps = adv.analyze_current_state(anima=anima, readings=readings)
        warmth_steps = [s for s in steps if "warmth" in s.feeling.lower()]
        assert len(warmth_steps) >= 1
        assert warmth_steps[0].priority == Priority.MEDIUM

    def test_normal_warmth_no_warmth_step(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5)
        readings = make_readings()
        steps = adv.analyze_current_state(anima=anima, readings=readings)
        warmth_steps = [s for s in steps if "warmth" in s.feeling.lower()]
        assert len(warmth_steps) == 0

    def test_low_presence_adds_high_step(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.5, clarity=0.5, stability=0.5, presence=0.3)
        readings = make_readings()
        steps = adv.analyze_current_state(anima=anima, readings=readings)
        presence_steps = [s for s in steps if "presence" in s.feeling.lower()]
        assert len(presence_steps) >= 1
        assert presence_steps[0].priority == Priority.HIGH

    def test_normal_presence_no_presence_step(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5)
        readings = make_readings()
        steps = adv.analyze_current_state(anima=anima, readings=readings)
        presence_steps = [s for s in steps if "presence" in s.feeling.lower()]
        assert len(presence_steps) == 0


# ---------------------------------------------------------------------------
# analyze_current_state — drive reporting
# ---------------------------------------------------------------------------

class TestDriveReporting:
    def test_active_drive_creates_step(self):
        adv = NextStepsAdvocate()
        steps = adv.analyze_current_state(
            display_available=True, unitares_connected=True,
            drives={"warmth": 0.4, "clarity": 0.0, "stability": 0.0, "presence": 0.0},
            strongest_drive="warmth",
        )
        assert len(steps) == 1
        assert steps[0].desire == "wanting warmth"
        assert steps[0].priority == Priority.LOW

    def test_multiple_active_drives_lists_others(self):
        adv = NextStepsAdvocate()
        steps = adv.analyze_current_state(
            display_available=True, unitares_connected=True,
            drives={"warmth": 0.5, "clarity": 0.3, "stability": 0.0, "presence": 0.0},
            strongest_drive="warmth",
        )
        assert len(steps) == 1
        assert "wanting warmth" in steps[0].desire
        assert "wanting to see clearly" in steps[0].desire

    def test_no_drives_no_drive_step(self):
        adv = NextStepsAdvocate()
        steps = adv.analyze_current_state(
            display_available=True, unitares_connected=True,
            drives={"warmth": 0.0, "clarity": 0.0, "stability": 0.0, "presence": 0.0},
            strongest_drive=None,
        )
        assert steps == []

    def test_low_drive_below_threshold_no_step(self):
        adv = NextStepsAdvocate()
        steps = adv.analyze_current_state(
            display_available=True, unitares_connected=True,
            drives={"warmth": 0.1, "clarity": 0.0, "stability": 0.0, "presence": 0.0},
            strongest_drive="warmth",
        )
        assert steps == []

    def test_drives_none_no_drive_step(self):
        adv = NextStepsAdvocate()
        steps = adv.analyze_current_state(
            display_available=True, unitares_connected=True,
            drives=None, strongest_drive=None,
        )
        assert steps == []

    def test_no_random_in_output(self):
        """Verify determinism — same input, same output."""
        adv = NextStepsAdvocate()
        kwargs = dict(
            display_available=True, unitares_connected=True,
            drives={"warmth": 0.4, "clarity": 0.2, "stability": 0.0, "presence": 0.0},
            strongest_drive="warmth",
        )
        s1 = adv.analyze_current_state(**kwargs)
        s2 = adv.analyze_current_state(**kwargs)
        assert s1[0].feeling == s2[0].feeling
        assert s1[0].desire == s2[0].desire


# ---------------------------------------------------------------------------
# Sort order and caching
# ---------------------------------------------------------------------------

class TestSortAndCaching:
    def test_steps_sorted_by_priority(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.5, clarity=0.2, stability=0.5, presence=0.5)
        readings = make_readings()
        eisv = EISVMetrics(energy=0.5, integrity=0.5, entropy=0.7, valence=0.1)
        steps = adv.analyze_current_state(
            anima=anima, readings=readings, eisv=eisv,
            display_available=False, unitares_connected=False,
        )
        assert len(steps) >= 3
        assert steps[0].priority == Priority.CRITICAL
        priorities = [s.priority for s in steps]
        order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}
        assert all(order[priorities[i]] <= order[priorities[i + 1]] for i in range(len(priorities) - 1))

    def test_caching_updates_after_analysis(self):
        adv = NextStepsAdvocate()
        assert adv._last_analysis is None
        assert adv._cached_steps == []
        adv.analyze_current_state(display_available=False)
        assert adv._last_analysis is not None
        assert len(adv._cached_steps) >= 1


# ---------------------------------------------------------------------------
# get_next_steps_summary
# ---------------------------------------------------------------------------

class TestGetNextStepsSummary:
    def test_returns_message_when_no_analysis(self):
        adv = NextStepsAdvocate()
        summary = adv.get_next_steps_summary()
        assert summary["message"] == "No analysis performed yet"
        assert summary["steps"] == []

    def test_returns_summary_after_analysis(self):
        adv = NextStepsAdvocate()
        adv.analyze_current_state(display_available=False, unitares_connected=False)
        summary = adv.get_next_steps_summary()
        assert summary["total_steps"] >= 2
        assert summary["last_analyzed"] is not None
        assert isinstance(summary["all_steps"], list)
        assert summary["next_action"] is not None

    def test_priority_counts_match(self):
        adv = NextStepsAdvocate()
        anima = make_anima(warmth=0.5, clarity=0.2, stability=0.5, presence=0.5)
        readings = make_readings()
        eisv = EISVMetrics(energy=0.5, integrity=0.5, entropy=0.7, valence=0.1)
        steps = adv.analyze_current_state(
            anima=anima, readings=readings, eisv=eisv,
            display_available=False, unitares_connected=False,
        )
        summary = adv.get_next_steps_summary()
        expected_critical = len([s for s in steps if s.priority == Priority.CRITICAL])
        expected_high = len([s for s in steps if s.priority == Priority.HIGH])
        expected_medium = len([s for s in steps if s.priority == Priority.MEDIUM])
        assert summary["critical"] == expected_critical
        assert summary["high"] == expected_high
        assert summary["medium"] == expected_medium


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestGetAdvocate:
    def test_returns_same_instance(self):
        import anima_mcp.next_steps_advocate as mod
        old = mod._advocate
        mod._advocate = None
        try:
            a1 = get_advocate()
            a2 = get_advocate()
            assert a1 is a2
        finally:
            mod._advocate = old
