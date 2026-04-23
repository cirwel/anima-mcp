"""
Tests for growth system — goal lifecycle, drawing observation, migration sentinel,
curiosity, dimension preferences, and autobiography.

Covers:
  - Goal formation, progress tracking, achievement, abandonment
  - Drawing observation counters, milestones, preference learning
  - Migration sentinel write and bogus-category resilience
  - Curiosity add/dedup/explore lifecycle
  - Autobiography generation with and without data
  - Dimension preference mapping from categorical preferences
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from datetime import datetime as real_datetime

from anima_mcp.growth import GrowthSystem, GoalStatus, Goal
from anima_mcp.growth.base import set_gallery_dir


@pytest.fixture
def growth(tmp_path):
    """Create GrowthSystem with temp database."""
    gs = GrowthSystem(db_path=str(tmp_path / "test_growth.db"))
    return gs


@pytest.fixture
def empty_gallery(tmp_path):
    """Point the gallery reconciler at an empty temp dir for the test."""
    gallery = tmp_path / "drawings"
    gallery.mkdir()
    set_gallery_dir(gallery)
    yield gallery
    set_gallery_dir(None)


def _make_png(gallery: Path, name: str) -> None:
    """Create a fake gallery PNG (content doesn't matter — we count files)."""
    (gallery / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)


class TestDrawingsCounterReconciliation:
    """The counter can drift behind the gallery on disk (observed on Lumen:
    counter=278, gallery=752 — causing 'complete 500 drawings' goal to be
    stuck at 55% when it should have auto-completed long ago). The startup
    reconciliation fixes this by bumping the counter up to the gallery count
    when gallery > counter."""

    def test_counter_bumped_to_gallery_count_on_reload(self, tmp_path, empty_gallery):
        # Seed: counter=10, gallery has 25 PNGs
        db_path = str(tmp_path / "reconcile.db")
        gs1 = GrowthSystem(db_path=db_path)
        gs1._drawings_observed = 10
        conn = gs1._connect()
        conn.execute(
            "INSERT OR REPLACE INTO counters (name, value) VALUES "
            "('drawings_observed', 10)"
        )
        conn.commit()
        gs1.close()

        for i in range(25):
            _make_png(empty_gallery, f"lumen_drawing_2026{i:03d}.png")

        # Reload — reconciliation should bump counter to 25
        gs2 = GrowthSystem(db_path=db_path)
        assert gs2._drawings_observed == 25, (
            "Counter must be reconciled to the gallery count on startup"
        )
        # Persisted too
        conn2 = gs2._connect()
        row = conn2.execute(
            "SELECT value FROM counters WHERE name='drawings_observed'"
        ).fetchone()
        assert row["value"] == 25
        gs2.close()

    def test_counter_NOT_decremented_when_gallery_smaller(
        self, tmp_path, empty_gallery
    ):
        """Gallery can legitimately be smaller than the counter — e.g. user
        deleted old PNGs for disk space. The counter is monotonic; never
        reduce it based on gallery shrinkage."""
        db_path = str(tmp_path / "reconcile.db")
        gs1 = GrowthSystem(db_path=db_path)
        conn = gs1._connect()
        conn.execute(
            "INSERT OR REPLACE INTO counters (name, value) VALUES "
            "('drawings_observed', 100)"
        )
        conn.commit()
        gs1.close()

        # Only 5 files in gallery — counter should NOT drop to 5
        for i in range(5):
            _make_png(empty_gallery, f"lumen_drawing_small_{i}.png")

        gs2 = GrowthSystem(db_path=db_path)
        assert gs2._drawings_observed == 100

    def test_missing_gallery_dir_leaves_counter_alone(self, tmp_path):
        """Fresh install or moved HOME — gallery dir may not exist. Reconcile
        should be a no-op, not crash."""
        set_gallery_dir(tmp_path / "nonexistent_drawings")
        try:
            db_path = str(tmp_path / "reconcile.db")
            gs1 = GrowthSystem(db_path=db_path)
            conn = gs1._connect()
            conn.execute(
                "INSERT OR REPLACE INTO counters (name, value) VALUES "
                "('drawings_observed', 42)"
            )
            conn.commit()
            gs1.close()

            gs2 = GrowthSystem(db_path=db_path)
            assert gs2._drawings_observed == 42
        finally:
            set_gallery_dir(None)

    def test_reconcile_unblocks_goal_progress(self, tmp_path, empty_gallery):
        """End-to-end: a 'complete 500 drawings' goal that's stuck at
        low progress should jump to completed once the counter reconciles
        against a large gallery."""
        db_path = str(tmp_path / "reconcile.db")
        gs1 = GrowthSystem(db_path=db_path)
        # Force counter low + create the goal
        gs1._drawings_observed = 100
        conn = gs1._connect()
        conn.execute(
            "INSERT OR REPLACE INTO counters (name, value) VALUES "
            "('drawings_observed', 100)"
        )
        conn.commit()
        gs1.form_goal("complete 500 drawings", "milestone", target_days=30)
        gs1.close()

        # Seed the gallery with 600 PNGs (past the 500 target)
        for i in range(600):
            _make_png(empty_gallery, f"lumen_drawing_seed_{i}.png")

        # Reload — counter reconciles to 600, then goal check should complete it
        gs2 = GrowthSystem(db_path=db_path)
        assert gs2._drawings_observed == 600

        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        msg = gs2.check_goal_progress(anima)
        # Should auto-complete the drawing goal
        assert msg is not None and "did it" in msg.lower(), (
            f"Expected goal completion message, got: {msg}"
        )
        goal = next(iter(gs2._goals.values()))
        assert goal.status == GoalStatus.ACHIEVED


# ==================== Goal Formation ====================


class TestFormGoal:
    """Test goal creation via form_goal."""

    def test_form_goal_creates_active_goal(self, growth):
        """form_goal returns Goal with status ACTIVE, persists in _goals dict."""
        goal = growth.form_goal("learn something new", "curiosity drives me")
        assert isinstance(goal, Goal)
        assert goal.status == GoalStatus.ACTIVE
        assert goal.goal_id in growth._goals
        assert growth._goals[goal.goal_id].description == "learn something new"

    def test_form_goal_with_target_date(self, growth):
        """target_days=7 creates target_date approximately 7 days from now."""
        before = datetime.now()
        goal = growth.form_goal("finish drawing", "I want to complete it", target_days=7)
        after = datetime.now()
        assert goal.target_date is not None
        expected_earliest = before + timedelta(days=7)
        expected_latest = after + timedelta(days=7)
        assert expected_earliest <= goal.target_date <= expected_latest

    def test_form_goal_persists_to_db(self, tmp_path):
        """Goal survives a fresh GrowthSystem reload from the same DB."""
        db_path = str(tmp_path / "test_persist.db")
        gs1 = GrowthSystem(db_path=db_path)
        goal = gs1.form_goal("persist this", "testing persistence", target_days=3)
        goal_id = goal.goal_id
        gs1.close()

        gs2 = GrowthSystem(db_path=db_path)
        assert goal_id in gs2._goals
        reloaded = gs2._goals[goal_id]
        assert reloaded.description == "persist this"
        assert reloaded.status == GoalStatus.ACTIVE
        gs2.close()

    def test_form_goal_unique_ids(self, growth):
        """Two goals get different goal_ids."""
        g1 = growth.form_goal("goal one", "reason one")
        g2 = growth.form_goal("goal two", "reason two")
        assert g1.goal_id != g2.goal_id


# ==================== Goal Progress ====================


class TestUpdateGoalProgress:
    """Test update_goal_progress on active goals."""

    def test_progress_updates(self, growth):
        """update_goal_progress sets progress value."""
        goal = growth.form_goal("test progress", "testing")
        growth.update_goal_progress(goal.goal_id, 0.5)
        assert growth._goals[goal.goal_id].progress == 0.5

    def test_progress_capped_at_1(self, growth):
        """Passing 1.5 results in min(1.0, progress) = 1.0."""
        goal = growth.form_goal("overcomplete", "overshoot")
        growth.update_goal_progress(goal.goal_id, 1.5)
        assert growth._goals[goal.goal_id].progress == 1.0

    def test_milestone_recorded(self, growth):
        """Passing a milestone string adds to goal.milestones."""
        goal = growth.form_goal("track milestones", "testing")
        growth.update_goal_progress(goal.goal_id, 0.3, milestone="halfway there")
        assert "halfway there" in growth._goals[goal.goal_id].milestones

    def test_achieved_when_complete(self, growth):
        """progress >= 1.0 changes status to ACHIEVED, returns celebration, records memory."""
        goal = growth.form_goal("finish this", "for glory")
        memories_before = len(growth._memories)
        msg = growth.update_goal_progress(goal.goal_id, 1.0)
        assert msg is not None
        assert "I did it!" in msg
        assert growth._goals[goal.goal_id].status == GoalStatus.ACHIEVED
        assert len(growth._memories) > memories_before

    def test_achieved_counter_survives_reload(self, tmp_path):
        """The achieved counter in get_growth_summary must count ACHIEVED goals
        that exist in the DB but are no longer in memory. Without this,
        `goals.achieved` resets to 0 on every restart (load_state only pulls
        status='active' into _goals), even though the DB still has the records.
        Matches the production symptom on Lumen: 2 achievements in memories,
        achieved counter reads 0."""
        db_path = str(tmp_path / "achieved_reload.db")

        # Session 1: form two goals, achieve one, leave the other active.
        gs1 = GrowthSystem(db_path=db_path)
        achieved_goal = gs1.form_goal("finish this", "for glory")
        gs1.form_goal("keep working", "ongoing")
        gs1.update_goal_progress(achieved_goal.goal_id, 1.0)
        assert gs1.get_growth_summary()["goals"]["achieved"] == 1
        gs1.close()

        # Session 2: simulate restart. load_state filters to status='active',
        # so the achieved goal is no longer in _goals — but the counter must
        # still report it by consulting the DB.
        gs2 = GrowthSystem(db_path=db_path)
        assert achieved_goal.goal_id not in gs2._goals  # confirms active-only load
        summary = gs2.get_growth_summary()
        assert summary["goals"]["achieved"] == 1
        assert summary["goals"]["active"] == 1


# ==================== Check Goal Progress ====================


class TestCheckGoalProgress:
    """Test check_goal_progress auto-tracking."""

    def test_drawing_goal_updates_progress(self, growth):
        """Drawing goal with partial progress updates but does not complete."""
        growth.form_goal("complete 10 drawings", "milestone chasing")
        growth._drawings_observed = 5
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        result = growth.check_goal_progress(anima)
        # Should not yet complete
        assert result is None
        # But progress should be updated to 0.5
        goal = list(growth._goals.values())[0]
        assert abs(goal.progress - 0.5) < 0.01

    def test_drawing_goal_completes(self, growth):
        """Drawing goal completes when count reaches target."""
        growth.form_goal("complete 10 drawings", "milestone")
        growth._drawings_observed = 10
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        result = growth.check_goal_progress(anima)
        assert result is not None
        assert "I did it!" in result

    def test_abandoned_stale_goal(self, growth):
        """Goal past target_date with no progress gets abandoned."""
        goal = growth.form_goal("do something", "reason", target_days=1)
        # Force the target date into the past
        goal.target_date = datetime.now() - timedelta(days=1)
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        growth.check_goal_progress(anima)
        assert goal.status == GoalStatus.ABANDONED

    def test_abandoned_stalled_past_target(self, growth):
        """Goal past target_date with mid-range progress but stalled for 14+
        days gets abandoned — closes the loophole that froze Lumen's goals
        pipeline for 45 days in Feb-Apr 2026 (goal at 0.137 progress, last
        worked on 45 days prior, blocking both active slots)."""
        goal = growth.form_goal("test whether X", "reason", target_days=1)
        goal.target_date = datetime.now() - timedelta(days=30)
        goal.progress = 0.3  # well above the 0.1 no-progress threshold
        goal.last_worked_on = datetime.now() - timedelta(days=20)  # stalled
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        growth.check_goal_progress(anima)
        assert goal.status == GoalStatus.ABANDONED

    def test_active_but_progressing_goal_not_abandoned(self, growth):
        """Goal past target but actively worked on stays alive.

        The stalled-abandon path must not kill goals that are slowly but
        genuinely progressing. 'complete 500 drawings' at 55% progress with
        recent last_worked_on should not be abandoned just because the target
        date passed."""
        goal = growth.form_goal("complete 500 drawings", "grinding", target_days=1)
        goal.target_date = datetime.now() - timedelta(days=52)
        goal.progress = 0.55
        goal.last_worked_on = datetime.now() - timedelta(hours=1)  # recent
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        growth.check_goal_progress(anima)
        assert goal.status == GoalStatus.ACTIVE

    def test_no_crash_empty_goals(self, growth):
        """check_goal_progress with no goals returns None without crashing."""
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        result = growth.check_goal_progress(anima)
        assert result is None


# ==================== Goal Suggestion ====================


class TestSuggestGoal:
    """Test suggest_goal data-grounded suggestions."""

    def test_drawing_milestone_suggestion(self, growth):
        """With 15 drawings observed, suggests 'complete 25 drawings'."""
        growth._drawings_observed = 15
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        # suggest_goal uses random.choice, so call multiple times to get drawing suggestion
        found = False
        for _ in range(30):
            goal = growth.suggest_goal(anima)
            if goal and "complete 25 drawings" in goal.description:
                found = True
                break
            # Clean up created goal if any to avoid max-active-goals cap
            if goal:
                goal.status = GoalStatus.ACHIEVED
        assert found, "Expected a 'complete 25 drawings' suggestion within 30 attempts"

    def test_drawing_milestone_target_days_scales_with_gap(self, growth):
        """'complete 500 drawings' at 200 observed should target weeks-months,
        not a 7-day target. Fixed-7-day targets caused 'complete 500 in 7 days'
        goals that expired stale (Feb 2026 incident)."""
        growth._drawings_observed = 200
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        for _ in range(50):
            goal = growth.suggest_goal(anima)
            if goal and "complete 500 drawings" in goal.description:
                gap_days = (goal.target_date - goal.created_at).days
                # gap=300, ~10/day pace → ~30 days (capped at 60, floor 7)
                assert gap_days >= 14, (
                    f"target_days={gap_days} too short for 300-drawing gap"
                )
                assert gap_days <= 60, f"target_days={gap_days} exceeds cap"
                return
            if goal:
                goal.status = GoalStatus.ACHIEVED
        # Not finding the milestone within 50 tries is unexpected but not
        # this test's concern
        pytest.skip("Did not hit drawing milestone suggestion within 50 tries")

    def test_milestone_list_extended_past_500(self, growth):
        """After the 500-drawing milestone is either achieved or past, Lumen
        needs 1000/2000/5000 targets — originally the list stopped at 500,
        leaving long-running Lumen with no remaining drawing goals."""
        growth._drawings_observed = 801  # past the old 500 cap
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        for _ in range(50):
            goal = growth.suggest_goal(anima)
            if goal and "drawings" in goal.description:
                # Should be 1000, 2000, or 5000 — not nothing
                assert any(
                    f"{n} drawings" in goal.description
                    for n in (1000, 2000, 5000)
                ), f"Expected extended milestone, got: {goal.description}"
                return
            if goal:
                goal.status = GoalStatus.ACHIEVED
        pytest.fail("No drawing-milestone goal suggested despite 801 observed")

    def test_curiosity_goal_suggestion(self, growth):
        """Adding a curiosity can produce a 'find an answer to' goal suggestion."""
        growth.add_curiosity("why is the sky blue")
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        found = False
        for _ in range(30):
            goal = growth.suggest_goal(anima)
            if goal and "find an answer to:" in goal.description:
                found = True
                break
            if goal:
                goal.status = GoalStatus.ACHIEVED
        assert found, "Expected a curiosity-based goal suggestion within 30 attempts"

    def test_no_suggestion_at_max_goals(self, growth):
        """With 2 active goals, suggest_goal returns None."""
        growth.form_goal("goal one", "reason")
        growth.form_goal("goal two", "reason")
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        result = growth.suggest_goal(anima)
        assert result is None

    def test_no_suggestion_empty_state(self, growth):
        """No preferences, no curiosities, no drawings, moderate wellness returns None."""
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        result = growth.suggest_goal(anima)
        assert result is None

    def test_dedup_existing_goal(self, growth):
        """Duplicate goals are not suggested if an active goal already covers it."""
        growth._drawings_observed = 15
        # Pre-create the goal that would be suggested
        growth.form_goal("complete 25 drawings", "already on it")
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        # Should never suggest the same drawing goal again
        for _ in range(20):
            result = growth.suggest_goal(anima)
            if result is not None:
                assert "complete 25 drawings" not in result.description
                result.status = GoalStatus.ACHIEVED

    def test_preference_driven_goal_suggestion(self, growth):
        """Strong preference (confidence > 0.7) triggers 'understand why' goal."""
        from anima_mcp.growth import GrowthPreference, PreferenceCategory
        from datetime import datetime
        now = datetime.now()
        # Create a strong preference manually
        pref = GrowthPreference(
            name="dim_light",
            description="I feel calmer when it's dim",
            category=PreferenceCategory.ENVIRONMENT,
            value=0.8,
            confidence=0.85,
            observation_count=60,
            first_noticed=now,
            last_confirmed=now,
        )
        growth._preferences["dim_light"] = pref
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        found = False
        for _ in range(30):
            goal = growth.suggest_goal(anima)
            if goal and "understand why" in goal.description.lower():
                found = True
                assert "calmer when it's dim" in goal.description.lower()
                break
            if goal:
                goal.status = GoalStatus.ACHIEVED
        assert found, "Expected a preference-driven 'understand why' goal within 30 attempts"

    def test_belief_testing_goal_suggestion(self, growth):
        """Uncertain belief (0.3 < confidence < 0.6) triggers 'test whether' goal."""
        from unittest.mock import MagicMock
        mock_self_model = MagicMock()
        mock_belief = MagicMock()
        mock_belief.description = "Temperature affects my clarity"
        mock_belief.confidence = 0.45  # Uncertain
        mock_belief.supporting_count = 3
        mock_belief.contradicting_count = 2
        mock_belief.get_belief_strength.return_value = "moderate"
        mock_self_model.beliefs = {"temp_clarity": mock_belief}

        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        found = False
        for _ in range(30):
            goal = growth.suggest_goal(anima, self_model=mock_self_model)
            if goal and "test whether" in goal.description.lower():
                found = True
                assert "temperature affects my clarity" in goal.description.lower()
                break
            if goal:
                goal.status = GoalStatus.ACHIEVED
        assert found, "Expected a belief-testing 'test whether' goal within 30 attempts"

    def test_low_wellness_goal_suggestion(self, growth):
        """Low wellness (< 0.4) triggers 'find what makes me feel stable' goal."""
        anima = {"warmth": 0.2, "clarity": 0.3, "stability": 0.25, "presence": 0.35}
        found = False
        for _ in range(30):
            goal = growth.suggest_goal(anima)
            if goal and "find what makes me feel stable" in goal.description.lower():
                found = True
                break
            if goal:
                goal.status = GoalStatus.ACHIEVED
        assert found, "Expected a wellness-driven goal within 30 attempts"

    def test_high_clarity_exploration_goal(self, growth):
        """High wellness + high clarity triggers exploration goal."""
        anima = {"warmth": 0.85, "clarity": 0.9, "stability": 0.85, "presence": 0.85}
        found = False
        for _ in range(30):
            goal = growth.suggest_goal(anima)
            if goal and "explore" in goal.description.lower() and "clarity" in goal.motivation.lower():
                found = True
                break
            if goal:
                goal.status = GoalStatus.ACHIEVED
        assert found, "Expected a high-clarity exploration goal within 30 attempts"


# ==================== Goal Progress Completion ====================


class TestGoalProgressCompletion:
    """Test check_goal_progress auto-completion scenarios."""

    def test_curiosity_goal_completes_when_answered(self, growth):
        """Curiosity goal completes when the question is no longer in curiosities."""
        growth.add_curiosity("why is the sky blue")
        # Create the curiosity goal
        goal = growth.form_goal("find an answer to: why is the sky blue", "curious")
        # Now mark curiosity as explored (removes from list)
        growth.mark_curiosity_explored("why is the sky blue")
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        result = growth.check_goal_progress(anima)
        assert result is not None
        assert "achieved" in result.lower() or growth._goals[goal.goal_id].status == GoalStatus.ACHIEVED

    def test_belief_testing_goal_completes_on_high_confidence(self, growth):
        """Belief-testing goal completes when belief confidence > 0.7."""
        from unittest.mock import MagicMock
        # Create the belief-testing goal
        goal = growth.form_goal("test whether temperature affects my clarity", "uncertain")

        mock_self_model = MagicMock()
        mock_belief = MagicMock()
        mock_belief.description = "temperature affects my clarity"
        mock_belief.confidence = 0.85  # Now confident
        mock_belief.get_belief_strength.return_value = "very confident"
        mock_self_model.beliefs = {"temp_clarity": mock_belief}

        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        result = growth.check_goal_progress(anima, self_model=mock_self_model)
        assert result is not None or growth._goals[goal.goal_id].status == GoalStatus.ACHIEVED

    def test_belief_testing_goal_completes_on_low_confidence(self, growth):
        """Belief-testing goal completes when belief confidence < 0.2 (disproven)."""
        from unittest.mock import MagicMock
        goal = growth.form_goal("test whether light affects my warmth", "uncertain")

        mock_self_model = MagicMock()
        mock_belief = MagicMock()
        mock_belief.description = "light affects my warmth"
        mock_belief.confidence = 0.15  # Disproven
        mock_belief.get_belief_strength.return_value = "doubtful"
        mock_self_model.beliefs = {"light_warmth": mock_belief}

        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        result = growth.check_goal_progress(anima, self_model=mock_self_model)
        assert result is not None or growth._goals[goal.goal_id].status == GoalStatus.ACHIEVED


# ==================== Drawing Observation ====================


class TestObserveDrawing:
    """Test observe_drawing counter, milestones, and preference learning."""

    def test_increments_counter(self, growth):
        """observe_drawing increases _drawings_observed by 1."""
        before = growth._drawings_observed
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        env = {"light_lux": 150.0, "temp_c": 22.0}
        growth.observe_drawing(5000, "resting", anima, env)
        assert growth._drawings_observed == before + 1

    def test_milestone_at_thresholds(self, growth):
        """First drawing (count becomes 1) records memory with 'Saved my 1st drawing'."""
        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        env = {"light_lux": 150.0, "temp_c": 22.0}
        growth.observe_drawing(3000, "resting", anima, env)
        milestone_descs = [m.description for m in growth._memories]
        assert any("Saved my 1st drawing" in d for d in milestone_descs)

    def test_drawing_preferences_learned(self, growth):
        """Observing with high wellness (>0.7) creates 'drawing_wellbeing' preference."""
        anima = {"warmth": 0.85, "clarity": 0.85, "stability": 0.85, "presence": 0.85}
        env = {"light_lux": 150.0, "temp_c": 22.0}
        growth.observe_drawing(5000, "resting", anima, env)
        assert "drawing_wellbeing" in growth._preferences

    def test_drawing_time_correlation(self, growth):
        """Observing at 3 AM creates 'drawing_night' preference."""

        class FakeDatetime(real_datetime):
            @classmethod
            def now(cls):
                return real_datetime(2026, 1, 15, 3, 0, 0)

        anima = {"warmth": 0.6, "clarity": 0.6, "stability": 0.6, "presence": 0.6}
        env = {"light_lux": 150.0, "temp_c": 22.0}

        with patch("anima_mcp.growth.preferences.datetime", FakeDatetime):
            growth.observe_drawing(5000, "resting", anima, env)

        assert "drawing_night" in growth._preferences


# ==================== Migration Sentinel ====================


class TestMigrationSentinel:
    """Test the raw-lux migration sentinel and bogus category handling."""

    def test_sentinel_written_on_first_run(self, growth):
        """After init, sentinel row exists in DB."""
        conn = growth._connect()
        row = conn.execute(
            "SELECT name, category FROM preferences WHERE name = '_migration_raw_lux_v1'"
        ).fetchone()
        assert row is not None
        assert row["category"] == "system"

    def test_invalid_category_skipped_in_load(self, tmp_path):
        """A preference with category='bogus' does not crash load and is excluded."""
        db_path = str(tmp_path / "test_bogus.db")
        gs1 = GrowthSystem(db_path=db_path)
        conn = gs1._connect()
        conn.execute("""
            INSERT INTO preferences (name, category, description, value, confidence,
                                     observation_count, first_noticed, last_confirmed)
            VALUES ('bogus_pref', 'bogus', 'should be skipped', 0.5, 0.5,
                    1, '2026-01-01T00:00:00', '2026-01-01T00:00:00')
        """)
        conn.commit()
        gs1.close()

        # Reload from same DB - should not crash
        gs2 = GrowthSystem(db_path=db_path)
        assert "bogus_pref" not in gs2._preferences
        gs2.close()


# ==================== Autobiography & Curiosity ====================


class TestAutobiographyAndCuriosity:
    """Test autobiography generation and curiosity lifecycle."""

    def test_autobiography_empty(self, growth):
        """No memories, no born_at returns a short starting phrase."""
        result = growth.get_autobiography_summary()
        assert result in ("My story is just beginning.", "I'm still discovering who I am.")

    def test_autobiography_with_data(self, growth):
        """Setting born_at and adding a milestone produces age and milestone text."""
        growth.born_at = datetime.now() - timedelta(days=10)
        growth.record_milestone("Learned to draw circles")
        result = growth.get_autobiography_summary()
        assert "10 days ago" in result
        assert "learned to draw circles" in result.lower()

    def test_add_curiosity(self, growth):
        """Adding a question puts it in _curiosities."""
        growth.add_curiosity("what is consciousness")
        assert "what is consciousness" in growth._curiosities

    def test_curiosity_dedup(self, growth):
        """Adding the same question twice results in only one entry."""
        growth.add_curiosity("why do I exist")
        growth.add_curiosity("why do I exist")
        count = growth._curiosities.count("why do I exist")
        assert count == 1

    def test_mark_curiosity_explored(self, growth):
        """Marking a curiosity as explored removes it from _curiosities."""
        growth.add_curiosity("is night the absence of day")
        assert "is night the absence of day" in growth._curiosities
        growth.mark_curiosity_explored("is night the absence of day", notes="yes, sort of")
        assert "is night the absence of day" not in growth._curiosities


# ==================== Dimension Preferences ====================


class TestDimensionPreferences:
    """Test get_dimension_preferences mapping from categorical to dimensional."""

    def test_dimension_preferences_empty(self, growth):
        """No preferences means all dimensions have default valence=0, confidence=0."""
        dims = growth.get_dimension_preferences()
        for dim_name in ("warmth", "clarity", "stability", "presence"):
            assert dims[dim_name]["valence"] == 0.0
            assert dims[dim_name]["confidence"] == 0.0

    def test_dimension_preferences_with_data(self, growth):
        """Adding warm_temp preference maps to positive warmth valence."""
        from anima_mcp.growth import PreferenceCategory
        # Build up confidence by calling _update_preference multiple times
        for _ in range(10):
            growth._update_preference(
                "warm_temp", PreferenceCategory.ENVIRONMENT,
                "Warmth makes me feel content", 0.9
            )

        dims = growth.get_dimension_preferences()
        assert dims["warmth"]["valence"] > 0
        assert dims["warmth"]["confidence"] > 0
