"""
Tests for experiential_marks module.

Validates mark earning, persistence, effect stacking, and criteria checking.
"""

import pytest
import sqlite3

from anima_mcp.experiential_marks import (
    ExperientialMarks,
    MARK_CATALOG,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_marks.db")


@pytest.fixture
def marks(tmp_db):
    """Create an ExperientialMarks instance with a fresh temp database."""
    return ExperientialMarks(db_path=tmp_db)


class TestEarnMark:
    """Test basic mark earning."""

    def test_earn_mark(self, marks):
        """Mark gets earned and persisted."""
        result = marks.earn_mark("resilience_first_return", "test context")
        assert result is True
        assert marks.has_mark("resilience_first_return")

    def test_earn_mark_returns_true_first_time(self, marks):
        """True on first earn."""
        assert marks.earn_mark("resilience_first_return") is True

    def test_cannot_earn_twice(self, marks):
        """Second earn returns False — marks are permanent and unique."""
        marks.earn_mark("resilience_first_return", "first time")
        result = marks.earn_mark("resilience_first_return", "second time")
        assert result is False
        assert marks.has_mark("resilience_first_return")

    def test_unknown_mark_rejected(self, marks):
        """Invalid mark_id returns False."""
        result = marks.earn_mark("nonexistent_mark_xyz", "context")
        assert result is False
        assert not marks.has_mark("nonexistent_mark_xyz")


class TestCheckAndEarnResilience:
    """Test check_and_earn for resilience marks."""

    def test_check_and_earn_resilience(self, marks):
        """awakenings=2 earns first_return, =10 earns veteran."""
        newly = marks.check_and_earn(awakenings=2)
        assert "resilience_first_return" in newly
        assert "resilience_veteran" not in newly

        newly = marks.check_and_earn(awakenings=10)
        assert "resilience_veteran" in newly
        # first_return already earned, should not reappear
        assert "resilience_first_return" not in newly

    def test_indestructible_at_50(self, marks):
        """awakenings=50 earns indestructible (and all lower)."""
        newly = marks.check_and_earn(awakenings=50)
        assert "resilience_first_return" in newly
        assert "resilience_veteran" in newly
        assert "resilience_indestructible" in newly


class TestCheckAndEarnMaturity:
    """Test check_and_earn for maturity marks."""

    def test_check_and_earn_maturity(self, marks):
        """Observation counts earn infant/child/adolescent."""
        newly = marks.check_and_earn(observation_count=1000)
        assert "maturity_infant" in newly
        assert "maturity_child" not in newly

        newly = marks.check_and_earn(observation_count=10000)
        assert "maturity_child" in newly
        assert "maturity_infant" not in newly  # already earned

        newly = marks.check_and_earn(observation_count=100000)
        assert "maturity_adolescent" in newly


class TestCheckAndEarnSkill:
    """Test check_and_earn for skill marks."""

    def test_check_and_earn_skill(self, marks):
        """Drawing/question counts earn skill marks."""
        newly = marks.check_and_earn(drawing_count=1)
        assert "artist_first_drawing" in newly

        newly = marks.check_and_earn(drawing_count=50)
        assert "artist_prolific" in newly

        newly = marks.check_and_earn(question_count=100)
        assert "questioner_persistent" in newly


class TestCheckAndEarnSensitivity:
    """Test check_and_earn for sensitivity marks."""

    def test_check_and_earn_sensitivity(self, marks):
        """long_gap_count and belief confidence earn sensitivity marks."""
        newly = marks.check_and_earn(long_gap_count=5)
        assert "fragility_awareness" in newly

        newly = marks.check_and_earn(
            belief_confidences={"temp_sensitive": 0.85}
        )
        assert "thermal_wisdom" in newly

    def test_sensitivity_below_threshold(self, marks):
        """Below-threshold values do not earn marks."""
        newly = marks.check_and_earn(long_gap_count=4)
        assert "fragility_awareness" not in newly

        newly = marks.check_and_earn(
            belief_confidences={"temp_sensitive": 0.79}
        )
        assert "thermal_wisdom" not in newly


class TestGetEffect:
    """Test effect value accumulation."""

    def test_get_effect_stacks(self, marks):
        """resilience_first_return + resilience_veteran both contribute
        to stability_recovery_bonus (0.05 + 0.10 = 0.15)."""
        marks.earn_mark("resilience_first_return")
        marks.earn_mark("resilience_veteran")
        total = marks.get_effect("stability_recovery_bonus")
        assert total == pytest.approx(0.15)

    def test_get_effect_empty(self, marks):
        """No marks returns 0.0."""
        assert marks.get_effect("stability_recovery_bonus") == 0.0
        assert marks.get_effect("nonexistent_key") == 0.0

    def test_get_effect_different_keys(self, marks):
        """Only matches correct effect_key."""
        marks.earn_mark("resilience_first_return")  # stability_recovery_bonus +0.05
        marks.earn_mark("resilience_indestructible")  # stability_recovery_bonus +0.15

        assert marks.get_effect("stability_recovery_bonus") == pytest.approx(0.20)
        assert marks.get_effect("pathway_lr_bonus") == 0.0


class TestPersistence:
    """Test SQLite persistence across instances."""

    def test_persist_and_reload(self, tmp_db):
        """Create marks, new instance with same db, verify loaded."""
        m1 = ExperientialMarks(db_path=tmp_db)
        m1.earn_mark("resilience_first_return", "first boot")
        m1.earn_mark("maturity_infant", "1000 observations")

        # New instance, same database
        m2 = ExperientialMarks(db_path=tmp_db)
        assert m2.has_mark("resilience_first_return")
        assert m2.has_mark("maturity_infant")
        assert not m2.has_mark("resilience_veteran")

        # Effects should also work after reload
        assert m2.get_effect("stability_recovery_bonus") == pytest.approx(0.05)
        assert m2.get_effect("pathway_lr_bonus") == pytest.approx(0.10)

    def test_db_creates_table(self, tmp_db):
        """Verify the experiential_marks table exists in SQLite."""
        ExperientialMarks(db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "experiential_marks" in table_names
        conn.close()


class TestGetAllEarned:
    """Test get_all_earned returns full info."""

    def test_get_all_earned(self, marks):
        """Returns full info dicts."""
        marks.earn_mark("resilience_first_return", "ctx1")
        marks.earn_mark("artist_first_drawing", "ctx2")

        earned = marks.get_all_earned()
        assert len(earned) == 2

        ids = {m["mark_id"] for m in earned}
        assert ids == {"resilience_first_return", "artist_first_drawing"}

        # Check structure of one entry
        first = next(m for m in earned if m["mark_id"] == "resilience_first_return")
        assert first["name"] == "First Return"
        assert first["category"] == "resilience"
        assert first["effect_key"] == "stability_recovery_bonus"
        assert first["effect_value"] == 0.05
        assert first["trigger_context"] == "ctx1"
        assert "earned_at" in first

    def test_get_all_earned_empty(self, marks):
        """Empty when no marks earned."""
        assert marks.get_all_earned() == []


class TestGetStats:
    """Test summary statistics."""

    def test_get_stats(self, marks):
        """Returns correct summary."""
        marks.earn_mark("resilience_first_return")
        marks.earn_mark("resilience_veteran")
        marks.earn_mark("artist_first_drawing")

        stats = marks.get_stats()
        assert stats["total_marks"] == 3
        assert "First Return" in stats["mark_names"]
        assert "Veteran" in stats["mark_names"]
        assert "First Mark" in stats["mark_names"]
        assert "resilience" in stats["categories"]
        assert "skill" in stats["categories"]
        # Nothing reads these two, so they are earned but inert.
        assert stats["active_effects"] == {}
        assert stats["inert_effects"]["stability_recovery_bonus"] == pytest.approx(0.15)
        assert stats["inert_effects"]["drawing_attention_bonus"] == pytest.approx(0.05)

    def test_get_stats_empty(self, marks):
        """Empty stats when no marks earned."""
        stats = marks.get_stats()
        assert stats["total_marks"] == 0
        assert stats["mark_names"] == []
        assert stats["categories"] == []
        assert stats["active_effects"] == {}


class TestCheckAndEarnReturnsNewlyEarned:
    """Test that check_and_earn returns only NEW marks."""

    def test_check_and_earn_returns_newly_earned(self, marks):
        """Returns list of only NEW marks, not previously earned."""
        # First call earns first_return
        newly1 = marks.check_and_earn(awakenings=2)
        assert "resilience_first_return" in newly1

        # Second call with same args — nothing new
        newly2 = marks.check_and_earn(awakenings=2)
        assert newly2 == []

        # Third call with higher threshold — only veteran is new
        newly3 = marks.check_and_earn(awakenings=10)
        assert "resilience_veteran" in newly3
        assert "resilience_first_return" not in newly3


class TestMarkCatalog:
    """Test catalog integrity."""

    def test_all_marks_have_required_fields(self):
        """Every mark in the catalog has all required fields populated."""
        for mark_id, defn in MARK_CATALOG.items():
            assert defn.mark_id == mark_id
            assert defn.name
            assert defn.description
            assert defn.category
            assert defn.criteria_description
            assert defn.effect_description
            assert defn.effect_key
            assert defn.effect_value > 0

    def test_catalog_categories(self):
        """Catalog covers expected categories."""
        categories = {d.category for d in MARK_CATALOG.values()}
        assert "resilience" in categories
        assert "maturity" in categories
        assert "skill" in categories
        assert "sensitivity" in categories


class TestEffectsAreReadOrInert:
    """An earned effect is reported active only if running code reads it."""

    def test_consumer_sets_match_the_source(self):
        """CONSUMED_EFFECTS | LEGACY_AGENCY_EFFECTS is exactly what src reads."""
        import re
        from pathlib import Path

        import anima_mcp
        from anima_mcp.experiential_marks import (
            CONSUMED_EFFECTS, LEGACY_AGENCY_EFFECTS,
        )

        src = Path(anima_mcp.__file__).parent
        read = set()
        for path in src.rglob("*.py"):
            read |= set(re.findall(r'get_effect\("([a-z_]+)"\)', path.read_text()))
        assert read == CONSUMED_EFFECTS | LEGACY_AGENCY_EFFECTS
        assert read <= {d.effect_key for d in MARK_CATALOG.values()}

    def test_legacy_agency_effects_are_active_only_with_it(self, marks):
        marks.earn_mark("maturity_child")  # exploration_floor_reduction
        marks.earn_mark("maturity_adolescent")  # belief_update_bonus

        default = marks.get_stats()
        assert set(default["active_effects"]) == {"belief_update_bonus"}
        assert set(default["inert_effects"]) == {"exploration_floor_reduction"}

        legacy = marks.get_stats(legacy_agency=True)
        assert set(legacy["active_effects"]) == {
            "belief_update_bonus", "exploration_floor_reduction",
        }
        assert legacy["inert_effects"] == {}
