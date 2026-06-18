"""
Tests for memory consolidation — DaySummary round-trip, consolidate(),
get_day_summaries(), detect_long_term_trend(), persistence, and reflect integration.

Run with: pytest tests/test_memory_consolidation.py -v
"""

import json
import pytest
from datetime import datetime, timedelta

from anima_mcp.anima_history import (
    AnimaHistory,
    DaySummary,
)


@pytest.fixture
def history(tmp_path):
    """Create AnimaHistory with temp persistence."""
    return AnimaHistory(
        max_size=5000,
        persistence_path=tmp_path / "anima_history.json",
        auto_save_interval=99999,  # Disable auto-save noise
    )


def _populate(history, n=200, base_warmth=0.5, base_clarity=0.5,
              base_stability=0.5, base_presence=0.5, spread=0.05):
    """Populate history with n observations around given centers."""
    import random
    random.seed(42)
    base_time = datetime(2025, 6, 15, 10, 0, 0)
    for i in range(n):
        history.record(
            warmth=base_warmth + random.uniform(-spread, spread),
            clarity=base_clarity + random.uniform(-spread, spread),
            stability=base_stability + random.uniform(-spread, spread),
            presence=base_presence + random.uniform(-spread, spread),
            timestamp=base_time + timedelta(seconds=i),
        )


# ==================== DaySummary Round-Trip ====================

class TestDaySummaryRoundTrip:
    """Test DaySummary serialization/deserialization."""

    def test_to_dict_and_back(self):
        """DaySummary survives to_dict → from_dict round-trip."""
        original = DaySummary(
            date="2025-06-15T10:00:00",
            attractor_center=[0.5, 0.6, 0.7, 0.8],
            attractor_variance=[0.001, 0.002, 0.003, 0.004],
            n_observations=200,
            time_span_hours=1.5,
            notable_perturbations=3,
            dimension_trends={"warmth": 0.5, "clarity": 0.6,
                              "stability": 0.7, "presence": 0.8},
        )
        d = original.to_dict()
        restored = DaySummary.from_dict(d)

        assert restored.date == original.date
        assert restored.attractor_center == original.attractor_center
        assert restored.attractor_variance == original.attractor_variance
        assert restored.n_observations == original.n_observations
        assert restored.notable_perturbations == original.notable_perturbations
        assert restored.dimension_trends == original.dimension_trends

    def test_to_dict_keys(self):
        """to_dict has the expected compact keys."""
        summary = DaySummary(
            date="2025-06-15", attractor_center=[0.5]*4,
            attractor_variance=[0.01]*4, n_observations=100,
            time_span_hours=1.0, notable_perturbations=0,
            dimension_trends={"warmth": 0.5},
        )
        d = summary.to_dict()
        assert set(d.keys()) == {"date", "center", "variance", "n_obs",
                                  "hours", "perturbations", "trends"}


# ==================== consolidate() ====================

class TestConsolidate:
    """Test AnimaHistory.consolidate()."""

    def test_returns_none_under_100_obs(self, history):
        """consolidate() returns None with <100 observations."""
        _populate(history, n=50)
        assert history.consolidate() is None

    def test_valid_summary_with_200_obs(self, history):
        """consolidate() returns valid DaySummary with 200 observations."""
        _populate(history, n=200, base_warmth=0.6, base_clarity=0.7)
        summary = history.consolidate()

        assert summary is not None
        assert isinstance(summary, DaySummary)
        assert summary.n_observations == 200
        assert len(summary.attractor_center) == 4
        assert len(summary.attractor_variance) == 4
        # Center should be near the populated values
        assert abs(summary.attractor_center[0] - 0.6) < 0.1  # warmth
        assert abs(summary.attractor_center[1] - 0.7) < 0.1  # clarity
        assert summary.time_span_hours > 0

    def test_saves_to_disk(self, history, tmp_path):
        """consolidate() persists the summary to day_summaries.json."""
        _populate(history, n=200)
        history.consolidate()

        summaries_path = tmp_path / "day_summaries.json"
        assert summaries_path.exists()

        data = json.loads(summaries_path.read_text())
        assert len(data["summaries"]) == 1
        assert data["summaries"][0]["n_obs"] == 200

    def test_max_30_summaries(self, history, tmp_path):
        """Only the last 30 summaries are kept on disk."""
        for i in range(35):
            _populate(history, n=150)
            history.consolidate()

        summaries_path = tmp_path / "day_summaries.json"
        data = json.loads(summaries_path.read_text())
        assert len(data["summaries"]) == 30

    def test_perturbation_count(self, history):
        """Perturbations are counted when observations are far from center."""
        import random
        random.seed(99)
        base_time = datetime(2025, 6, 15, 10, 0, 0)
        # 150 normal observations around 0.5
        for i in range(150):
            history.record(
                warmth=0.5 + random.uniform(-0.02, 0.02),
                clarity=0.5 + random.uniform(-0.02, 0.02),
                stability=0.5 + random.uniform(-0.02, 0.02),
                presence=0.5 + random.uniform(-0.02, 0.02),
                timestamp=base_time + timedelta(seconds=i),
            )
        # Add 10 outlier observations (far from center)
        for i in range(10):
            history.record(
                warmth=0.9,
                clarity=0.1,
                stability=0.9,
                presence=0.1,
                timestamp=base_time + timedelta(seconds=150 + i),
            )

        summary = history.consolidate()
        assert summary is not None
        assert summary.notable_perturbations >= 5  # Outliers should count


# ==================== get_day_summaries() ====================

class TestGetDaySummaries:
    """Test loading persisted day summaries."""

    def test_empty_returns_empty(self, history):
        """No summaries file → empty list."""
        assert history.get_day_summaries() == []

    def test_returns_stored_summaries(self, history):
        """Summaries round-trip through consolidate → get_day_summaries."""
        _populate(history, n=200)
        history.consolidate()

        summaries = history.get_day_summaries()
        assert len(summaries) == 1
        assert summaries[0].n_observations == 200

    def test_limit_parameter(self, history):
        """get_day_summaries respects the limit parameter."""
        for i in range(5):
            _populate(history, n=150)
            history.consolidate()

        assert len(history.get_day_summaries(limit=3)) == 3

    def test_newest_first_order(self, history):
        """get_day_summaries returns newest first."""
        for i in range(3):
            _populate(history, n=150, base_warmth=0.3 + i * 0.1)
            history.consolidate()

        summaries = history.get_day_summaries()
        # The last consolidation had base_warmth=0.5, first had 0.3
        assert summaries[0].attractor_center[0] > summaries[-1].attractor_center[0]


# ==================== detect_long_term_trend() ====================

class TestDetectLongTermTrend:
    """Test trend detection across day summaries."""

    def _write_summaries(self, history, values_warmth):
        """Directly write summaries with specific warmth values."""
        summaries_path = history._get_summaries_path()
        summaries = []
        for i, w in enumerate(values_warmth):
            summaries.append({
                "date": f"2025-06-{10+i:02d}T12:00:00",
                "center": [w, 0.5, 0.5, 0.5],
                "variance": [0.001]*4,
                "n_obs": 200,
                "hours": 1.0,
                "perturbations": 0,
                "trends": {"warmth": w, "clarity": 0.5,
                           "stability": 0.5, "presence": 0.5},
            })
        summaries_path.parent.mkdir(parents=True, exist_ok=True)
        summaries_path.write_text(json.dumps({"summaries": summaries}))

    def test_none_with_fewer_than_3_summaries(self, history):
        """Returns None with <3 day summaries."""
        self._write_summaries(history, [0.5, 0.6])
        assert history.detect_long_term_trend("warmth") is None

    def test_detects_upward_trend(self, history):
        """Detects increasing trend in warmth."""
        self._write_summaries(history, [0.3, 0.4, 0.5, 0.6, 0.7])
        trend = history.detect_long_term_trend("warmth")

        assert trend is not None
        assert trend["direction"] == "increasing"
        assert trend["trend"] > 0
        assert trend["dimension"] == "warmth"
        assert trend["n_summaries"] == 5

    def test_detects_downward_trend(self, history):
        """Detects decreasing trend in clarity."""
        self._write_summaries(history, [0.7, 0.6, 0.5, 0.4, 0.3])
        # Write clarity values instead — use full summaries
        summaries_path = history._get_summaries_path()
        summaries = []
        for i, c in enumerate([0.7, 0.6, 0.5, 0.4, 0.3]):
            summaries.append({
                "date": f"2025-06-{10+i:02d}T12:00:00",
                "center": [0.5, c, 0.5, 0.5],
                "variance": [0.001]*4,
                "n_obs": 200,
                "hours": 1.0,
                "perturbations": 0,
                "trends": {"warmth": 0.5, "clarity": c,
                           "stability": 0.5, "presence": 0.5},
            })
        summaries_path.write_text(json.dumps({"summaries": summaries}))

        trend = history.detect_long_term_trend("clarity")
        assert trend is not None
        assert trend["direction"] == "decreasing"
        assert trend["trend"] < 0

    def test_stable_trend(self, history):
        """Near-constant values → 'stable' direction."""
        self._write_summaries(history, [0.50, 0.50, 0.50, 0.50])
        trend = history.detect_long_term_trend("warmth")
        assert trend is not None
        assert trend["direction"] == "stable"

    def test_window_days_limits_summaries(self, history):
        """window_days parameter limits how many summaries are used."""
        self._write_summaries(history, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        trend = history.detect_long_term_trend("warmth", window_days=3)
        assert trend is not None
        assert trend["n_summaries"] == 3


# ==================== Persistence Round-Trip ====================

class TestPersistence:
    """Test day summaries survive across AnimaHistory instances."""

    def test_summaries_persist_across_instances(self, tmp_path):
        """Day summaries written by one instance readable by another."""
        h1 = AnimaHistory(
            persistence_path=tmp_path / "anima_history.json",
            auto_save_interval=99999,
        )
        _populate(h1, n=200)
        h1.consolidate()

        h2 = AnimaHistory(
            persistence_path=tmp_path / "anima_history.json",
            auto_save_interval=99999,
        )
        summaries = h2.get_day_summaries()
        assert len(summaries) == 1
        assert summaries[0].n_observations == 200


# ==================== Reflect Integration ====================

class TestReflectIntegration:
    """Test that reflect() uses long-term trends."""

    def test_reflect_generates_trend_insight(self, tmp_path):
        """reflect() generates insight from long-term trend."""
        import sqlite3
        from anima_mcp.self_reflection import SelfReflectionSystem
        from anima_mcp.anima_history import get_anima_history, reset_anima_history

        # Reset singleton and create history with temp path
        reset_anima_history()

        db_path = str(tmp_path / "reflection.db")
        # Create state_history table so analyze_patterns() doesn't crash
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS state_history (
                timestamp TEXT, warmth REAL, clarity REAL,
                stability REAL, presence REAL, sensors TEXT
            )
        """)
        conn.commit()
        conn.close()

        reflection = SelfReflectionSystem(db_path=db_path)

        # Create a history with summaries showing upward warmth trend
        history = get_anima_history()
        summaries_path = history._get_summaries_path()
        summaries_path.parent.mkdir(parents=True, exist_ok=True)

        summaries = []
        for i, w in enumerate([0.3, 0.4, 0.5, 0.6, 0.7]):
            summaries.append({
                "date": f"2025-06-{10+i:02d}T12:00:00",
                "center": [w, 0.5, 0.5, 0.5],
                "variance": [0.001]*4,
                "n_obs": 200,
                "hours": 1.0,
                "perturbations": 0,
                "trends": {"warmth": w, "clarity": 0.5,
                           "stability": 0.5, "presence": 0.5},
            })
        summaries_path.write_text(json.dumps({"summaries": summaries}))

        # Run reflect — it should pick up the trend
        reflection.reflect()

        # Check that trend insight was created
        trend_insights = [
            i for i in reflection._insights.values()
            if "trend" in i.id
        ]
        assert len(trend_insights) >= 1
        assert any("warmth" in i.description and "increasing" in i.description
                    for i in trend_insights)

        # Cleanup singleton
        reset_anima_history()
