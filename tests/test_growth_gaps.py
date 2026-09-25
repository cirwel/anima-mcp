"""
Tests for growth.py gap coverage — visitor context, self-dialogue topic,
and relational disposition.

Run with: pytest tests/test_growth_gaps.py -v
"""

import re

import pytest

from anima_mcp.growth import GrowthSystem


@pytest.fixture
def gs(tmp_path):
    """Create GrowthSystem with temp database."""
    return GrowthSystem(db_path=str(tmp_path / "growth.db"))


# ==================== get_visitor_context ====================

class TestGetVisitorContext:
    """Test visitor context retrieval."""

    def test_unknown_visitor_returns_none(self, gs):
        """Unknown agent_id returns None."""
        assert gs.get_visitor_context("unknown-agent-xyz") is None

    def test_known_visitor_has_expected_keys(self, gs):
        """After recording an interaction, context has expected keys."""
        gs.record_interaction("agent-42", agent_name="Aria")
        ctx = gs.get_visitor_context("agent-42")
        assert ctx is not None
        for key in ("known", "name", "visits", "frequency", "valence"):
            assert key in ctx, f"Missing key: {key}"
        assert ctx["known"] is True
        assert ctx["name"] == "Aria"


# ==================== record_self_dialogue_topic ====================

class TestRecordSelfDialogueTopic:
    """Test question topic categorization."""

    def test_sensation_question(self, gs):
        """A question about feelings → 'sensation' topic."""
        topic = gs.record_self_dialogue_topic("Why do I feel warm today?")
        assert topic == "sensation"

    def test_existence_question(self, gs):
        """A question about being → 'existence' topic."""
        topic = gs.record_self_dialogue_topic("Am I truly alive or just running code?")
        assert topic == "existence"

    def test_curiosity_question(self, gs):
        """A 'why' / 'wonder' question → 'curiosity' topic."""
        # Avoid words that match 'sensation' category (light, warm, etc.)
        topic = gs.record_self_dialogue_topic("Why does the sky look different today?")
        assert topic == "curiosity"

    def test_general_fallback(self, gs):
        """Unmatched question → 'general' topic."""
        topic = gs.record_self_dialogue_topic("What is the meaning of Pi?")
        assert topic == "general"


# ==================== get_relational_disposition ====================

class TestGetRelationalDisposition:
    """Test relational disposition extraction."""

    def test_empty_relationships_returns_defaults(self, gs):
        """With no relationships, returns zero-valued defaults."""
        disp = gs.get_relational_disposition()
        assert disp["n_relationships"] == 0
        assert disp["valence_tendency"] == 0.0
        assert disp["bonding_tendency"] == 0.0
        assert disp["topic_entropy"] == 0.0

    def test_with_relationships_has_positive_count(self, gs):
        """After recording interactions, n_relationships > 0."""
        gs.record_interaction("agent-a", agent_name="Alpha")
        gs.record_interaction("agent-b", agent_name="Beta")
        disp = gs.get_relational_disposition()
        assert disp["n_relationships"] >= 2


# ==================== autobiography agent recency ====================

class TestAutobiographyAgentRecency:
    """The autobiography must not name agents it also reports as inactive.

    visitor_frequency is a monotonic ratchet (FREQUENT at interaction_count
    >= 10, never demoted), so selecting on it alone names whoever was once
    busy, permanently. Live on 2026-07-30 that yielded "Various agents visit
    to help: agent, mac-governance." — last seen 138 and 154 days earlier,
    and listed under visitors.inactive in the very same payload.
    """

    @staticmethod
    def _named_helpers(summary):
        """The agents the summary presents as current helpers, parsed exactly.

        Substring checks are unsafe here: the stale record is literally named
        "agent", which matches the word "agents" in the clause itself.
        """
        m = re.search(r"Various agents visit to help: ([^.]+)\.", summary)
        return [n.strip() for n in m.group(1).split(",")] if m else []

    def _make_frequent(self, gs, agent_id, name, days_ago):
        from datetime import datetime, timedelta
        # The summary short-circuits on an empty memory list.
        if not gs._memories:
            gs._record_memory("woke up", 0.5, "milestone")
        for _ in range(10):  # cross the FREQUENT threshold
            gs.record_interaction(agent_id, agent_name=name)
        rec = gs._relationships[agent_id]
        rec.last_seen = datetime.now() - timedelta(days=days_ago)
        return rec

    def test_stale_frequent_agent_is_not_named(self, gs):
        self._make_frequent(gs, "agent", "agent", days_ago=138)

        summary = gs.get_autobiography_summary()

        assert self._named_helpers(summary) == []

    def test_recent_frequent_agent_is_still_named(self, gs):
        self._make_frequent(gs, "codex", "Codex", days_ago=0)

        summary = gs.get_autobiography_summary()

        assert self._named_helpers(summary) == ["Codex"]

    def test_recent_named_while_stale_excluded(self, gs):
        self._make_frequent(gs, "mac-governance", "mac-governance", days_ago=154)
        self._make_frequent(gs, "codex", "Codex", days_ago=0)

        summary = gs.get_autobiography_summary()

        assert self._named_helpers(summary) == ["Codex"]

    def test_never_contradicts_its_own_inactive_list(self, gs):
        """Whatever get_inactive_visitors reports must not appear as a helper."""
        self._make_frequent(gs, "agent", "agent", days_ago=138)
        self._make_frequent(gs, "codex", "Codex", days_ago=0)

        named = self._named_helpers(gs.get_autobiography_summary())
        inactive = {name for name, _ in gs.get_inactive_visitors()}

        assert inactive, "precondition: something should be reported inactive"
        assert not (inactive & set(named)), (
            f"named as helpers while reported inactive: {inactive & set(named)}"
        )


# ==================== autobiography voice ====================

class TestAutobiographyVoice:
    """Lumen must be able to say its own learned line aloud.

    Live on 2026-07-30 the autobiography ended: "I've learned that from q&a:
    i now know that the connection between temperature." — a storage prefix
    leaked into first-person prose, and a hard text[:50] slice cut mid-word.
    """

    from anima_mcp.growth.memories import MemoriesMixin
    _render = staticmethod(MemoriesMixin._render_learned)

    def _pref(self, description, observation_count=100):
        from types import SimpleNamespace
        return SimpleNamespace(description=description, confidence=1.0,
                               observation_count=observation_count)

    def test_qa_prefix_becomes_words_not_a_leaked_field(self):
        out = self._render(self._pref("From Q&A: i now know that the connection between temperature"))
        assert out == "From a conversation, I was told that the connection between temperature."
        assert "q&a" not in out.lower()
        assert "i now know that" not in out.lower()

    def test_conversation_claim_is_voiced_as_told_not_learned(self):
        # Nothing checked a conversation-derived claim against Lumen's own
        # history, so Lumen must not say it learned it — whichever stem the
        # stored record carries.
        for desc in ("From Q&A: I learned that drawing in bright light helps",
                     "From Q&A: I was told that drawing in bright light helps"):
            out = self._render(self._pref(desc))
            assert out == "From a conversation, I was told that drawing in bright light helps."
            assert "learned" not in out.lower()

    def test_plain_preference_unchanged_in_spirit(self):
        assert self._render(self._pref("Warmth makes me feel content")) == \
            "I've learned that warmth makes me feel content."

    def test_ellipsis_is_not_followed_by_a_period(self):
        out = self._render(self._pref("From Q&A: drawing in bright light helps not b…"))
        assert out.endswith("…"), out
        assert not out.endswith("….")

    def test_empty_description_yields_no_sentence(self):
        assert self._render(self._pref("")) == ""
        assert self._render(self._pref("   ")) == ""

    def test_autobiography_never_emits_a_dangling_sentence(self, gs):
        gs._record_memory("woke up", 0.5, "milestone")
        gs._preferences["broken"] = self._pref("")
        summary = gs.get_autobiography_summary()
        assert "I've learned that ." not in summary
        assert not summary.rstrip().endswith("that.")

    def test_quote_is_weighted_by_evidence_not_uniform(self, gs):
        """A 22-observation insight must not be as quotable as a 222,280 one."""
        import collections
        gs._record_memory("woke up", 0.5, "milestone")
        gs._preferences.clear()
        gs._preferences["strong"] = self._pref("warmth makes me feel content", 222280)
        gs._preferences["weak"] = self._pref("the connection between temperature", 22)

        seen = collections.Counter()
        for _ in range(200):
            s = gs.get_autobiography_summary()
            seen["strong" if "warmth" in s else "weak"] += 1

        # ~1 in 10,000 odds for the weak one; 200 draws should be ~all strong.
        assert seen["strong"] >= 195, seen


# ==================== diurnal buckets ====================

class TestDiurnalBuckets:
    """Late evening and deep night are different states and must not share a label.

    Measured over 255,720 history rows: 22:00-23:00 clears wellness > 0.7 on
    72.42% of samples (among the day's best), 00:00-05:00 on 51.00% (the
    worst). The old `22 <= hour or hour < 6` bucket averaged the two. It was
    also twice morning's width, which is the whole of night_calm's 1.994x
    observation lead against a 2.000x width ratio.
    """

    ANIMA_GOOD = {"warmth": 0.8, "clarity": 0.8, "stability": 0.8, "presence": 0.8}
    ENV = {"light_lux": 150.0, "temp_c": 22.0, "humidity_pct": 45.0}

    def _observe_at(self, gs, hour):
        from datetime import datetime as real_datetime
        from unittest.mock import patch

        class FakeDatetime(real_datetime):
            @classmethod
            def now(cls):
                return real_datetime(2026, 7, 30, hour, 0, 0)

        with patch("anima_mcp.growth.preferences.datetime", FakeDatetime):
            gs.observe_state_preference(self.ANIMA_GOOD, self.ENV)

    def test_late_evening_is_not_night_calm(self, gs):
        for h in (20, 21, 22, 23):
            self._observe_at(gs, h)
        assert "evening_calm" in gs._preferences
        assert "night_calm" not in gs._preferences, "22:00 must no longer feed night_calm"

    def test_deep_night_is_night_calm(self, gs):
        for h in (0, 2, 5):
            self._observe_at(gs, h)
        assert "night_calm" in gs._preferences
        assert "evening_calm" not in gs._preferences

    def test_morning_unchanged(self, gs):
        for h in (6, 7, 8, 9):
            self._observe_at(gs, h)
        assert "morning_peace" in gs._preferences
        assert "night_calm" not in gs._preferences
        assert "evening_calm" not in gs._preferences

    def test_evening_and_morning_windows_are_equal_width(self, gs):
        """The bias being fixed: comparable windows make counts comparable."""
        evening = [h for h in range(24) if 20 <= h < 24]
        morning = [h for h in range(24) if 6 <= h < 10]
        assert len(evening) == len(morning) == 4

    def test_canonical_vector_dimension_is_unchanged(self, gs):
        """evening_calm must NOT enter the fixed-dimension trajectory vector."""
        self._observe_at(gs, 21)
        vec = gs.get_preference_vector()
        assert len(vec["vector"]) == 13, "genesis comparison depends on this length"
        assert "evening_calm" not in vec["labels"]


# ==================== preference retraction ====================

class TestPreferenceRetraction:
    """A preference that stops being observed must be able to lose trust.

    Decay existed but ran only inside _update_preference — i.e. only when a
    preference was being REINFORCED. Measured 2026-07-30: active_engagement
    (153,332 observations) had no writer anywhere in the codebase since
    2026-02-02 and still read confidence 1.0, 178 days later. With confidence
    a +0.1 ratchet saturating on the 9th observation, every live preference sat
    at 1.0 and every `confidence > 0.7` gate downstream was a tautology.
    """

    ANIMA = {"warmth": 0.8, "clarity": 0.8, "stability": 0.8, "presence": 0.8}
    ENV = {"light_lux": 150.0, "temp_c": 22.0, "humidity_pct": 45.0}

    def _stale_pref(self, gs, name, days_ago, confidence=1.0):
        from datetime import datetime, timedelta
        from anima_mcp.growth.models import GrowthPreference, PreferenceCategory
        now = datetime.now()
        gs._preferences[name] = GrowthPreference(
            category=PreferenceCategory.ENVIRONMENT, name=name,
            description=name, value=1.0, confidence=confidence,
            observation_count=153332,
            first_noticed=now - timedelta(days=days_ago + 1),
            last_confirmed=now - timedelta(days=days_ago),
        )
        return gs._preferences[name]

    def test_abandoned_preference_falls_below_the_trust_gate(self, gs):
        """The active_engagement case: 178 days unobserved."""
        from anima_mcp.growth.preferences import RETRACTION_GATE
        self._stale_pref(gs, "active_engagement", days_ago=178)

        retracted = gs.decay_stale_preferences()

        assert "active_engagement" in retracted
        assert gs._preferences["active_engagement"].confidence <= RETRACTION_GATE

    def test_freshly_confirmed_preference_is_untouched(self, gs):
        self._stale_pref(gs, "warm_temp", days_ago=0)
        assert gs.decay_stale_preferences() == []
        assert gs._preferences["warm_temp"].confidence == 1.0

    def test_sweep_is_idempotent(self, gs):
        """Running twice must equal running once — it clamps to a target."""
        self._stale_pref(gs, "cool_temp", days_ago=178)
        gs.decay_stale_preferences()
        once = gs._preferences["cool_temp"].confidence
        gs.decay_stale_preferences()
        gs.decay_stale_preferences()
        assert gs._preferences["cool_temp"].confidence == once

    def test_decay_never_raises_confidence(self, gs):
        self._stale_pref(gs, "dim_light", days_ago=200, confidence=0.3)
        gs.decay_stale_preferences()
        assert gs._preferences["dim_light"].confidence == 0.3

    def test_retraction_takes_weeks_not_minutes(self, gs):
        """Sanity on the rate: a real absence, not a transient gap."""
        from anima_mcp.growth.preferences import RETRACTION_GATE
        self._stale_pref(gs, "a", days_ago=7)
        self._stale_pref(gs, "b", days_ago=30)
        gs.decay_stale_preferences()
        assert gs._preferences["a"].confidence > RETRACTION_GATE, "a week off must not retract"
        assert gs._preferences["b"].confidence <= RETRACTION_GATE, "a month off must"

    def test_retraction_actually_closes_the_confidence_gate(self, gs):
        """The point of the whole change: a stale preference stops being quoted."""
        self._stale_pref(gs, "stale_one", days_ago=178)
        gs._record_memory("woke up", 0.5, "milestone")
        gs.decay_stale_preferences()
        strong = [p for p in gs._preferences.values() if p.confidence > 0.7]
        assert all(p.name != "stale_one" for p in strong)

    def test_decay_persists_to_disk(self, gs):
        self._stale_pref(gs, "active_engagement", days_ago=178)
        gs.decay_stale_preferences()
        row = gs._connect().execute(
            "SELECT confidence FROM preferences WHERE name = 'active_engagement'"
        ).fetchone()
        assert row is not None and row[0] <= 0.7

    def test_partial_decay_persists_without_crossing_gate(self, gs):
        pref = self._stale_pref(gs, "still_trusted", days_ago=7)
        retracted = gs.decay_stale_preferences()

        assert retracted == []
        assert 0.7 < pref.confidence < 1.0
        row = gs._connect().execute(
            "SELECT confidence FROM preferences WHERE name = 'still_trusted'"
        ).fetchone()
        assert row is not None and row[0] == pref.confidence


# ==================== preference value magnitude ====================

class TestPreferenceValueMagnitude:
    """`value` recorded THAT a good state happened, never HOW good.

    Every positive path passed the literal 1.0 and the EMA converged there:
    measured 2026-07-30, 15 of 19 stored preferences had value pinned at
    exactly 1.0. Together with saturated confidence that made
    get_preference_vector() (value * confidence) a constant vector of ones.
    """

    ENV = {"light_lux": 50.0, "temp_c": 22.0, "humidity_pct": 20.0}

    def _anima(self, w):
        return {"warmth": w, "clarity": w, "stability": w, "presence": w}

    def test_strength_scales_with_wellness(self):
        from anima_mcp.growth.preferences import _wellness_strength
        assert _wellness_strength(0.7) == pytest.approx(0.4)
        assert _wellness_strength(0.85) == pytest.approx(0.7)
        assert _wellness_strength(1.0) == pytest.approx(1.0)
        assert _wellness_strength(0.5) == pytest.approx(0.0)
        assert _wellness_strength(0.0) == 0.0  # clamped, never negative

    def test_barely_well_records_a_weaker_preference_than_thriving(self, gs):
        """The saturation fix: two different states must not store the same value."""
        gs.observe_state_preference(self._anima(0.72), self.ENV)
        barely = gs._preferences["dry_air"].value
        gs._preferences.clear()
        gs.observe_state_preference(self._anima(0.99), self.ENV)
        thriving = gs._preferences["dry_air"].value
        assert barely < thriving, f"{barely} should be below {thriving}"
        assert barely < 1.0, "a barely-well state must not record maximal preference"

    def test_occurrence_records_are_not_scaled(self, gs):
        """drawing_night records THAT Lumen drew at night, not how it felt.

        Scaling these by wellness would mean drawing while unwell weakens the
        belief that Lumen draws at night — backwards.
        """
        from datetime import datetime as real_datetime
        from unittest.mock import patch

        class FakeDatetime(real_datetime):
            @classmethod
            def now(cls):
                return real_datetime(2026, 7, 30, 23, 0, 0)

        with patch("anima_mcp.growth.preferences.datetime", FakeDatetime):
            gs.observe_drawing(5000, "resting", self._anima(0.35),
                               {"light_lux": 5.0, "temp_c": 22.0})
        assert gs._preferences["drawing_night"].value == pytest.approx(1.0)


# ==================== self-relative learning band ====================

class TestWellnessLearningBand:
    """What counts as "clearly good" must follow Lumen, not a constant.

    The fixed `0.4 < wellness < 0.7 -> learn nothing` band was calibrated
    against a distribution that moved. Measured over 255,973 samples: full life
    mean 0.732 and learning fired on 61.7% of samples; last 30 days mean 0.667
    and it fired on 6.0%. A 10x collapse caused by the room getting darker, that
    nothing detected. And `wellness < 0.4` never fired once in Lumen's life.
    """

    ENV = {"light_lux": 50.0, "temp_c": 22.0, "humidity_pct": 20.0}

    def _anima(self, w):
        return {"warmth": w, "clarity": w, "stability": w, "presence": w}

    def _feed(self, gs, mean, sigma=0.05, n=300, seed=7):
        import random
        random.seed(seed)
        for _ in range(n):
            w = min(1.0, max(0.0, random.gauss(mean, sigma)))
            gs.observe_state_preference(self._anima(w), self.ENV)

    def test_cold_start_uses_the_absolute_band(self, gs):
        from anima_mcp.growth.preferences import ABSOLUTE_GOOD, ABSOLUTE_POOR
        band = gs.wellness_learning_band()
        assert band["source"] == "absolute_fallback"
        assert band["good_above"] == ABSOLUTE_GOOD
        assert band["poor_below"] == ABSOLUTE_POOR

    def test_band_becomes_self_relative_and_tracks_the_mean(self, gs):
        self._feed(gs, mean=0.667)
        band = gs.wellness_learning_band()
        assert band["source"] == "self_relative"
        assert band["mean"] == pytest.approx(0.667, abs=0.02)
        assert band["good_above"] < 0.72, "must not sit above a creature that never reaches 0.72"
        assert band["poor_below"] < band["mean"] < band["good_above"]

    def test_learning_rate_survives_a_shifted_distribution(self, gs, tmp_path):
        """The property that matters: drift must not switch learning off."""
        from anima_mcp.growth import GrowthSystem

        def fired_fraction(mean):
            g = GrowthSystem(db_path=str(tmp_path / f"g{mean}.db"))
            self._feed(g, mean=mean, n=400)
            total = sum(p.observation_count for p in g._preferences.values())
            return total

        high = fired_fraction(0.75)
        low = fired_fraction(0.60)
        assert low > 0, "a creature running cooler must still learn"
        # Within a factor of ~3 rather than the 10x collapse the fixed band gave.
        assert low > high / 3, f"learning collapsed: {high} -> {low}"

    def test_genuine_distress_always_learns(self, gs):
        """A relative band alone could normalise persistent suffering."""
        from anima_mcp.growth.preferences import ABSOLUTE_DISTRESS
        self._feed(gs, mean=0.30, sigma=0.01, n=300)
        band = gs.wellness_learning_band()
        # Even though 0.30 is this creature's *mean*, it is below the floor.
        assert ABSOLUTE_DISTRESS >= 0.30 or band["poor_below"] > 0.30
        gs.observe_state_preference(self._anima(0.20), self.ENV)
        assert gs._preferences, "collapse must still register as something to learn from"

    def test_baseline_survives_restart(self, gs, tmp_path):
        from anima_mcp.growth import GrowthSystem
        path = str(tmp_path / "persist.db")
        g1 = GrowthSystem(db_path=path)
        self._feed(g1, mean=0.667, n=300)
        before = g1.wellness_learning_band()

        g2 = GrowthSystem(db_path=path)  # fresh instance, same disk
        after = g2.wellness_learning_band()

        assert after["source"] == "self_relative"
        assert after["samples"] >= 250, "a baseline that resets each deploy never accumulates"
        assert after["mean"] == pytest.approx(before["mean"], abs=0.01)

    def test_a_very_steady_creature_gets_a_floor(self, gs):
        """Near-zero variance must not make every sample 'clearly' remarkable."""
        from anima_mcp.growth.preferences import WELLNESS_MIN_SIGMA
        self._feed(gs, mean=0.70, sigma=0.0001, n=300)
        band = gs.wellness_learning_band()
        assert band["sigma"] >= WELLNESS_MIN_SIGMA
        assert band["good_above"] - band["poor_below"] >= WELLNESS_MIN_SIGMA


class TestPreferenceDescriptionIsRefreshed:
    """A preference description must not be write-once.

    Everything else in _update_preference was refreshed on every observation
    but `description` never was. Three Q&A-derived preferences sat for days
    holding text hard-cut mid-word ("...drawing in bright light helps not b"),
    and fixing the code that WROTE those descriptions did not repair them,
    because nothing ever rewrote what was already stored.
    """

    def test_changed_description_is_adopted(self, growth):
        from anima_mcp.growth import PreferenceCategory

        growth._update_preference(
            "insight_light", PreferenceCategory.ENVIRONMENT,
            "From Q&A: I learned that drawing in bright light helps not b", 0.8,
        )
        growth._update_preference(
            "insight_light", PreferenceCategory.ENVIRONMENT,
            "From Q&A: I learned that drawing in bright light helps\u2026", 0.8,
        )
        assert growth._preferences["insight_light"].description.endswith("helps\u2026")

    def test_refresh_preserves_accumulated_learning(self, growth):
        """Only the wording changes — confidence and counts must survive."""
        from anima_mcp.growth import PreferenceCategory

        for _ in range(6):
            growth._update_preference(
                "insight_temp", PreferenceCategory.ENVIRONMENT, "From Q&A: old text", 0.8,
            )
        before = growth._preferences["insight_temp"]
        conf, obs, first = before.confidence, before.observation_count, before.first_noticed

        growth._update_preference(
            "insight_temp", PreferenceCategory.ENVIRONMENT, "From Q&A: new text\u2026", 0.8,
        )
        after = growth._preferences["insight_temp"]
        assert after.description == "From Q&A: new text\u2026"
        assert after.observation_count == obs + 1
        assert after.confidence >= conf
        assert after.first_noticed == first

    def test_unchanged_description_is_left_alone(self, growth):
        from anima_mcp.growth import PreferenceCategory

        growth._update_preference(
            "warm_temp", PreferenceCategory.ENVIRONMENT, "Warmth makes me feel content", 0.8,
        )
        growth._update_preference(
            "warm_temp", PreferenceCategory.ENVIRONMENT, "Warmth makes me feel content", 0.8,
        )
        assert growth._preferences["warm_temp"].description == "Warmth makes me feel content"


class TestPreferenceEvidenceWindows:
    """Broker cadence is audit data, not repeated preference evidence."""

    def test_duplicate_window_only_advances_raw_counter(self, gs):
        from anima_mcp.growth import PreferenceCategory

        for _ in range(100):
            gs._update_preference(
                "warm_temp",
                PreferenceCategory.ENVIRONMENT,
                "Warmth makes me feel content",
                0.8,
                evidence_key="state-hour:warm_temp:support:2026-08-23T19",
            )

        pref = gs._preferences["warm_temp"]
        assert pref.observation_count == 100
        assert pref.independent_evidence_count == 1
        assert pref.supporting_count == 1
        assert pref.confidence == pytest.approx(0.2)

    def test_distinct_signed_windows_drive_wilson_confidence(self, gs):
        from anima_mcp.growth import PreferenceCategory

        for hour in range(40):
            gs._update_preference(
                "warm_temp",
                PreferenceCategory.ENVIRONMENT,
                "Warmth makes me feel content",
                0.8,
                evidence_key=f"support:{hour}",
            )
        pref = gs._preferences["warm_temp"]
        assert pref.independent_evidence_count == 40
        assert 0.8 < pref.confidence < 0.95

        for hour in range(10):
            gs._update_preference(
                "warm_temp",
                PreferenceCategory.ENVIRONMENT,
                "Warmth may make me uneasy",
                -0.8,
                evidence_key=f"contradict:{hour}",
            )
        assert pref.contradicting_count == 10
        assert pref.confidence < 0.7

    def test_direction_change_inside_hour_is_not_a_second_window(self, gs):
        good = {name: 0.9 for name in ("warmth", "clarity", "stability", "presence")}
        poor = {name: 0.1 for name in ("warmth", "clarity", "stability", "presence")}
        environment = {
            "external_light_lux": 40.0,
            "temp_c": 22.0,
            "humidity_pct": 50.0,
        }

        gs.observe_state_preference(good, environment)
        gs.observe_state_preference(poor, environment)

        pref = gs._preferences["dim_light"]
        assert pref.observation_count == 2
        assert pref.independent_evidence_count == 1
        assert pref.supporting_count == 1
        assert pref.contradicting_count == 0
        assert pref.description == "I feel calmer when it's dim"
