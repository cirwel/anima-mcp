"""question_asking_tendency is judged once per surprise episode, both ways."""

import json

import pytest

from anima_mcp.question_evidence import EPISODE_SECONDS, QuestionEpisodes


class _Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def next_episode(self):
        self.t += EPISODE_SECONDS


@pytest.fixture
def episodes():
    emitted = []
    clock = _Clock()
    tracker = QuestionEpisodes(
        emit=lambda supports, strength, source: emitted.append((supports, strength)),
        clock=clock,
    )
    return tracker, clock, emitted


def test_an_episode_is_judged_once_when_the_next_one_starts(episodes):
    tracker, clock, emitted = episodes
    for _ in range(50):
        tracker.note(0.5, asked=False, source="miss")
    assert emitted == []
    clock.next_episode()
    tracker.note(0.0, asked=False, source="miss")
    assert emitted == [(False, 0.5)]


def test_any_question_in_the_episode_makes_it_support(episodes):
    tracker, clock, emitted = episodes
    tracker.note(0.3, asked=False, source="miss")
    tracker.note(0.6, asked=True, source="post")
    tracker.note(0.4, asked=False, source="miss")
    clock.next_episode()
    tracker.note(0.0, asked=False, source="miss")
    assert emitted == [(True, 0.6)]


def test_unsurprised_posts_are_not_evidence(episodes):
    """The live failure: exploration posted questions at low surprise and each
    one supported 'I ask questions when surprised'."""
    tracker, clock, emitted = episodes
    for _ in range(5):
        tracker.note(0.05, asked=True, source="post")
        clock.next_episode()
    tracker.note(0.0, asked=False, source="miss")
    assert emitted == []


def test_support_and_contradiction_carry_the_same_weight(episodes):
    tracker, clock, emitted = episodes
    tracker.note(0.5, asked=True, source="post")
    clock.next_episode()
    tracker.note(0.5, asked=False, source="miss")
    clock.next_episode()
    tracker.note(0.0, asked=False, source="miss")
    assert emitted == [(True, 0.5), (False, 0.5)]


def test_nan_surprise_is_ignored(episodes):
    tracker, clock, emitted = episodes
    tracker.note(float("nan"), asked=True, source="post")
    clock.next_episode()
    tracker.note(0.0, asked=False, source="miss")
    assert emitted == []


def test_v8_retires_the_one_sided_tally(tmp_path):
    from anima_mcp.self_model import SelfModel

    path = tmp_path / "self_model.json"
    earlier = [
        "_migrated_noise_reset", "_migrated_episode_evidence_v2",
        "_migrated_dead_channel_reset_v3", "_migrated_light_attribution_reset_v4",
        "_migrated_clarity_semantics_reset_v5", "_migrated_led_causal_evidence_reset_v6",
        "_migrated_time_pattern_relative_reset_v7",
    ]
    path.write_text(json.dumps({
        **{flag: True for flag in earlier},
        "beliefs": {
            "question_asking_tendency": {
                "confidence": 0.722, "value": 0.811,
                "supporting_count": 111, "contradicting_count": 0,
            },
            "light_sensitive": {
                "confidence": 0.254, "value": 0.601,
                "supporting_count": 7, "contradicting_count": 198,
            },
        },
    }))
    model = SelfModel(persistence_path=path, read_only=False)

    belief = model.beliefs["question_asking_tendency"]
    assert (belief.confidence, belief.value) == (0.5, 0.7)
    assert (belief.supporting_count, belief.contradicting_count) == (0, 0)
    assert model.beliefs["light_sensitive"].contradicting_count == 198

    audit = json.loads(path.read_text())["_migrated_question_episode_reset_v8"]
    assert audit["question_asking_tendency"]["supporting_count"] == 111
    assert audit["question_asking_tendency"]["confidence"] == pytest.approx(0.722)

    # Runs once: later evidence survives a reload.
    model.apply_evidence("question_asking_tendency", supports=False, strength=0.5)
    model._save()
    again = SelfModel(persistence_path=path, read_only=False)
    assert again.beliefs["question_asking_tendency"].contradicting_count == 1
