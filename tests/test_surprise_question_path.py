"""
End-to-end tests for the surprise-driven curiosity path (#189).

From 2026-02-01 (2d94c1f) to 2026-08-22, server.py's surprise block read
``prediction_error.predicted`` / ``.actual`` — attributes PredictionError has
never defined — and the AttributeError was swallowed by the main loop's
blanket except. Question posting, curiosity credit, growth curiosities, and
SUPPORTING question_asking_tendency evidence were dead for six months (the
no-question branch kept filing negative evidence — a one-sided stream).
These tests drive the extracted ``handle_surprise_question`` helper with REAL
PredictionError objects and a REAL MetacognitiveMonitor, so any drift between
the helper and the metacognition dataclasses fails loudly instead of being
eaten by the loop's exception handler.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from anima_mcp.metacognition import MetacognitiveMonitor, Prediction, PredictionError
from anima_mcp.server import handle_surprise_question


def _real_error(surprise=0.5, sources=("light", "warmth")):
    prediction = Prediction(
        timestamp=datetime.now(),
        light_lux=50.0, warmth=0.5, clarity=0.7,
    )
    return PredictionError(
        timestamp=datetime.now(),
        prediction=prediction,
        actual_light_lux=200.0,
        actual_warmth=0.8,
        error_light=0.6,
        error_warmth=0.4,
        surprise=surprise,
        surprise_sources=list(sources),
    )


@pytest.fixture
def metacog(tmp_path):
    return MetacognitiveMonitor(data_dir=str(tmp_path))


class TestSurpriseQuestionPath:
    def test_posts_question_and_records_curiosity(self, metacog):
        error = _real_error()
        growth = MagicMock()
        with patch("anima_mcp.messages.add_question", return_value={"id": "q1"}) as add_q, \
             patch("anima_mcp.question_evidence.note_question") as note, \
             patch("anima_mcp.accessors._get_growth", return_value=growth):
            posted = handle_surprise_question(metacog, error)

        assert posted is True
        add_q.assert_called_once()
        _, kwargs = add_q.call_args
        assert kwargs["author"] == "lumen"
        # Context derives from surprise_sources, not the nonexistent
        # predicted/actual dicts.
        assert "surprise=0.50" in kwargs["context"]
        assert "light changed unexpectedly" in kwargs["context"]
        # Curiosity credit recorded for the surprising domains.
        assert metacog._curiosity_log, "record_curiosity did not run"
        logged_domains = {d for e in metacog._curiosity_log for d in e["domains"]}
        assert "light" in logged_domains
        # Growth curiosity seeded with the posted question.
        growth.add_curiosity.assert_called_once()
        posted_question = add_q.call_args.args[0]
        assert growth.add_curiosity.call_args.args[0] == posted_question
        # The surprise episode is noted as one that became a question; the
        # episode tracker (question_evidence) judges it once.
        note.assert_called_once()
        assert note.call_args.args[0] == pytest.approx(0.5)
        assert note.call_args.kwargs["asked"] is True

    def test_real_prediction_error_has_no_legacy_fields(self):
        # The regression #189 fixed: the old block read .predicted/.actual.
        # Pin that those attributes do not exist, so any revival of the old
        # access pattern fails at the dataclass, not inside a blanket except.
        error = _real_error()
        assert not hasattr(error, "predicted")
        assert not hasattr(error, "actual")

    def test_no_question_notes_a_missed_episode(self, metacog):
        error = _real_error(surprise=0.5)
        with patch.object(metacog, "generate_curiosity_question", return_value=None), \
             patch("anima_mcp.question_evidence.note_question") as note:
            posted = handle_surprise_question(metacog, error)
        assert posted is False
        note.assert_called_once()
        assert note.call_args.kwargs["asked"] is False
        assert note.call_args.kwargs["source"] == "surprise_unasked"

    def test_rate_limited_question_is_a_missed_episode(self, metacog):
        # add_question returning None must not credit curiosity. (This
        # injects the None; the board's actual rate-limit/dedup gates are
        # covered by tests/test_messages.py — here the contract under test
        # is the helper's handling of a rejected post.)
        error = _real_error()
        with patch("anima_mcp.messages.add_question", return_value=None), \
             patch("anima_mcp.question_evidence.note_question") as note:
            posted = handle_surprise_question(metacog, error)
        assert posted is False
        assert not metacog._curiosity_log
        # The surprise did not become a question. This used to record
        # nothing, which left only the support side reachable on this path.
        note.assert_called_once()
        assert note.call_args.kwargs["asked"] is False
        assert note.call_args.kwargs["source"] == "surprise_suppressed"

    def test_domain_weights_parked_by_default(self, metacog):
        # The credit rule has never run in production and is structurally
        # unsound (saturating ratchet); weight movement stays off until
        # redesigned. Entries past the horizon retire unevaluated.
        error = _real_error()
        metacog._save_counter = 10
        metacog.record_curiosity(["light"], error)
        metacog._save_counter = 10 + metacog._eval_horizon + 1
        metacog._evaluate_curiosity_outcomes(_real_error())
        assert metacog._domain_weights == {}
        assert metacog._curiosity_log == []

    def test_call_site_wired_in_loop(self):
        # The original bug lived at the loop call site, which unit tests of
        # the helper cannot see. Pin that the loop still calls the helper.
        import inspect
        import anima_mcp.server as server
        src = inspect.getsource(server._update_display_loop)
        assert "handle_surprise_question(metacog, prediction_error)" in src

    def test_empty_sources_still_posts_with_bare_context(self, metacog):
        # Deterministic: pin the generated question so the posted branch
        # always runs, then assert the bare-context fallback.
        error = _real_error(sources=())
        with patch.object(metacog, "generate_curiosity_question",
                          return_value="what changed just now?"), \
             patch("anima_mcp.messages.add_question", return_value={"id": "q1"}) as add_q, \
             patch("anima_mcp.learning_events.enqueue_self_belief_evidence"), \
             patch("anima_mcp.accessors._get_growth", return_value=None):
            posted = handle_surprise_question(metacog, error)
        assert posted is True
        assert add_q.call_args.kwargs["context"] == "surprise=0.50"
