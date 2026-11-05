"""Attention projection for agent and operator self-iteration visibility."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from anima_mcp.self_iteration import (
    ATTENTION_SCHEMA,
    PROVENANCE_SCHEMA,
    SelfIterationError,
    SelfIterationSystem,
)
from conftest import parse_result


@pytest.fixture
def system(tmp_path):
    source = tmp_path / "anima-mcp"
    files = {
        "pyproject.toml": '[project]\nname = "anima-mcp"\nversion = "9.9.9"\n',
        "README.md": "# Test creature\n",
        "src/anima_mcp/display/eras/test_era.py": "def draw():\n    return None\n",
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    fixed_now = datetime(2026, 8, 11, 21, 30, tzinfo=timezone.utc)

    def provenance_provider(recorded_at):
        return {
            "schema": PROVENANCE_SCHEMA,
            "recorded_by": "anima-mcp-test-server",
            "recorded_at": recorded_at,
            "transport": {"kind": "test", "server_observed": True},
            "authentication": {"method": "none", "verified": False},
            "actor": None,
            "session": {"present": False, "verified": False},
            "trust": {},
        }

    return SelfIterationSystem(
        repo_root=source,
        ledger_path=tmp_path / "state" / "self_iteration.json",
        clock=lambda: fixed_now,
        provenance_provider=provenance_provider,
    )


def _proposal(system, **overrides):
    values = {
        "observation": "The current era repeats the same mark.",
        "hypothesis": "A bounded seed change will reduce repetition.",
        "expected_outcome": "Duplicate marks fall below ten percent.",
        "evidence": ["Eight of twenty drawings repeated the same mark."],
        "target_paths": ["src/anima_mcp/display/eras/test_era.py"],
        "verification": ["Measure twenty transient-canary drawings."],
        "risk": "low",
    }
    values.update(overrides)
    return system.propose(**values)


def _candidate_proposal() -> dict:
    candidate_id = "sip-" + "1" * 32
    return {
        "id": "si-20260811-attention",
        "created_at": "2026-08-11T21:30:00+00:00",
        "content_sha256": "a" * 64,
        "status": "ready_for_isolated_implementation",
        "target_paths": ["src/anima_mcp/display/eras/test_era.py"],
        "verification_state": {"status": "verified"},
        "events": [
            {"type": "patch_candidate_constructed", "candidate_id": candidate_id}
        ],
    }


def _patch_snapshot() -> dict:
    return {
        "evaluations": [],
        "unledgered_evaluation_artifact_count": 0,
        "current_state": {
            "source_fingerprint_current": True,
            "eligible_for_execution_approval": False,
        },
    }


class TestAttentionProjection:
    def test_unverified_proposal_surfaces_distinct_verifier(self, system):
        proposal = _proposal(system)

        result = system.attention()

        assert result["schema"] == ATTENTION_SCHEMA
        assert result["active_count"] == 1
        item = result["items"][0]
        assert item["proposal_id"] == proposal["id"]
        assert item["state"] == "unverified"
        assert item["required_role"] == "independent_verifier"
        assert item["signed_approval_required"] is True
        assert item["status_query"]["read_only"] is True
        assert item["acknowledgement_is_approval"] is False
        assert item["authority_granted"] is False
        assert "signature" not in item
        assert item["claim_provenance"] == {
            "source_epistemic_status": "caller_claimed",
            "request_trust_classification": "unverified_request",
            "request_actor_authenticated": False,
            "claims_verified_by_request_provenance": False,
            "independent_verification_status": "unverified",
            "effective_weight": 0.0,
            "authority_granted": False,
        }
        assert (
            result["projection_provenance"]["request_provenance_verifies_claim_truth"]
            is False
        )

    def test_protected_proposal_routes_to_caretaker(self, system):
        proposal = _proposal(
            system,
            target_paths=["src/anima_mcp/identity/store.py"],
            risk="high",
        )

        item = system.attention()["items"][0]

        assert item["proposal_id"] == proposal["id"]
        assert item["state"] == "protected_review_required"
        assert item["priority"] == "high"
        assert item["required_role"] == "caretaker"

    def test_extracted_ledger_and_attention_modules_are_protected_surfaces(
        self, system
    ):
        for target in (
            "src/anima_mcp/self_iteration_attention.py",
            "src/anima_mcp/self_iteration_ledger.py",
        ):
            boundary = system.classify_target(target)
            assert boundary["boundary"] == "protected_core"
            assert boundary["auto_implementation_eligible"] is False

    def test_terminal_outcome_is_notification_only(self, system):
        proposal = _proposal(system)
        system.record_outcome(
            proposal_id=proposal["id"],
            decision="revert",
            observed_outcome="The duplicate rate increased.",
            evidence=["Canary duplicate rate was twelve percent."],
            implementation_ref="commit:deadbeef",
            claimed_measurement_source="automated_test",
        )

        result = system.attention()

        assert result["active_count"] == 0
        assert result["items"][0]["state"] == "reverted"
        assert result["items"][0]["active"] is False

    def test_indeterminate_execution_is_critical(self, system, monkeypatch):
        proposal = _candidate_proposal()
        candidate_id = proposal["events"][0]["candidate_id"]
        monkeypatch.setattr(
            system,
            "list_proposals",
            lambda **_: {"count": 1, "proposals": [proposal]},
        )
        monkeypatch.setattr(system, "patch_status", lambda **_: _patch_snapshot())
        monkeypatch.setattr(
            system,
            "execution_status",
            lambda **_: {
                "executions": [
                    {
                        "challenge_id": "sich-" + "2" * 32,
                        "issued_at": "2026-08-11T21:31:00+00:00",
                        "state": "claimed_result_indeterminate",
                    }
                ]
            },
        )
        monkeypatch.setattr(
            system, "application_status", lambda **_: {"applications": []}
        )
        monkeypatch.setattr(system, "canary_status", lambda **_: {"canaries": []})

        item = system.attention()["items"][0]

        assert item["candidate_id"] == candidate_id
        assert item["priority"] == "critical"
        assert item["state"] == "claimed_result_indeterminate"
        assert item["required_role"] == "operator_recovery"

    def test_restored_canary_surfaces_human_merge_review(self, system, monkeypatch):
        proposal = _candidate_proposal()
        candidate_id = proposal["events"][0]["candidate_id"]
        monkeypatch.setattr(
            system,
            "list_proposals",
            lambda **_: {"count": 1, "proposals": [proposal]},
        )
        monkeypatch.setattr(system, "patch_status", lambda **_: _patch_snapshot())
        monkeypatch.setattr(system, "execution_status", lambda **_: {"executions": []})
        monkeypatch.setattr(
            system, "application_status", lambda **_: {"applications": []}
        )
        monkeypatch.setattr(
            system,
            "canary_status",
            lambda **_: {
                "canaries": [
                    {
                        "challenge_id": "sich-" + "3" * 32,
                        "issued_at": "2026-08-11T21:32:00+00:00",
                        "state": "recorded",
                        "eligible_for_merge_review": True,
                        "result": {
                            "canary_result_id": "sicr-" + "4" * 32,
                            "finished_at": "2026-08-11T21:34:00+00:00",
                        },
                    }
                ]
            },
        )

        item = system.attention()["items"][0]

        assert item["candidate_id"] == candidate_id
        assert item["state"] == "eligible_for_human_merge_review"
        assert item["required_role"] == "human_merge_reviewer"
        assert item["acknowledgement_is_approval"] is False

    def test_attention_limit_is_bounded(self, system):
        with pytest.raises(SelfIterationError, match="between 1 and 50"):
            system.attention(limit=51)

    def test_attention_record_limit_bounds_candidate_reconciliation(
        self, system, monkeypatch
    ):
        proposal = _candidate_proposal()
        proposal["events"] = [
            {
                "type": "patch_candidate_constructed",
                "candidate_id": "sip-" + str(index) * 32,
            }
            for index in range(1, 4)
        ]
        monkeypatch.setattr(
            system,
            "list_proposals",
            lambda **_: {"count": 1, "proposals": [proposal]},
        )
        observed: list[str] = []

        def patch_status(**arguments):
            observed.append(arguments["candidate_id"])
            return _patch_snapshot()

        monkeypatch.setattr(system, "patch_status", patch_status)
        monkeypatch.setattr(system, "execution_status", lambda **_: {"executions": []})
        monkeypatch.setattr(
            system, "application_status", lambda **_: {"applications": []}
        )
        monkeypatch.setattr(system, "canary_status", lambda **_: {"canaries": []})

        result = system.attention(limit=2)

        assert len(result["items"]) == 2
        assert len(observed) == 2
        assert result["unexamined_candidate_count"] == 1
        assert result["truncated"] is True


@pytest.mark.asyncio
async def test_attention_action_and_lumen_context_surface(system, monkeypatch):
    _proposal(system)
    monkeypatch.setattr(
        "anima_mcp.handlers.self_iteration.get_self_iteration_system",
        lambda: system,
    )
    monkeypatch.setattr(
        "anima_mcp.self_iteration.get_self_iteration_system", lambda: system
    )
    monkeypatch.setattr("anima_mcp.accessors._get_store", lambda: None)
    monkeypatch.setattr(
        "anima_mcp.accessors._get_sensors",
        lambda: SimpleNamespace(is_pi=lambda: False),
    )
    monkeypatch.setattr(
        "anima_mcp.accessors._get_readings_and_anima", lambda: (None, None)
    )

    from anima_mcp.handlers.self_iteration import handle_self_iteration
    from anima_mcp.handlers.workflows import handle_get_lumen_context

    direct = parse_result(
        await handle_self_iteration({"action": "attention", "limit": 10})
    )
    context = parse_result(await handle_get_lumen_context({"include": ["attention"]}))

    assert direct["schema"] == ATTENTION_SCHEMA
    assert context["self_iteration_attention"]["items"] == direct["items"]


def test_extracted_pipeline_and_reflection_modules_are_protected_surfaces(system):
    for rel_path in (
        "src/anima_mcp/self_iteration_ledger.py",
        "src/anima_mcp/self_iteration_attention.py",
        "src/anima_mcp/self_iteration_pipeline.py",
        "src/anima_mcp/self_reflection_analyzers.py",
    ):
        classified = system.classify_target(rel_path)
        assert classified["boundary"] == "protected_core"
        assert classified["auto_implementation_eligible"] is False

