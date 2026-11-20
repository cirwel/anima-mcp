"""Unit tests for shared self-iteration stage claim and challenge helpers."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from anima_mcp.self_iteration_application import ApplicationError
from anima_mcp.self_iteration_application_stage import _APPLICATION_CLAIM_SPEC
from anima_mcp.self_iteration_canary import CanaryError
from anima_mcp.self_iteration_canary_stage import _CANARY_CLAIM_SPEC
from anima_mcp.self_iteration_execution import ExecutionError
from anima_mcp.self_iteration_execution_stage import _EXECUTION_CLAIM_SPEC
from anima_mcp.self_iteration_ledger import SelfIterationError
from anima_mcp.self_iteration_stage_helpers import (
    _StageClaimSpec,
    _build_stage_claim,
    _stage_challenge_event,
    _validate_stage_claim,
)
from anima_mcp.self_iteration_verification import VerifierKey


class _FakeSigError(ValueError):
    pass


def _make_spec(
    *,
    stage: str = "demo",
    parent_id_field: str | None = "parent_id",
    extra_false_fields: tuple[str, ...] = ("persistent_activation_allowed",),
    key_registry_msg: str = "demo key registry is unavailable",
    verify_result: bool = True,
) -> _StageClaimSpec:
    def _sig_record(approval: dict[str, Any], value: Any) -> dict[str, Any]:
        if not isinstance(value, str) or len(value) != 64:
            raise _FakeSigError("bad signature")
        return {
            "reviewer_id": approval["reviewer_identity"]["id"],
            "key_id": "k1",
            "value": value,
        }

    return _StageClaimSpec(
        stage=stage,
        schema=f"anima.self_iteration.{stage}_claim.v1",
        stage_id_field=f"{stage}_id",
        parent_id_field=parent_id_field,
        actor_identity_field="reviewer_identity",
        signature_actor_id_field="reviewer_id",
        extra_false_fields=extra_false_fields,
        key_registry_unavailable_message=key_registry_msg,
        signature_error_type=_FakeSigError,
        approval_sha256=lambda approval: f"sha-{approval['challenge_id']}",
        signature_record=_sig_record,
        verify_signature=lambda _approval, _sig, _key: verify_result,
    )


def _sample_approval(stage: str = "demo") -> dict[str, Any]:
    return {
        "challenge_id": "ch-1",
        f"{stage}_id": "st-1",
        "proposal_id": "prop-1",
        "candidate_id": "cand-1",
        "parent_id": "parent-1",
        "reviewer_identity": {"kind": "agent", "id": "rev-1", "issuer": None},
    }


def _valid_key() -> VerifierKey:
    return VerifierKey(verifier_id="rev-1", key_id="k1", secret=b"secret")


class TestStageClaimHelpers:
    def test_build_and_validate_round_trip_returns_deep_copy(self):
        spec = _make_spec()
        approval = _sample_approval()
        sig = "a" * 64
        claim = _build_stage_claim(
            spec,
            approval=approval,
            approval_signature=sig,
            actor_identity=approval["reviewer_identity"],
            claimed_at="2026-11-19T12:00:00Z",
        )
        assert list(claim.keys()) == [
            "schema",
            "challenge_id",
            "demo_id",
            "approval_sha256",
            "approval_signature",
            "proposal_id",
            "candidate_id",
            "parent_id",
            "reviewer_identity",
            "claimed_at",
            "automatic_retry_allowed",
            "persistent_activation_allowed",
            "authority_granted",
        ]

        validated = _validate_stage_claim(
            spec,
            claim,
            approval=approval,
            verifier_key_provider=lambda _actor, _kid: _valid_key(),
        )
        assert validated == claim
        validated["reviewer_identity"]["id"] = "mutated"
        assert claim["reviewer_identity"]["id"] == "rev-1"

    def test_build_and_validate_without_parent_or_extra_flags(self):
        spec = _make_spec(parent_id_field=None, extra_false_fields=())
        approval = _sample_approval()
        claim = _build_stage_claim(
            spec,
            approval=approval,
            approval_signature="b" * 64,
            actor_identity=approval["reviewer_identity"],
            claimed_at="2026-11-19T12:00:00Z",
        )
        assert "parent_id" not in claim
        assert "persistent_activation_allowed" not in claim
        assert _validate_stage_claim(
            spec,
            claim,
            approval=approval,
            verifier_key_provider=lambda _actor, _kid: _valid_key(),
        ) == claim

    @pytest.mark.parametrize(
        "mutator, expected_msg",
        [
            (lambda c: "not-a-dict", "demo claim fields are malformed"),
            (lambda c: {k: v for k, v in c.items() if k != "parent_id"}, "demo claim fields are malformed"),
            (lambda c: {**c, "unexpected": 1}, "demo claim fields are malformed"),
            (lambda c: {**c, "claimed_at": "not-a-timestamp"}, "demo claim timestamp is malformed"),
            (lambda c: {**c, "approval_signature": "bad"}, "demo claim signature is malformed"),
            (lambda c: {**c, "approval_signature": {"value": "short"}}, "demo claim signature is malformed"),
            (lambda c: {**c, "schema": "wrong"}, "demo claim binding is malformed"),
            (lambda c: {**c, "parent_id": "wrong"}, "demo claim binding is malformed"),
            (lambda c: {**c, "automatic_retry_allowed": True}, "demo claim binding is malformed"),
            (lambda c: {**c, "automatic_retry_allowed": 0}, "demo claim binding is malformed"),
            (lambda c: {**c, "persistent_activation_allowed": True}, "demo claim binding is malformed"),
            (lambda c: {**c, "persistent_activation_allowed": 0}, "demo claim binding is malformed"),
            (lambda c: {**c, "authority_granted": True}, "demo claim binding is malformed"),
            (lambda c: {**c, "authority_granted": 0}, "demo claim binding is malformed"),
        ],
    )
    def test_validate_rejects_malformed_claim(self, mutator, expected_msg):
        spec = _make_spec()
        approval = _sample_approval()
        claim = _build_stage_claim(
            spec,
            approval=approval,
            approval_signature="c" * 64,
            actor_identity=approval["reviewer_identity"],
            claimed_at="2026-11-19T12:00:00Z",
        )
        broken = mutator(copy.deepcopy(claim))
        with pytest.raises(SelfIterationError, match=expected_msg):
            _validate_stage_claim(
                spec,
                broken,
                approval=approval,
                verifier_key_provider=lambda _actor, _kid: _valid_key(),
            )

    def test_validate_handles_key_registry_failure_and_invalid_signature(self):
        spec = _make_spec()
        approval = _sample_approval()
        claim = _build_stage_claim(
            spec,
            approval=approval,
            approval_signature="d" * 64,
            actor_identity=approval["reviewer_identity"],
            claimed_at="2026-11-19T12:00:00Z",
        )

        def _raising_provider(_actor: str, _kid: str) -> VerifierKey:
            raise RuntimeError("registry offline")

        with pytest.raises(SelfIterationError, match="demo key registry is unavailable"):
            _validate_stage_claim(
                spec, claim, approval=approval, verifier_key_provider=_raising_provider
            )

        with pytest.raises(SelfIterationError, match="demo claim approval signature is invalid"):
            _validate_stage_claim(
                spec, claim, approval=approval, verifier_key_provider=lambda _a, _k: None
            )

        rejecting_spec = _make_spec(verify_result=False)
        with pytest.raises(SelfIterationError, match="demo claim approval signature is invalid"):
            _validate_stage_claim(
                rejecting_spec,
                claim,
                approval=approval,
                verifier_key_provider=lambda _a, _k: _valid_key(),
            )


class TestStageChallengeEvent:
    def test_requires_exact_single_match(self):
        event = {
            "type": "demo_challenge_issued",
            "approval": {"challenge_id": "ch-1"},
        }
        non_dict_event = {"type": "demo_challenge_issued", "approval": "ignored"}

        assert (
            _stage_challenge_event(
                {"events": [non_dict_event, event]},
                "ch-1",
                event_type="demo_challenge_issued",
                stage="demo",
            )
            == event
        )

        with pytest.raises(
            SelfIterationError, match="challenge_id does not name one demo approval"
        ):
            _stage_challenge_event(
                {"events": [non_dict_event]},
                "ch-1",
                event_type="demo_challenge_issued",
                stage="demo",
            )

        with pytest.raises(
            SelfIterationError, match="challenge_id does not name one demo approval"
        ):
            _stage_challenge_event(
                {"events": [event, dict(event)]},
                "ch-1",
                event_type="demo_challenge_issued",
                stage="demo",
            )


def test_production_stage_claim_specs_preserve_exact_contracts():
    assert _EXECUTION_CLAIM_SPEC.stage == "execution"
    assert _EXECUTION_CLAIM_SPEC.schema == "anima.self_iteration.execution_claim.v1"
    assert _EXECUTION_CLAIM_SPEC.stage_id_field == "approval_id"
    assert _EXECUTION_CLAIM_SPEC.parent_id_field is None
    assert _EXECUTION_CLAIM_SPEC.actor_identity_field == "approver_identity"
    assert _EXECUTION_CLAIM_SPEC.signature_actor_id_field == "approver_id"
    assert _EXECUTION_CLAIM_SPEC.extra_false_fields == ()
    assert (
        _EXECUTION_CLAIM_SPEC.key_registry_unavailable_message
        == "execution approval signing key registry is unavailable"
    )
    assert _EXECUTION_CLAIM_SPEC.signature_error_type is ExecutionError

    assert _APPLICATION_CLAIM_SPEC.stage == "application"
    assert _APPLICATION_CLAIM_SPEC.schema == "anima.self_iteration.application_claim.v1"
    assert _APPLICATION_CLAIM_SPEC.stage_id_field == "application_id"
    assert _APPLICATION_CLAIM_SPEC.parent_id_field == "execution_id"
    assert _APPLICATION_CLAIM_SPEC.actor_identity_field == "reviewer_identity"
    assert _APPLICATION_CLAIM_SPEC.signature_actor_id_field == "reviewer_id"
    assert _APPLICATION_CLAIM_SPEC.extra_false_fields == ()
    assert (
        _APPLICATION_CLAIM_SPEC.key_registry_unavailable_message
        == "application reviewer key registry is unavailable"
    )
    assert _APPLICATION_CLAIM_SPEC.signature_error_type is ApplicationError

    assert _CANARY_CLAIM_SPEC.stage == "canary"
    assert _CANARY_CLAIM_SPEC.schema == "anima.self_iteration.canary_claim.v1"
    assert _CANARY_CLAIM_SPEC.stage_id_field == "canary_id"
    assert _CANARY_CLAIM_SPEC.parent_id_field == "application_result_id"
    assert _CANARY_CLAIM_SPEC.actor_identity_field == "reviewer_identity"
    assert _CANARY_CLAIM_SPEC.signature_actor_id_field == "reviewer_id"
    assert _CANARY_CLAIM_SPEC.extra_false_fields == ("persistent_activation_allowed",)
    assert (
        _CANARY_CLAIM_SPEC.key_registry_unavailable_message
        == "canary reviewer key registry is unavailable"
    )
    assert _CANARY_CLAIM_SPEC.signature_error_type is CanaryError


def test_stage_claim_spec_rejects_overlapping_or_empty_field_names():
    with pytest.raises(ValueError, match="collide with core keys"):
        _StageClaimSpec(
            stage="bad",
            schema="bad.v1",
            stage_id_field="challenge_id",
            parent_id_field=None,
            actor_identity_field="actor_identity",
            signature_actor_id_field="actor_id",
            extra_false_fields=(),
            key_registry_unavailable_message="unavailable",
            signature_error_type=ValueError,
            approval_sha256=lambda _a: "sha",
            signature_record=lambda _a, _s: {},
            verify_signature=lambda _a, _s, _k: True,
        )
    with pytest.raises(ValueError, match="must be unique"):
        _StageClaimSpec(
            stage="bad",
            schema="bad.v1",
            stage_id_field="stage_id",
            parent_id_field="stage_id",
            actor_identity_field="actor_identity",
            signature_actor_id_field="actor_id",
            extra_false_fields=(),
            key_registry_unavailable_message="unavailable",
            signature_error_type=ValueError,
            approval_sha256=lambda _a: "sha",
            signature_record=lambda _a, _s: {},
            verify_signature=lambda _a, _s, _k: True,
        )


def test_stage_mixins_have_pairwise_disjoint_method_names():
    """Ensure composing the stage mixins in SelfIterationPipelineMixin shadows nothing."""
    from anima_mcp.self_iteration_application_stage import SelfIterationApplicationStageMixin
    from anima_mcp.self_iteration_canary_stage import SelfIterationCanaryStageMixin
    from anima_mcp.self_iteration_execution_stage import SelfIterationExecutionStageMixin
    from anima_mcp.self_iteration_pipeline import SelfIterationPipelineMixin

    mixins = [
        SelfIterationPipelineMixin,
        SelfIterationExecutionStageMixin,
        SelfIterationApplicationStageMixin,
        SelfIterationCanaryStageMixin,
    ]
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for cls in mixins:
        for name in cls.__dict__:
            if name.startswith("__") and name.endswith("__"):
                continue
            if name in seen:
                collisions.append((name, seen[name], cls.__name__))
            else:
                seen[name] = cls.__name__
    assert collisions == [], f"Unexpected mixin method collisions: {collisions}"


def test_every_self_iteration_module_is_covered_by_protected_rules():
    """Any new src/anima_mcp/self_iteration*.py split must appear in _PROTECTED_RULES."""
    from pathlib import Path
    from anima_mcp.self_iteration import _PROTECTED_RULES

    src_dir = Path(__file__).resolve().parents[1] / "src" / "anima_mcp"
    protected_paths = {path for path, _reason in _PROTECTED_RULES}
    actual_modules = sorted(
        f"src/anima_mcp/{p.name}" for p in src_dir.glob("self_iteration*.py")
    )
    assert actual_modules
    missing = [p for p in actual_modules if p not in protected_paths]
    assert missing == [], f"Unprotected self_iteration modules: {missing}"


def test_validate_stage_claim_rejects_malformed_signature_types():
    spec = _make_spec()
    approval = _sample_approval()
    base_claim = _build_stage_claim(
        spec,
        approval=approval,
        approval_signature="a" * 64,
        actor_identity=approval["reviewer_identity"],
        claimed_at="2026-11-19T00:00:00Z",
    )
    for bad_sig in (None, "not-a-dict", [], {}, {"value": 123}):
        mutated = dict(base_claim)
        mutated["approval_signature"] = bad_sig
        with pytest.raises(SelfIterationError, match="claim signature is malformed"):
            _validate_stage_claim(
                spec,
                mutated,
                approval=approval,
                verifier_key_provider=lambda _actor, _kid: None,
            )
