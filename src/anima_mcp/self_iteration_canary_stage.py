"""Transient canary supervision stage mixin for self-iteration proposals.

Contains the canary stage controller methods (`prepare_canary`, `run_canary`,
`canary_status`) and their canary-specific claim, challenge, and result-event
helpers.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from datetime import timedelta
from typing import Any

from .self_iteration_canary import (
    CANARY_SIGNER_ID_ENV,
    CanaryError,
    build_canary_approval,
    build_canary_request,
    canary_approval_sha256,
    canary_contract,
    canary_signature_record,
    canary_signing_input_b64,
    validate_canary_approval,
    validate_signed_canary_result,
    validate_supervisor_identity,
    verify_canary_approval_signature,
)
from .self_iteration_ledger import SelfIterationError, _isoformat
from .self_iteration_sandbox import PatchSandboxError
from .self_iteration_stage_helpers import (
    _build_stage_claim,
    _required_text,
    _stage_challenge_event,
    _StageClaimSpec,
    _validate_stage_claim,
)
from .self_iteration_verification import (
    SIGNATURE_ALGORITHM,
    canonical_json_bytes,
    parse_utc_timestamp,
)

_CANARY_CLAIM_SPEC = _StageClaimSpec(
    stage="canary",
    schema="anima.self_iteration.canary_claim.v1",
    stage_id_field="canary_id",
    parent_id_field="application_result_id",
    actor_identity_field="reviewer_identity",
    signature_actor_id_field="reviewer_id",
    extra_false_fields=("persistent_activation_allowed",),
    key_registry_unavailable_message="canary reviewer key registry is unavailable",
    signature_error_type=CanaryError,
    approval_sha256=canary_approval_sha256,
    signature_record=canary_signature_record,
    verify_signature=verify_canary_approval_signature,
)


class SelfIterationCanaryStageMixin:
    """Transient canary supervision stage methods for SelfIterationPipelineMixin."""

    @staticmethod
    def _canary_claim_record(
        *,
        approval: dict[str, Any],
        approval_signature: str,
        reviewer_identity: dict[str, Any],
        claimed_at: str,
    ) -> dict[str, Any]:
        return _build_stage_claim(
            _CANARY_CLAIM_SPEC,
            approval=approval,
            approval_signature=approval_signature,
            actor_identity=reviewer_identity,
            claimed_at=claimed_at,
        )

    def _validate_canary_claim(
        self, claim: Any, *, approval: dict[str, Any]
    ) -> dict[str, Any]:
        return _validate_stage_claim(
            _CANARY_CLAIM_SPEC,
            claim,
            approval=approval,
            verifier_key_provider=self._verifier_key_provider,
        )

    @staticmethod
    def _canary_challenge_event(
        proposal: dict[str, Any], challenge_id: str
    ) -> dict[str, Any]:
        return _stage_challenge_event(
            proposal,
            challenge_id,
            event_type="canary_review_challenge_issued",
            stage="canary",
        )

    @staticmethod
    def _canary_result_event(result: dict[str, Any]) -> dict[str, Any]:
        approval = result["approval"]
        return {
            "type": "transient_canary_recorded",
            "at": result["finished_at"],
            "challenge_id": result["challenge_id"],
            "canary_id": result["canary_id"],
            "approval_sha256": canary_approval_sha256(approval),
            "approval_signature_sha256": hashlib.sha256(
                canonical_json_bytes(result["approval_signature"])
            ).hexdigest(),
            "canary_result_id": result["canary_result_id"],
            "result_sha256": result["result_sha256"],
            "candidate_id": result["candidate_id"],
            "application_result_id": result["application_result_id"],
            "application_result_sha256": result["application_result_sha256"],
            "reviewer_identity": approval["reviewer_identity"],
            "supervisor_signer_id": result["signature"]["signer_id"],
            "outcome": result["outcome"],
            "activation_performed": result["activation_performed"],
            "health_checks_sha256": hashlib.sha256(
                canonical_json_bytes(result["health_checks"])
            ).hexdigest(),
            "baseline_restore_attempted": result["baseline_restore_attempted"],
            "baseline_restored": result["baseline_restored"],
            "persistent_activation_retained": False,
            "recommended_decision": result["recommended_decision"],
            "eligible_for_merge_review": result["eligible_for_merge_review"],
            "eligible_for_live_activation": False,
            "authority_granted": False,
        }

    def prepare_canary(
        self,
        *,
        proposal_id: Any,
        candidate_id: Any,
        application_result_id: Any,
        expected_application_result_sha256: Any,
    ) -> dict[str, Any]:
        """Issue a signed plan for one externally supervised transient canary."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        candidate_id_text = _required_text(candidate_id, "candidate_id", max_length=100)
        application_result_id_text = _required_text(
            application_result_id, "application_result_id", max_length=100
        )
        application_digest = _required_text(
            expected_application_result_sha256,
            "expected_application_result_sha256",
            max_length=64,
        ).lower()
        if not re.fullmatch(r"siar-[0-9a-f]{32}", application_result_id_text):
            raise SelfIterationError("application_result_id is malformed")
        if not re.fullmatch(r"[0-9a-f]{64}", application_digest):
            raise SelfIterationError("expected_application_result_sha256 is malformed")

        with self._lock:
            try:
                ledger = self._load_ledger()
                proposal = self._find_proposal(ledger, proposal_id_text)
                now = self._now_utc()
                state = self._patch_eligible_state(
                    proposal,
                    expected_content_sha256=proposal["content_sha256"],
                    now=now,
                )
                source_fingerprint = self._assert_current_proposal_source(proposal)
                candidate = self._patch_sandbox.load_manifest(candidate_id_text)
                self._assert_candidate_binding(
                    proposal=proposal, candidate=candidate, state=state
                )
                application_result = self._passing_application_result(
                    proposal=proposal,
                    candidate=candidate,
                    application_result_id=application_result_id_text,
                    expected_result_sha256=application_digest,
                )
                application_approval = application_result["approval"]
                if application_approval[
                    "source_fingerprint"
                ] != source_fingerprint or application_approval[
                    "active_attestation_ids"
                ] != sorted(state["active_attestation_ids"]):
                    raise SelfIterationError(
                        "proposal source or verification changed after application"
                    )

                recorded_at = _isoformat(now)
                receipt = self._server_provenance(recorded_at)
                reviewer_identity = self._identity_from_provenance(
                    receipt,
                    label="canary review request",
                    required=True,
                )
                assert reviewer_identity is not None
                verifier_identities = self._active_verifier_identities(proposal, state)
                execution_events = [
                    event
                    for event in proposal["events"]
                    if event.get("type") == "isolated_execution_recorded"
                    and event.get("execution_id") == application_result["execution_id"]
                ]
                if len(execution_events) != 1:
                    raise SelfIterationError(
                        "canary review requires one recorded execution chain"
                    )
                execution_event = execution_events[0]
                prior_ids = {
                    identity.get("id")
                    for identity in [
                        proposal.get("proposer_identity"),
                        execution_event.get("approver_identity"),
                        application_approval["reviewer_identity"],
                        *verifier_identities,
                    ]
                    if isinstance(identity, dict)
                }
                prior_ids.update(
                    {
                        execution_event.get("result_signer_id"),
                        application_result["signature"]["signer_id"],
                    }
                )
                reviewer_id = reviewer_identity["id"]
                if reviewer_id in prior_ids:
                    raise SelfIterationError(
                        "canary reviewer must differ from all prior participants"
                    )
                reviewer_key = self._resolve_verifier_key(reviewer_identity)

                configured_supervisor_id = os.environ.get(CANARY_SIGNER_ID_ENV)
                if (
                    not isinstance(configured_supervisor_id, str)
                    or not configured_supervisor_id
                ):
                    raise SelfIterationError(
                        f"{CANARY_SIGNER_ID_ENV} must name the dedicated supervisor signer"
                    )
                if configured_supervisor_id in {reviewer_id, *prior_ids}:
                    raise SelfIterationError(
                        "canary supervisor signer must differ from all participants"
                    )
                supervisor_identity = validate_supervisor_identity(
                    self._canary_supervisor.probe()
                )
                result_signer = supervisor_identity.get("result_signer")
                if (
                    supervisor_identity.get("supervisor_id") != configured_supervisor_id
                    or not isinstance(result_signer, dict)
                    or result_signer.get("id") != configured_supervisor_id
                ):
                    raise SelfIterationError(
                        "canary supervisor identity does not match configuration"
                    )
                supervisor_key = self._resolve_verifier_key(
                    {
                        "kind": "service_signer",
                        "id": configured_supervisor_id,
                        "issuer": None,
                    },
                    result_signer.get("key_id"),
                )
                approval = build_canary_approval(
                    application_result=application_result,
                    reviewer_identity=reviewer_identity,
                    reviewer_key_id=reviewer_key.key_id,
                    supervisor_identity=supervisor_identity,
                    issued_at=now,
                )
                if (
                    approval["supervisor_identity"]["result_signer"]["key_id"]
                    != supervisor_key.key_id
                ):
                    raise SelfIterationError(
                        "canary supervisor signing key binding is malformed"
                    )
                event = {
                    "type": "canary_review_challenge_issued",
                    "at": recorded_at,
                    "approval": copy.deepcopy(approval),
                    "approval_sha256": canary_approval_sha256(approval),
                    "requester_identity": reviewer_identity,
                    "provenance": receipt,
                    "activation_performed": False,
                    "baseline_restore_attempted": False,
                    "persistent_activation_retained": False,
                    "authority_granted": False,
                }
                proposal["events"].append(event)
                self._save_ledger(ledger)
            except (CanaryError, PatchSandboxError) as exc:
                raise SelfIterationError(str(exc)) from exc

        return {
            "approval": approval,
            "approval_sha256": canary_approval_sha256(approval),
            "signing_input_b64": canary_signing_input_b64(approval),
            "signature": {
                "algorithm": SIGNATURE_ALGORITHM,
                "key_id": reviewer_key.key_id,
                "assurance": "symmetric_mac_server_verifiable",
            },
            "next_step": (
                "The authenticated canary reviewer signs these exact bytes, then "
                "submits run_canary before expiry. The claim is consumed before the "
                "external supervisor may activate the transient candidate."
            ),
            "authority_granted": False,
        }

    def run_canary(
        self,
        *,
        proposal_id: Any,
        challenge_id: Any,
        signature: Any,
    ) -> dict[str, Any]:
        """Consume one review and request its fixed transient canary exactly once."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        challenge_id_text = _required_text(challenge_id, "challenge_id", max_length=100)
        signature_text = _required_text(signature, "signature", max_length=64).lower()
        if not re.fullmatch(r"sicc-[0-9a-f]{32}", challenge_id_text):
            raise SelfIterationError("canary challenge_id is malformed")
        if not re.fullmatch(r"[0-9a-f]{64}", signature_text):
            raise SelfIterationError("canary approval signature is malformed")

        with self._lock:
            try:
                ledger = self._load_ledger()
                proposal = self._find_proposal(ledger, proposal_id_text)
                challenge_event = self._canary_challenge_event(
                    proposal, challenge_id_text
                )
                approval = validate_canary_approval(challenge_event["approval"])
                now = self._now_utc()
                issued = parse_utc_timestamp(approval["issued_at"], "issued_at")
                expires = parse_utc_timestamp(
                    approval["challenge_expires_at"], "challenge_expires_at"
                )
                if now < issued - timedelta(seconds=30) or now > expires:
                    raise SelfIterationError(
                        "canary approval is expired or not yet valid"
                    )
                receipt = self._server_provenance(_isoformat(now))
                reviewer_identity = self._identity_from_provenance(
                    receipt,
                    label="transient canary request",
                    required=True,
                )
                assert reviewer_identity is not None
                if reviewer_identity != approval["reviewer_identity"]:
                    raise SelfIterationError(
                        "only the authenticated canary reviewer may consume this challenge"
                    )
                reviewer_key = self._resolve_verifier_key(
                    reviewer_identity, approval["reviewer_key_id"]
                )
                if not verify_canary_approval_signature(
                    approval, signature_text, reviewer_key
                ):
                    raise SelfIterationError("canary approval signature is invalid")

                state = self._patch_eligible_state(
                    proposal,
                    expected_content_sha256=approval["proposal_content_sha256"],
                    now=now,
                )
                self._assert_current_proposal_source(proposal)
                candidate = self._patch_sandbox.load_manifest(approval["candidate_id"])
                if candidate.get("candidate_sha256") != approval["candidate_sha256"]:
                    raise SelfIterationError("canary candidate digest is stale")
                self._assert_candidate_binding(
                    proposal=proposal, candidate=candidate, state=state
                )
                application_result = self._passing_application_result(
                    proposal=proposal,
                    candidate=candidate,
                    application_result_id=approval["application_result_id"],
                    expected_result_sha256=approval["application_result_sha256"],
                )
                if (
                    application_result["application_id"] != approval["application_id"]
                    or application_result["application_approval_sha256"]
                    != approval["application_approval_sha256"]
                    or application_result["target_ref"] != approval["target_ref"]
                    or application_result["parent_revision"]
                    != approval["baseline_revision"]
                    or application_result["commit_oid"]
                    != approval["candidate_commit_oid"]
                    or application_result["tree_oid"] != approval["candidate_tree_oid"]
                ):
                    raise SelfIterationError(
                        "reviewed application changed before canary execution"
                    )
                if (
                    validate_supervisor_identity(self._canary_supervisor.probe())
                    != approval["supervisor_identity"]
                ):
                    raise SelfIterationError(
                        "canary supervisor identity changed after review"
                    )
                supervisor_signer = approval["supervisor_identity"]["result_signer"]
                self._resolve_verifier_key(
                    {
                        "kind": "service_signer",
                        "id": supervisor_signer["id"],
                        "issuer": None,
                    },
                    supervisor_signer["key_id"],
                )

                requested_at = _isoformat(now)
                claim = self._canary_claim_record(
                    approval=approval,
                    approval_signature=signature_text,
                    reviewer_identity=reviewer_identity,
                    claimed_at=requested_at,
                )
                self._patch_sandbox.claim_canary(
                    approval["candidate_id"], challenge_id_text, claim
                )

                result = self._canary_supervisor.evaluate(
                    approval=approval,
                    approval_signature=signature_text,
                    requested_at=requested_at,
                )
                result = validate_signed_canary_result(
                    result, self._verifier_key_provider
                )
                expected_request = build_canary_request(
                    approval=approval,
                    approval_signature=signature_text,
                    requested_at=requested_at,
                )
                if (
                    result["approval"] != approval
                    or result["approval_signature"] != claim["approval_signature"]
                    or result["request_sha256"] != expected_request["request_sha256"]
                    or result["supervisor_identity"] != approval["supervisor_identity"]
                ):
                    raise SelfIterationError(
                        "signed canary result does not match its one-use request"
                    )
                self._patch_sandbox.store_canary_result(
                    approval["candidate_id"], result
                )

                fresh_ledger = self._load_ledger()
                fresh_proposal = self._find_proposal(fresh_ledger, proposal_id_text)
                self._canary_challenge_event(fresh_proposal, challenge_id_text)
                if any(
                    event.get("type") == "transient_canary_recorded"
                    and event.get("challenge_id") == challenge_id_text
                    for event in fresh_proposal["events"]
                ):
                    raise SelfIterationError(
                        "canary challenge already has a recorded result"
                    )
                fresh_proposal["events"].append(self._canary_result_event(result))
                self._save_ledger(fresh_ledger)
            except (CanaryError, PatchSandboxError) as exc:
                raise SelfIterationError(str(exc)) from exc

        return {
            "result": result,
            "canary_contract": canary_contract(),
            "next_step": (
                "The candidate is no longer live. A passing restored canary may be "
                "reviewed for merge; no result grants persistent activation authority."
            ),
            "authority_granted": False,
        }

    def canary_status(
        self,
        *,
        proposal_id: Any,
        candidate_id: Any,
    ) -> dict[str, Any]:
        """Reconcile canary reviews, claims, signed results, and ledger events."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        candidate_id_text = _required_text(candidate_id, "candidate_id", max_length=100)
        with self._lock:
            ledger = self._load_ledger()
            proposal = self._find_proposal(ledger, proposal_id_text)
            try:
                candidate = self._patch_sandbox.load_manifest(candidate_id_text)
                self._assert_candidate_binding(proposal=proposal, candidate=candidate)
                raw_claims = self._patch_sandbox.load_canary_claims(candidate_id_text)
                raw_results = self._patch_sandbox.load_canary_results(candidate_id_text)
            except PatchSandboxError as exc:
                raise SelfIterationError(str(exc)) from exc
            challenge_events = [
                event
                for event in proposal["events"]
                if event.get("type") == "canary_review_challenge_issued"
                and event.get("approval", {}).get("candidate_id") == candidate_id_text
            ]
            approvals = {
                event["approval"]["challenge_id"]: event["approval"]
                for event in challenge_events
            }
            claims: dict[str, dict[str, Any]] = {}
            for raw_claim in raw_claims:
                challenge_id_value = raw_claim.get("challenge_id")
                if not isinstance(challenge_id_value, str):
                    raise SelfIterationError(
                        "canary claim challenge identifier is malformed"
                    )
                approval = approvals.get(challenge_id_value)
                if approval is None or challenge_id_value in claims:
                    raise SelfIterationError(
                        "canary claim has no unique ledger-recorded approval"
                    )
                claims[challenge_id_value] = self._validate_canary_claim(
                    raw_claim, approval=approval
                )
            results: dict[str, dict[str, Any]] = {}
            for raw_result in raw_results:
                try:
                    result = validate_signed_canary_result(
                        raw_result, self._verifier_key_provider
                    )
                except CanaryError as exc:
                    raise SelfIterationError(
                        "signed canary result artifact is invalid"
                    ) from exc
                challenge_id_value = result["challenge_id"]
                approval = approvals.get(challenge_id_value)
                claim = claims.get(challenge_id_value)
                if approval is None or claim is None or challenge_id_value in results:
                    raise SelfIterationError(
                        "signed canary result has no unique one-use approval"
                    )
                expected_request = build_canary_request(
                    approval=approval,
                    approval_signature=claim["approval_signature"]["value"],
                    requested_at=claim["claimed_at"],
                )
                if (
                    result["approval"] != approval
                    or result["approval_signature"] != claim["approval_signature"]
                    or result["request_sha256"] != expected_request["request_sha256"]
                ):
                    raise SelfIterationError(
                        "signed canary result does not match its one-use approval"
                    )
                results[challenge_id_value] = result

            recorded_events = [
                event
                for event in proposal["events"]
                if event.get("type") == "transient_canary_recorded"
                and event.get("candidate_id") == candidate_id_text
            ]
            recorded_by_challenge = {
                event["challenge_id"]: event for event in recorded_events
            }
            now = self._now_utc()
            statuses: list[dict[str, Any]] = []
            for challenge_id_value, approval in approvals.items():
                selected_claim = claims.get(challenge_id_value)
                selected_result = results.get(challenge_id_value)
                recorded = recorded_by_challenge.get(challenge_id_value)
                if recorded is not None and (
                    selected_result is None
                    or recorded != self._canary_result_event(selected_result)
                ):
                    raise SelfIterationError(
                        "ledger canary event does not match its signed result artifact"
                    )
                application_ref_intact = False
                try:
                    application_result = self._passing_application_result(
                        proposal=proposal,
                        candidate=candidate,
                        application_result_id=approval["application_result_id"],
                        expected_result_sha256=approval["application_result_sha256"],
                    )
                except SelfIterationError:
                    if selected_result is not None:
                        raise
                else:
                    application_ref_intact = bool(
                        application_result["commit_oid"]
                        == approval["candidate_commit_oid"]
                    )
                if (
                    selected_result is not None
                    and selected_result["baseline_restored"] is not True
                ):
                    state = "recorded_recovery_required"
                elif selected_result is not None and recorded is not None:
                    state = "recorded"
                elif selected_result is not None:
                    state = "signed_result_unledgered"
                elif selected_claim is not None:
                    state = "claimed_result_indeterminate"
                elif now > parse_utc_timestamp(
                    approval["challenge_expires_at"], "challenge_expires_at"
                ):
                    state = "expired_unclaimed"
                else:
                    state = "awaiting_signature"
                statuses.append(
                    {
                        "challenge_id": challenge_id_value,
                        "canary_id": approval["canary_id"],
                        "approval_sha256": canary_approval_sha256(approval),
                        "issued_at": approval["issued_at"],
                        "challenge_expires_at": approval["challenge_expires_at"],
                        "reviewer_identity": approval["reviewer_identity"],
                        "supervisor_identity": approval["supervisor_identity"],
                        "state": state,
                        "claim": copy.deepcopy(selected_claim),
                        "result": copy.deepcopy(selected_result),
                        "ledger_recorded": recorded is not None,
                        "application_ref_intact": application_ref_intact,
                        "automatic_retry_allowed": False,
                        "persistent_activation_retained": False,
                        "eligible_for_merge_review": bool(
                            selected_result is not None
                            and recorded is not None
                            and selected_result["eligible_for_merge_review"] is True
                            and application_ref_intact
                        ),
                        "eligible_for_live_activation": False,
                        "authority_granted": False,
                    }
                )
            return {
                "proposal_id": proposal_id_text,
                "candidate_id": candidate_id_text,
                "candidate_sha256": candidate["candidate_sha256"],
                "canaries": statuses,
                "canary_contract": canary_contract(),
                "persistent_activation_retained": False,
                "eligible_for_live_activation": False,
                "authority_granted": False,
            }


__all__ = ["SelfIterationCanaryStageMixin"]
