"""Reviewed Git branch application stage mixin for self-iteration proposals.

Contains the application stage controller methods (`prepare_application`,
`apply_candidate`, `application_status`) and their application-specific claim,
challenge, result-event, and passing-result helpers.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from datetime import timedelta
from typing import Any

from .self_iteration_application import (
    APPLICATION_SIGNER_ID_ENV,
    ApplicationError,
    application_approval_sha256,
    application_contract,
    application_signature_record,
    application_signing_input_b64,
    build_application_approval,
    build_signed_application_result,
    target_ref_for_candidate,
    validate_application_approval,
    validate_signed_application_result,
    verify_application_approval_signature,
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

_APPLICATION_CLAIM_SPEC = _StageClaimSpec(
    stage="application",
    schema="anima.self_iteration.application_claim.v1",
    stage_id_field="application_id",
    parent_id_field="execution_id",
    actor_identity_field="reviewer_identity",
    signature_actor_id_field="reviewer_id",
    extra_false_fields=(),
    key_registry_unavailable_message="application reviewer key registry is unavailable",
    signature_error_type=ApplicationError,
    approval_sha256=application_approval_sha256,
    signature_record=application_signature_record,
    verify_signature=verify_application_approval_signature,
)


class SelfIterationApplicationStageMixin:
    """Reviewed Git branch application stage methods for SelfIterationPipelineMixin."""

    @staticmethod
    def _application_claim_record(
        *,
        approval: dict[str, Any],
        approval_signature: str,
        reviewer_identity: dict[str, Any],
        claimed_at: str,
    ) -> dict[str, Any]:
        return _build_stage_claim(
            _APPLICATION_CLAIM_SPEC,
            approval=approval,
            approval_signature=approval_signature,
            actor_identity=reviewer_identity,
            claimed_at=claimed_at,
        )

    def _validate_application_claim(
        self, claim: Any, *, approval: dict[str, Any]
    ) -> dict[str, Any]:
        return _validate_stage_claim(
            _APPLICATION_CLAIM_SPEC,
            claim,
            approval=approval,
            verifier_key_provider=self._verifier_key_provider,
        )

    @staticmethod
    def _application_challenge_event(
        proposal: dict[str, Any], challenge_id: str
    ) -> dict[str, Any]:
        return _stage_challenge_event(
            proposal,
            challenge_id,
            event_type="application_review_challenge_issued",
            stage="application",
        )

    @staticmethod
    def _application_result_event(result: dict[str, Any]) -> dict[str, Any]:
        approval = result["approval"]
        return {
            "type": "reviewed_application_recorded",
            "at": result["applied_at"],
            "challenge_id": result["challenge_id"],
            "application_id": result["application_id"],
            "approval_sha256": result["application_approval_sha256"],
            "approval_signature_sha256": hashlib.sha256(
                canonical_json_bytes(result["approval_signature"])
            ).hexdigest(),
            "application_result_id": result["application_result_id"],
            "result_sha256": result["result_sha256"],
            "candidate_id": result["candidate_id"],
            "candidate_sha256": result["candidate_sha256"],
            "execution_id": result["execution_id"],
            "execution_result_sha256": result["execution_result_sha256"],
            "reviewer_identity": approval["reviewer_identity"],
            "result_signer_id": result["signature"]["signer_id"],
            "target_ref": result["target_ref"],
            "commit_oid": result["commit_oid"],
            "tree_oid": result["tree_oid"],
            "branch_created": True,
            "live_source_writes": False,
            "pushed": False,
            "merged": False,
            "deployed": False,
            "eligible_for_canary_review": True,
            "eligible_for_live_activation": False,
            "authority_granted": False,
        }

    def _passing_application_result(
        self,
        *,
        proposal: dict[str, Any],
        candidate: dict[str, Any],
        application_result_id: str,
        expected_result_sha256: str,
    ) -> dict[str, Any]:
        try:
            raw_results = self._patch_sandbox.load_application_results(
                candidate["candidate_id"]
            )
        except PatchSandboxError as exc:
            raise SelfIterationError(str(exc)) from exc
        matches: list[dict[str, Any]] = []
        for raw_result in raw_results:
            try:
                result = validate_signed_application_result(
                    raw_result, self._verifier_key_provider
                )
            except ApplicationError as exc:
                raise SelfIterationError(
                    "signed application result artifact is invalid"
                ) from exc
            if result.get("application_result_id") == application_result_id:
                matches.append(result)
        if len(matches) != 1:
            raise SelfIterationError(
                "canary review requires one exact signed application result"
            )
        result = matches[0]
        if (
            result.get("result_sha256") != expected_result_sha256
            or result.get("candidate_id") != candidate.get("candidate_id")
            or result.get("candidate_sha256") != candidate.get("candidate_sha256")
            or result.get("eligible_for_canary_review") is not True
            or result.get("eligible_for_live_activation") is not False
            or result.get("branch_created") is not True
            or result.get("pushed") is not False
            or result.get("merged") is not False
            or result.get("deployed") is not False
        ):
            raise SelfIterationError(
                "canary review requires the exact eligible application digest"
            )
        recorded = [
            event
            for event in proposal["events"]
            if event.get("type") == "reviewed_application_recorded"
            and event.get("application_result_id") == application_result_id
        ]
        if len(recorded) != 1 or recorded[0] != self._application_result_event(result):
            raise SelfIterationError(
                "signed application result does not match its ledger event"
            )
        challenge = self._application_challenge_event(proposal, result["challenge_id"])
        if (
            challenge.get("approval") != result["approval"]
            or challenge.get("approval_sha256") != result["application_approval_sha256"]
        ):
            raise SelfIterationError(
                "signed application result does not match its ledger approval"
            )
        if self.repo_root is None or not self._application_writer.verify_result(
            repo_root=self.repo_root, result=result
        ):
            raise SelfIterationError(
                "reviewed application Git ref failed integrity verification"
            )
        return result

    def prepare_application(
        self,
        *,
        proposal_id: Any,
        candidate_id: Any,
        execution_id: Any,
        expected_execution_result_sha256: Any,
    ) -> dict[str, Any]:
        """Issue a short-lived review plan for one dedicated Git branch."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        candidate_id_text = _required_text(candidate_id, "candidate_id", max_length=100)
        execution_id_text = _required_text(execution_id, "execution_id", max_length=100)
        execution_digest = _required_text(
            expected_execution_result_sha256,
            "expected_execution_result_sha256",
            max_length=64,
        ).lower()
        if not re.fullmatch(r"six-[0-9a-f]{32}", execution_id_text):
            raise SelfIterationError("execution_id is malformed")
        if not re.fullmatch(r"[0-9a-f]{64}", execution_digest):
            raise SelfIterationError("expected_execution_result_sha256 is malformed")

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
                candidate, contents = self._patch_sandbox.execution_material(
                    candidate_id_text
                )
                self._assert_candidate_binding(
                    proposal=proposal, candidate=candidate, state=state
                )
                execution_result = self._passing_execution_result(
                    proposal=proposal,
                    candidate=candidate,
                    execution_id=execution_id_text,
                    expected_result_sha256=execution_digest,
                )
                execution_approval = execution_result["approval"]
                if execution_approval[
                    "source_fingerprint"
                ] != source_fingerprint or execution_approval[
                    "active_attestation_ids"
                ] != sorted(state["active_attestation_ids"]):
                    raise SelfIterationError(
                        "proposal source or verification changed after execution"
                    )

                recorded_at = _isoformat(now)
                receipt = self._server_provenance(recorded_at)
                reviewer_identity = self._identity_from_provenance(
                    receipt,
                    label="application review request",
                    required=True,
                )
                assert reviewer_identity is not None
                verifier_identities = self._active_verifier_identities(proposal, state)
                prior_ids = {
                    identity.get("id")
                    for identity in [
                        proposal.get("proposer_identity"),
                        execution_approval["approver_identity"],
                        *verifier_identities,
                    ]
                    if isinstance(identity, dict)
                }
                prior_ids.add(execution_result["signature"]["signer_id"])
                reviewer_id = reviewer_identity["id"]
                if reviewer_id in prior_ids:
                    raise SelfIterationError(
                        "application reviewer must differ from all prior participants"
                    )
                reviewer_key = self._resolve_verifier_key(reviewer_identity)

                result_signer_id = os.environ.get(APPLICATION_SIGNER_ID_ENV)
                if not isinstance(result_signer_id, str) or not result_signer_id:
                    raise SelfIterationError(
                        f"{APPLICATION_SIGNER_ID_ENV} must name a dedicated result signer"
                    )
                if result_signer_id in {reviewer_id, *prior_ids}:
                    raise SelfIterationError(
                        "application result signer must differ from all review participants"
                    )
                result_signer_identity = {
                    "kind": "service_signer",
                    "id": result_signer_id,
                    "issuer": None,
                }
                result_key = self._resolve_verifier_key(result_signer_identity)
                if self.repo_root is None:
                    raise SelfIterationError("source repository not found")
                snapshot = self._workspace_builder.fingerprint(
                    repo_root=self.repo_root,
                    expected_revision=execution_result["source_snapshot"]["revision"],
                    candidate_manifest=candidate,
                    candidate_contents=contents,
                )
                if snapshot != execution_result["source_snapshot"]:
                    raise SelfIterationError(
                        "source snapshot changed after isolated execution"
                    )
                git_identity = self._application_writer.probe(
                    repo_root=self.repo_root,
                    expected_parent_revision=snapshot["revision"],
                    target_ref=target_ref_for_candidate(candidate_id_text),
                )
                approval = build_application_approval(
                    proposal_id=proposal_id_text,
                    proposal_content_sha256=proposal["content_sha256"],
                    source_fingerprint=source_fingerprint,
                    active_attestation_ids=state["active_attestation_ids"],
                    candidate_id=candidate_id_text,
                    candidate_sha256=candidate["candidate_sha256"],
                    execution_id=execution_id_text,
                    execution_result_sha256=execution_digest,
                    execution_approval_sha256=execution_result["approval_sha256"],
                    execution_finished_at=execution_result["finished_at"],
                    source_snapshot=snapshot,
                    reviewer_identity=reviewer_identity,
                    reviewer_key_id=reviewer_key.key_id,
                    git_identity=git_identity,
                    result_signer_id=result_signer_id,
                    result_signer_key_id=result_key.key_id,
                    issued_at=now,
                )
                event = {
                    "type": "application_review_challenge_issued",
                    "at": recorded_at,
                    "approval": copy.deepcopy(approval),
                    "approval_sha256": application_approval_sha256(approval),
                    "requester_identity": reviewer_identity,
                    "provenance": receipt,
                    "branch_created": False,
                    "live_source_writes": False,
                    "pushed": False,
                    "merged": False,
                    "deployed": False,
                    "authority_granted": False,
                }
                proposal["events"].append(event)
                self._save_ledger(ledger)
            except (ApplicationError, PatchSandboxError) as exc:
                raise SelfIterationError(str(exc)) from exc

        return {
            "approval": approval,
            "approval_sha256": application_approval_sha256(approval),
            "signing_input_b64": application_signing_input_b64(approval),
            "signature": {
                "algorithm": SIGNATURE_ALGORITHM,
                "key_id": reviewer_key.key_id,
                "assurance": "symmetric_mac_server_verifiable",
            },
            "next_step": (
                "The authenticated application reviewer signs these exact bytes, then "
                "submits apply_candidate before expiry. The one-use claim is consumed "
                "before Git creates the dedicated branch."
            ),
            "authority_granted": False,
        }

    def apply_candidate(
        self,
        *,
        proposal_id: Any,
        challenge_id: Any,
        signature: Any,
    ) -> dict[str, Any]:
        """Consume one review and create its exact dedicated Git branch once."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        challenge_id_text = _required_text(challenge_id, "challenge_id", max_length=100)
        signature_text = _required_text(signature, "signature", max_length=64).lower()
        if not re.fullmatch(r"siac-[0-9a-f]{32}", challenge_id_text):
            raise SelfIterationError("application challenge_id is malformed")
        if not re.fullmatch(r"[0-9a-f]{64}", signature_text):
            raise SelfIterationError("application approval signature is malformed")

        with self._lock:
            try:
                ledger = self._load_ledger()
                proposal = self._find_proposal(ledger, proposal_id_text)
                challenge_event = self._application_challenge_event(
                    proposal, challenge_id_text
                )
                approval = validate_application_approval(challenge_event["approval"])
                now = self._now_utc()
                issued = parse_utc_timestamp(approval["issued_at"], "issued_at")
                expires = parse_utc_timestamp(
                    approval["challenge_expires_at"], "challenge_expires_at"
                )
                if now < issued - timedelta(seconds=30) or now > expires:
                    raise SelfIterationError(
                        "application approval is expired or not yet valid"
                    )
                receipt = self._server_provenance(_isoformat(now))
                reviewer_identity = self._identity_from_provenance(
                    receipt,
                    label="candidate application request",
                    required=True,
                )
                assert reviewer_identity is not None
                if reviewer_identity != approval["reviewer_identity"]:
                    raise SelfIterationError(
                        "only the authenticated application reviewer may consume this challenge"
                    )
                approval_key = self._resolve_verifier_key(
                    reviewer_identity, approval["reviewer_key_id"]
                )
                if not verify_application_approval_signature(
                    approval, signature_text, approval_key
                ):
                    raise SelfIterationError(
                        "application approval signature is invalid"
                    )

                state = self._patch_eligible_state(
                    proposal,
                    expected_content_sha256=approval["proposal_content_sha256"],
                    now=now,
                )
                source_fingerprint = self._assert_current_proposal_source(proposal)
                if (
                    source_fingerprint != approval["source_fingerprint"]
                    or sorted(state["active_attestation_ids"])
                    != approval["active_attestation_ids"]
                ):
                    raise SelfIterationError(
                        "proposal source or verification changed after application review"
                    )
                candidate, contents = self._patch_sandbox.execution_material(
                    approval["candidate_id"]
                )
                if candidate.get("candidate_sha256") != approval["candidate_sha256"]:
                    raise SelfIterationError("reviewed candidate digest is stale")
                self._assert_candidate_binding(
                    proposal=proposal, candidate=candidate, state=state
                )
                execution_result = self._passing_execution_result(
                    proposal=proposal,
                    candidate=candidate,
                    execution_id=approval["execution_id"],
                    expected_result_sha256=approval["execution_result_sha256"],
                )
                if (
                    execution_result["approval_sha256"]
                    != approval["execution_approval_sha256"]
                    or execution_result["finished_at"]
                    != approval["execution_finished_at"]
                    or execution_result["source_snapshot"]
                    != approval["source_snapshot"]
                ):
                    raise SelfIterationError(
                        "reviewed execution binding changed before application"
                    )
                result_signer = approval["result_signer"]
                result_key = self._resolve_verifier_key(
                    {
                        "kind": "service_signer",
                        "id": result_signer["id"],
                        "issuer": None,
                    },
                    result_signer["key_id"],
                )
                if self.repo_root is None:
                    raise SelfIterationError("source repository not found")
                current_snapshot = self._workspace_builder.fingerprint(
                    repo_root=self.repo_root,
                    expected_revision=approval["source_snapshot"]["revision"],
                    candidate_manifest=candidate,
                    candidate_contents=contents,
                )
                if current_snapshot != approval["source_snapshot"]:
                    raise SelfIterationError(
                        "source snapshot changed after application review"
                    )
                if (
                    self._application_writer.probe(
                        repo_root=self.repo_root,
                        expected_parent_revision=approval["expected_parent_revision"],
                        target_ref=approval["target_ref"],
                    )
                    != approval["git_identity"]
                ):
                    raise SelfIterationError(
                        "Git application backend changed after review"
                    )

                claimed_at = _isoformat(now)
                claim = self._application_claim_record(
                    approval=approval,
                    approval_signature=signature_text,
                    reviewer_identity=reviewer_identity,
                    claimed_at=claimed_at,
                )
                self._patch_sandbox.claim_application(
                    approval["candidate_id"], challenge_id_text, claim
                )

                applied_at = _isoformat(self._now_utc())
                writer_receipt = self._application_writer.apply(
                    repo_root=self.repo_root,
                    expected_identity=approval["git_identity"],
                    approval=approval,
                    candidate_manifest=candidate,
                    candidate_contents=contents,
                    applied_at=applied_at,
                )
                result = build_signed_application_result(
                    approval=approval,
                    approval_signature=signature_text,
                    approval_key=approval_key,
                    writer_receipt=writer_receipt,
                    applied_at=applied_at,
                    signer_key=result_key,
                )
                validate_signed_application_result(result, self._verifier_key_provider)
                self._patch_sandbox.store_application_result(
                    approval["candidate_id"], result
                )

                fresh_ledger = self._load_ledger()
                fresh_proposal = self._find_proposal(fresh_ledger, proposal_id_text)
                self._application_challenge_event(fresh_proposal, challenge_id_text)
                if any(
                    event.get("type") == "reviewed_application_recorded"
                    and event.get("challenge_id") == challenge_id_text
                    for event in fresh_proposal["events"]
                ):
                    raise SelfIterationError(
                        "application challenge already has a recorded result"
                    )
                fresh_proposal["events"].append(self._application_result_event(result))
                self._save_ledger(fresh_ledger)
            except (ApplicationError, PatchSandboxError) as exc:
                raise SelfIterationError(str(exc)) from exc

        return {
            "result": result,
            "application_contract": application_contract(),
            "next_step": (
                "The signed result names a dedicated local branch eligible for canary "
                "review. It is not pushed, merged, deployed, or live."
            ),
            "authority_granted": False,
        }

    def application_status(
        self,
        *,
        proposal_id: Any,
        candidate_id: Any,
    ) -> dict[str, Any]:
        """Reconcile application reviews, claims, signed results, and Git refs."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        candidate_id_text = _required_text(candidate_id, "candidate_id", max_length=100)
        with self._lock:
            ledger = self._load_ledger()
            proposal = self._find_proposal(ledger, proposal_id_text)
            try:
                candidate = self._patch_sandbox.load_manifest(candidate_id_text)
                self._assert_candidate_binding(proposal=proposal, candidate=candidate)
                raw_claims = self._patch_sandbox.load_application_claims(
                    candidate_id_text
                )
                raw_results = self._patch_sandbox.load_application_results(
                    candidate_id_text
                )
            except PatchSandboxError as exc:
                raise SelfIterationError(str(exc)) from exc
            challenge_events = [
                event
                for event in proposal["events"]
                if event.get("type") == "application_review_challenge_issued"
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
                        "application claim challenge identifier is malformed"
                    )
                approval = approvals.get(challenge_id_value)
                if approval is None or challenge_id_value in claims:
                    raise SelfIterationError(
                        "application claim has no unique ledger-recorded approval"
                    )
                claims[challenge_id_value] = self._validate_application_claim(
                    raw_claim, approval=approval
                )
            results: dict[str, dict[str, Any]] = {}
            for raw_result in raw_results:
                try:
                    result = validate_signed_application_result(
                        raw_result, self._verifier_key_provider
                    )
                except ApplicationError as exc:
                    raise SelfIterationError(
                        "signed application result artifact is invalid"
                    ) from exc
                challenge_id_value = result["challenge_id"]
                approval = approvals.get(challenge_id_value)
                claim = claims.get(challenge_id_value)
                if (
                    approval is None
                    or claim is None
                    or challenge_id_value in results
                    or result["approval"] != approval
                    or result["approval_signature"] != claim["approval_signature"]
                    or result["application_approval_sha256"]
                    != application_approval_sha256(approval)
                ):
                    raise SelfIterationError(
                        "signed application result does not match its one-use approval"
                    )
                results[challenge_id_value] = result

            recorded_events = [
                event
                for event in proposal["events"]
                if event.get("type") == "reviewed_application_recorded"
                and event.get("candidate_id") == candidate_id_text
            ]
            recorded_by_challenge = {
                event["challenge_id"]: event for event in recorded_events
            }
            now = self._now_utc()
            statuses: list[dict[str, Any]] = []
            if self.repo_root is None and results:
                raise SelfIterationError("source repository not found")
            for challenge_id_value, approval in approvals.items():
                selected_claim = claims.get(challenge_id_value)
                selected_result = results.get(challenge_id_value)
                recorded = recorded_by_challenge.get(challenge_id_value)
                if recorded is not None and (
                    selected_result is None
                    or recorded != self._application_result_event(selected_result)
                ):
                    raise SelfIterationError(
                        "ledger application event does not match its signed result artifact"
                    )
                ref_intact = bool(
                    selected_result is not None
                    and self.repo_root is not None
                    and self._application_writer.verify_result(
                        repo_root=self.repo_root, result=selected_result
                    )
                )
                if selected_result is not None and not ref_intact:
                    state = "ref_integrity_failed"
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
                        "application_id": approval["application_id"],
                        "approval_sha256": application_approval_sha256(approval),
                        "issued_at": approval["issued_at"],
                        "challenge_expires_at": approval["challenge_expires_at"],
                        "reviewer_identity": approval["reviewer_identity"],
                        "target_ref": approval["target_ref"],
                        "state": state,
                        "claim": copy.deepcopy(selected_claim),
                        "result": copy.deepcopy(selected_result),
                        "ledger_recorded": recorded is not None,
                        "ref_integrity_verified": ref_intact,
                        "automatic_retry_allowed": False,
                        "eligible_for_canary_review": bool(
                            state == "recorded" and ref_intact
                        ),
                        "eligible_for_live_activation": False,
                        "authority_granted": False,
                    }
                )
            return {
                "proposal_id": proposal_id_text,
                "candidate_id": candidate_id_text,
                "candidate_sha256": candidate["candidate_sha256"],
                "applications": statuses,
                "application_contract": application_contract(),
                "eligible_for_live_activation": False,
                "authority_granted": False,
            }


__all__ = ["SelfIterationApplicationStageMixin"]
