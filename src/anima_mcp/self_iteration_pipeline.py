"""Stage-lifecycle pipeline mixin for self-iteration proposals and candidates.

Contains the stage controller methods (verification, quarantined patch
construction and static evaluation, isolated container execution, reviewed Git
branch application, and transient canary supervision) along with their
stage-specific claim, challenge, and validation helpers.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import PurePosixPath
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
from .self_iteration_execution import (
    RUNNER_SIGNER_ID_ENV,
    ExecutionError,
    approval_signature_record,
    approval_signing_input_b64,
    build_execution_approval,
    build_signed_execution_result,
    execution_approval_sha256,
    execution_contract,
    execution_profile,
    validate_execution_approval,
    validate_signed_execution_result,
    verify_execution_approval_signature,
)
from .self_iteration_ledger import (
    SelfIterationError,
    _isoformat,
)
from .self_iteration_sandbox import (
    PatchSandboxError,
    sandbox_contract,
)
from .self_iteration_verification import (
    MAX_ATTESTATION_VALIDITY,
    SIGNATURE_ALGORITHM,
    VerificationError,
    VerifierKey,
    attestation_signing_input_b64,
    build_attestation,
    canonical_json_bytes,
    parse_utc_timestamp,
    proposal_content_sha256,
    proposal_subject_fingerprint,
    validate_evidence,
    validate_recorded_attestation,
    verification_contract,
    verify_attestation_signature,
)


def _normalize_repo_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SelfIterationError("path must be a non-empty repository-relative string")
    raw = value.strip().replace("\\", "/")
    if "\x00" in raw or raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        raise SelfIterationError("path must stay inside the source repository")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SelfIterationError("path may not contain empty, '.' or '..' segments")
    normalized = str(PurePosixPath(raw))
    if normalized == ".":
        raise SelfIterationError("path must name a file or directory")
    return normalized



def _required_text(value: Any, field: str, *, max_length: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SelfIterationError(f"{field} must be a non-empty string")
    result = value.strip()
    if len(result) > max_length:
        raise SelfIterationError(f"{field} must be at most {max_length} characters")
    return result


def _string_list(
    value: Any,
    field: str,
    *,
    required: bool = True,
    max_items: int = 20,
    item_max_length: int = 1000,
) -> list[str]:
    if value is None and not required:
        return []
    if not isinstance(value, list) or (required and not value):
        requirement = "a non-empty list" if required else "a list"
        raise SelfIterationError(f"{field} must be {requirement} of strings")
    if len(value) > max_items:
        raise SelfIterationError(f"{field} may contain at most {max_items} items")
    return [_required_text(item, field, max_length=item_max_length) for item in value]



class SelfIterationPipelineMixin:
    """Stage-controller lifecycle methods for SelfIterationSystem."""

    def _validated_attestation(
        self,
        proposal: dict[str, Any],
        attestation_id: str,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        matches = [
            (index, event)
            for index, event in enumerate(proposal["events"])
            if event.get("type") == "verification_attested"
            and isinstance(event.get("attestation"), dict)
            and event["attestation"].get("attestation_id") == attestation_id
        ]
        if len(matches) != 1:
            raise SelfIterationError(
                "target_attestation_id does not name one recorded attestation"
            )
        event_index, event = matches[0]
        try:
            return validate_recorded_attestation(
                proposal=proposal,
                event=event,
                event_index=event_index,
                key_provider=self._verifier_key_provider,
                now=now,
            )
        except VerificationError as exc:
            raise SelfIterationError(
                "target attestation is not currently valid"
            ) from exc


    def _assert_current_proposal_source(
        self, proposal: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            expected = proposal_subject_fingerprint(proposal)
        except VerificationError as exc:
            raise SelfIterationError(
                "proposal source fingerprint is unavailable or malformed"
            ) from exc
        current = self._code_fingerprint()
        current_subject = {
            "revision": current.get("revision"),
            "manifest_sha256": current.get("manifest_sha256"),
        }
        if current_subject != expected:
            raise SelfIterationError(
                "live source no longer matches the verified proposal fingerprint"
            )
        return expected

    def _patch_eligible_state(
        self,
        proposal: dict[str, Any],
        *,
        expected_content_sha256: str,
        now: datetime,
    ) -> dict[str, Any]:
        if (
            proposal.get("content_sha256") != expected_content_sha256
            or proposal_content_sha256(proposal) != expected_content_sha256
        ):
            raise SelfIterationError(
                "proposal content digest changed; inspect the proposal again"
            )
        if (
            proposal.get("status") != "ready_for_isolated_implementation"
            or proposal.get("risk", {}).get("effective") != "low"
        ):
            raise SelfIterationError(
                "only low-risk proposals ready for isolated implementation may construct or evaluate patches"
            )
        target_paths = proposal.get("target_paths")
        if not isinstance(target_paths, list) or not target_paths:
            raise SelfIterationError("proposal target paths are malformed")
        current_boundaries = [self.classify_target(path) for path in target_paths]
        if any(
            boundary.get("boundary") != "bounded_candidate"
            or boundary.get("auto_implementation_eligible") is not True
            for boundary in current_boundaries
        ):
            raise SelfIterationError(
                "proposal targets are no longer within the bounded candidate allowlist"
            )
        state = self._verification_state(proposal, now=now)
        if (
            state.get("status") != "verified"
            or state.get("priority_eligible") is not True
        ):
            raise SelfIterationError(
                "a current independent verified attestation is required"
            )
        if (
            state.get("automation_eligible") is not False
            or state.get("authority_eligible") is not False
        ):
            raise SelfIterationError(
                "verification state violates the no-authority contract"
            )
        active_ids = state.get("active_attestation_ids")
        if not isinstance(active_ids, list) or not active_ids:
            raise SelfIterationError("active verification binding is unavailable")
        return state

    @staticmethod
    def _candidate_construction_event(
        proposal: dict[str, Any], candidate_id: str
    ) -> dict[str, Any]:
        matches = [
            event
            for event in proposal["events"]
            if event.get("type") == "patch_candidate_constructed"
            and event.get("candidate_id") == candidate_id
        ]
        if len(matches) != 1:
            raise SelfIterationError(
                "candidate is not bound to one recorded construction event"
            )
        return matches[0]

    def _assert_candidate_binding(
        self,
        *,
        proposal: dict[str, Any],
        candidate: dict[str, Any],
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise SelfIterationError("candidate manifest identifier is malformed")
        construction = self._candidate_construction_event(proposal, candidate_id)
        binding = candidate.get("proposal_binding")
        files = candidate.get("files")
        if not isinstance(binding, dict) or not isinstance(files, list):
            raise SelfIterationError("candidate manifest binding is malformed")
        try:
            source_fingerprint = proposal_subject_fingerprint(proposal)
        except VerificationError as exc:
            raise SelfIterationError(
                "proposal source fingerprint is unavailable or malformed"
            ) from exc
        expected_binding = {
            "proposal_id": proposal.get("id"),
            "proposal_content_sha256": proposal.get("content_sha256"),
            "source_fingerprint": source_fingerprint,
            "proposer_identity": proposal.get("proposer_identity"),
            "active_attestation_ids": construction.get("active_attestation_ids"),
        }
        paths: list[str] = []
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise SelfIterationError("candidate manifest file paths are malformed")
            paths.append(item["path"])
        paths.sort()
        if (
            binding != expected_binding
            or candidate.get("candidate_sha256") != construction.get("candidate_sha256")
            or candidate.get("patch", {}).get("sha256")
            != construction.get("patch_sha256")
            or paths != construction.get("paths")
        ):
            raise SelfIterationError(
                "candidate artifact does not match its ledger construction event"
            )
        if state is not None and binding.get("active_attestation_ids") != state.get(
            "active_attestation_ids"
        ):
            raise SelfIterationError(
                "candidate verification binding is stale; construct a new candidate"
            )
        return construction

    @staticmethod
    def _active_verifier_identities(
        proposal: dict[str, Any], state: dict[str, Any]
    ) -> list[dict[str, str | None]]:
        active_ids = set(state.get("active_attestation_ids", []))
        identities: list[dict[str, str | None]] = []
        for event in proposal["events"]:
            attestation = event.get("attestation")
            if (
                event.get("type") != "verification_attested"
                or not isinstance(attestation, dict)
                or attestation.get("attestation_id") not in active_ids
                or attestation.get("decision") != "verified"
            ):
                continue
            identity = attestation.get("verifier_identity")
            if not isinstance(identity, dict):
                raise SelfIterationError("active verifier identity is malformed")
            identities.append(copy.deepcopy(identity))
        if len(identities) != len(active_ids):
            raise SelfIterationError("active verifier identity binding is incomplete")
        return identities

    def _passing_evaluation(
        self,
        *,
        proposal: dict[str, Any],
        candidate: dict[str, Any],
        evaluation_id: str,
        expected_evaluation_sha256: str,
    ) -> dict[str, Any]:
        matches = [
            event
            for event in proposal["events"]
            if event.get("type") == "patch_candidate_evaluated"
            and event.get("candidate_id") == candidate.get("candidate_id")
            and event.get("evaluation_id") == evaluation_id
        ]
        if len(matches) != 1:
            raise SelfIterationError(
                "execution requires one recorded static evaluation event"
            )
        event = matches[0]
        if (
            event.get("status") != "static_checks_passed"
            or event.get("eligible_for_external_review") is not True
            or event.get("eligible_for_execution") is not False
            or event.get("evaluation_sha256") != expected_evaluation_sha256
        ):
            raise SelfIterationError(
                "execution requires the exact passing static evaluation digest"
            )
        try:
            artifacts = self._patch_sandbox.load_evaluations(candidate["candidate_id"])
        except PatchSandboxError as exc:
            raise SelfIterationError(str(exc)) from exc
        artifact_matches = [
            item for item in artifacts if item.get("evaluation_id") == evaluation_id
        ]
        if len(artifact_matches) != 1:
            raise SelfIterationError(
                "recorded static evaluation artifact is unavailable"
            )
        evaluation = artifact_matches[0]
        if (
            evaluation.get("evaluation_sha256") != expected_evaluation_sha256
            or evaluation.get("candidate_sha256") != candidate.get("candidate_sha256")
            or evaluation.get("status") != "static_checks_passed"
        ):
            raise SelfIterationError(
                "static evaluation artifact does not match its ledger event"
            )
        return evaluation

    @staticmethod
    def _execution_claim_record(
        *,
        approval: dict[str, Any],
        approval_signature: str,
        approver_identity: dict[str, Any],
        claimed_at: str,
    ) -> dict[str, Any]:
        return {
            "schema": "anima.self_iteration.execution_claim.v1",
            "challenge_id": approval["challenge_id"],
            "approval_id": approval["approval_id"],
            "approval_sha256": execution_approval_sha256(approval),
            "approval_signature": approval_signature_record(
                approval, approval_signature
            ),
            "proposal_id": approval["proposal_id"],
            "candidate_id": approval["candidate_id"],
            "approver_identity": copy.deepcopy(approver_identity),
            "claimed_at": claimed_at,
            "automatic_retry_allowed": False,
            "authority_granted": False,
        }

    def _validate_execution_claim(
        self, claim: Any, *, approval: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(claim, dict) or set(claim) != {
            "schema",
            "challenge_id",
            "approval_id",
            "approval_sha256",
            "approval_signature",
            "proposal_id",
            "candidate_id",
            "approver_identity",
            "claimed_at",
            "automatic_retry_allowed",
            "authority_granted",
        }:
            raise SelfIterationError("execution claim fields are malformed")
        try:
            parse_utc_timestamp(claim.get("claimed_at"), "claimed_at")
        except VerificationError as exc:
            raise SelfIterationError("execution claim timestamp is malformed") from exc
        raw_signature = claim.get("approval_signature")
        try:
            expected_signature = approval_signature_record(
                approval,
                (
                    raw_signature.get("value")
                    if isinstance(raw_signature, dict)
                    else None
                ),
            )
        except ExecutionError as exc:
            raise SelfIterationError("execution claim signature is malformed") from exc
        if (
            claim.get("schema") != "anima.self_iteration.execution_claim.v1"
            or claim.get("challenge_id") != approval["challenge_id"]
            or claim.get("approval_id") != approval["approval_id"]
            or claim.get("approval_sha256") != execution_approval_sha256(approval)
            or raw_signature != expected_signature
            or claim.get("proposal_id") != approval["proposal_id"]
            or claim.get("candidate_id") != approval["candidate_id"]
            or claim.get("approver_identity") != approval["approver_identity"]
            or claim.get("automatic_retry_allowed") is not False
            or claim.get("authority_granted") is not False
        ):
            raise SelfIterationError("execution claim binding is malformed")
        approval_signature = claim["approval_signature"]
        try:
            key = self._verifier_key_provider(
                approval_signature["approver_id"], approval_signature["key_id"]
            )
        except Exception as exc:
            raise SelfIterationError(
                "execution approval signing key registry is unavailable"
            ) from exc
        if not isinstance(key, VerifierKey) or not verify_execution_approval_signature(
            approval, approval_signature["value"], key
        ):
            raise SelfIterationError("execution claim approval signature is invalid")
        return copy.deepcopy(claim)

    @staticmethod
    def _execution_challenge_event(
        proposal: dict[str, Any], challenge_id: str
    ) -> dict[str, Any]:
        matches = [
            event
            for event in proposal["events"]
            if event.get("type") == "execution_approval_challenge_issued"
            and isinstance(event.get("approval"), dict)
            and event["approval"].get("challenge_id") == challenge_id
        ]
        if len(matches) != 1:
            raise SelfIterationError(
                "challenge_id does not name one execution approval"
            )
        return matches[0]

    def _passing_execution_result(
        self,
        *,
        proposal: dict[str, Any],
        candidate: dict[str, Any],
        execution_id: str,
        expected_result_sha256: str,
    ) -> dict[str, Any]:
        try:
            raw_results = self._patch_sandbox.load_execution_results(
                candidate["candidate_id"]
            )
        except PatchSandboxError as exc:
            raise SelfIterationError(str(exc)) from exc
        validated_results: list[dict[str, Any]] = []
        for raw_result in raw_results:
            try:
                result = validate_signed_execution_result(
                    raw_result, self._verifier_key_provider
                )
            except ExecutionError as exc:
                raise SelfIterationError(
                    "signed execution result artifact is invalid"
                ) from exc
            if result.get("execution_id") == execution_id:
                validated_results.append(result)
        if len(validated_results) != 1:
            raise SelfIterationError(
                "application requires one exact signed execution result"
            )
        result = validated_results[0]
        recorded = [
            event
            for event in proposal["events"]
            if event.get("type") == "isolated_execution_recorded"
            and event.get("execution_id") == execution_id
        ]
        if (
            result.get("result_sha256") != expected_result_sha256
            or result.get("candidate_id") != candidate.get("candidate_id")
            or result.get("candidate_sha256") != candidate.get("candidate_sha256")
            or result.get("outcome") != "passed"
            or result.get("execution_performed") is not True
            or result.get("tests_executed") is not True
            or result.get("cleanup_confirmed") is not True
            or result.get("eligible_for_external_review") is not True
            or result.get("eligible_for_apply") is not False
            or result.get("live_source_writes") is not False
            or len(recorded) != 1
            or recorded[0].get("result_sha256") != result["result_sha256"]
            or recorded[0].get("candidate_sha256") != result["candidate_sha256"]
            or recorded[0].get("approval_sha256") != result["approval_sha256"]
            or recorded[0].get("outcome") != "passed"
            or recorded[0].get("eligible_for_external_review") is not True
        ):
            raise SelfIterationError(
                "application requires the exact passing recorded execution digest"
            )
        challenge = self._execution_challenge_event(proposal, result["challenge_id"])
        if (
            challenge.get("approval") != result["approval"]
            or challenge.get("approval_sha256") != result["approval_sha256"]
        ):
            raise SelfIterationError(
                "signed execution result does not match its ledger approval"
            )
        return result

    @staticmethod
    def _application_claim_record(
        *,
        approval: dict[str, Any],
        approval_signature: str,
        reviewer_identity: dict[str, Any],
        claimed_at: str,
    ) -> dict[str, Any]:
        return {
            "schema": "anima.self_iteration.application_claim.v1",
            "challenge_id": approval["challenge_id"],
            "application_id": approval["application_id"],
            "approval_sha256": application_approval_sha256(approval),
            "approval_signature": application_signature_record(
                approval, approval_signature
            ),
            "proposal_id": approval["proposal_id"],
            "candidate_id": approval["candidate_id"],
            "execution_id": approval["execution_id"],
            "reviewer_identity": copy.deepcopy(reviewer_identity),
            "claimed_at": claimed_at,
            "automatic_retry_allowed": False,
            "authority_granted": False,
        }

    def _validate_application_claim(
        self, claim: Any, *, approval: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(claim, dict) or set(claim) != {
            "schema",
            "challenge_id",
            "application_id",
            "approval_sha256",
            "approval_signature",
            "proposal_id",
            "candidate_id",
            "execution_id",
            "reviewer_identity",
            "claimed_at",
            "automatic_retry_allowed",
            "authority_granted",
        }:
            raise SelfIterationError("application claim fields are malformed")
        try:
            parse_utc_timestamp(claim.get("claimed_at"), "claimed_at")
        except VerificationError as exc:
            raise SelfIterationError(
                "application claim timestamp is malformed"
            ) from exc
        raw_signature = claim.get("approval_signature")
        try:
            expected_signature = application_signature_record(
                approval,
                raw_signature.get("value") if isinstance(raw_signature, dict) else None,
            )
        except ApplicationError as exc:
            raise SelfIterationError(
                "application claim signature is malformed"
            ) from exc
        if (
            claim.get("schema") != "anima.self_iteration.application_claim.v1"
            or claim.get("challenge_id") != approval["challenge_id"]
            or claim.get("application_id") != approval["application_id"]
            or claim.get("approval_sha256") != application_approval_sha256(approval)
            or raw_signature != expected_signature
            or claim.get("proposal_id") != approval["proposal_id"]
            or claim.get("candidate_id") != approval["candidate_id"]
            or claim.get("execution_id") != approval["execution_id"]
            or claim.get("reviewer_identity") != approval["reviewer_identity"]
            or claim.get("automatic_retry_allowed") is not False
            or claim.get("authority_granted") is not False
        ):
            raise SelfIterationError("application claim binding is malformed")
        signature = claim["approval_signature"]
        try:
            key = self._verifier_key_provider(
                signature["reviewer_id"], signature["key_id"]
            )
        except Exception as exc:
            raise SelfIterationError(
                "application reviewer key registry is unavailable"
            ) from exc
        if not isinstance(
            key, VerifierKey
        ) or not verify_application_approval_signature(
            approval, signature["value"], key
        ):
            raise SelfIterationError("application claim approval signature is invalid")
        return copy.deepcopy(claim)

    @staticmethod
    def _application_challenge_event(
        proposal: dict[str, Any], challenge_id: str
    ) -> dict[str, Any]:
        matches = [
            event
            for event in proposal["events"]
            if event.get("type") == "application_review_challenge_issued"
            and isinstance(event.get("approval"), dict)
            and event["approval"].get("challenge_id") == challenge_id
        ]
        if len(matches) != 1:
            raise SelfIterationError(
                "challenge_id does not name one application approval"
            )
        return matches[0]

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

    @staticmethod
    def _canary_claim_record(
        *,
        approval: dict[str, Any],
        approval_signature: str,
        reviewer_identity: dict[str, Any],
        claimed_at: str,
    ) -> dict[str, Any]:
        return {
            "schema": "anima.self_iteration.canary_claim.v1",
            "challenge_id": approval["challenge_id"],
            "canary_id": approval["canary_id"],
            "approval_sha256": canary_approval_sha256(approval),
            "approval_signature": canary_signature_record(approval, approval_signature),
            "proposal_id": approval["proposal_id"],
            "candidate_id": approval["candidate_id"],
            "application_result_id": approval["application_result_id"],
            "reviewer_identity": copy.deepcopy(reviewer_identity),
            "claimed_at": claimed_at,
            "automatic_retry_allowed": False,
            "persistent_activation_allowed": False,
            "authority_granted": False,
        }

    def _validate_canary_claim(
        self, claim: Any, *, approval: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(claim, dict) or set(claim) != {
            "schema",
            "challenge_id",
            "canary_id",
            "approval_sha256",
            "approval_signature",
            "proposal_id",
            "candidate_id",
            "application_result_id",
            "reviewer_identity",
            "claimed_at",
            "automatic_retry_allowed",
            "persistent_activation_allowed",
            "authority_granted",
        }:
            raise SelfIterationError("canary claim fields are malformed")
        try:
            parse_utc_timestamp(claim.get("claimed_at"), "claimed_at")
        except VerificationError as exc:
            raise SelfIterationError("canary claim timestamp is malformed") from exc
        raw_signature = claim.get("approval_signature")
        try:
            expected_signature = canary_signature_record(
                approval,
                raw_signature.get("value") if isinstance(raw_signature, dict) else None,
            )
        except CanaryError as exc:
            raise SelfIterationError("canary claim signature is malformed") from exc
        if (
            claim.get("schema") != "anima.self_iteration.canary_claim.v1"
            or claim.get("challenge_id") != approval["challenge_id"]
            or claim.get("canary_id") != approval["canary_id"]
            or claim.get("approval_sha256") != canary_approval_sha256(approval)
            or raw_signature != expected_signature
            or claim.get("proposal_id") != approval["proposal_id"]
            or claim.get("candidate_id") != approval["candidate_id"]
            or claim.get("application_result_id") != approval["application_result_id"]
            or claim.get("reviewer_identity") != approval["reviewer_identity"]
            or claim.get("automatic_retry_allowed") is not False
            or claim.get("persistent_activation_allowed") is not False
            or claim.get("authority_granted") is not False
        ):
            raise SelfIterationError("canary claim binding is malformed")
        signature = claim["approval_signature"]
        try:
            key = self._verifier_key_provider(
                signature["reviewer_id"], signature["key_id"]
            )
        except Exception as exc:
            raise SelfIterationError(
                "canary reviewer key registry is unavailable"
            ) from exc
        if not isinstance(key, VerifierKey) or not verify_canary_approval_signature(
            approval, signature["value"], key
        ):
            raise SelfIterationError("canary claim approval signature is invalid")
        return copy.deepcopy(claim)

    @staticmethod
    def _canary_challenge_event(
        proposal: dict[str, Any], challenge_id: str
    ) -> dict[str, Any]:
        matches = [
            event
            for event in proposal["events"]
            if event.get("type") == "canary_review_challenge_issued"
            and isinstance(event.get("approval"), dict)
            and event["approval"].get("challenge_id") == challenge_id
        ]
        if len(matches) != 1:
            raise SelfIterationError("challenge_id does not name one canary approval")
        return matches[0]

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


    def prepare_verification(
        self,
        *,
        proposal_id: Any,
        verification_decision: Any,
        verification_statement: Any,
        verification_evidence: Any,
        expected_content_sha256: Any,
        expires_at: Any = None,
        target_attestation_id: Any = None,
    ) -> dict[str, Any]:
        """Append a one-time challenge for an independent verifier to sign."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        if verification_decision not in {"verified", "rejected", "revoke"}:
            raise SelfIterationError(
                "verification_decision must be one of: verified, rejected, revoke"
            )
        statement = _required_text(
            verification_statement, "verification_statement", max_length=4000
        )
        try:
            evidence = validate_evidence(verification_evidence)
        except VerificationError as exc:
            raise SelfIterationError(str(exc)) from exc
        expected_digest = _required_text(
            expected_content_sha256,
            "expected_content_sha256",
            max_length=64,
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise SelfIterationError(
                "expected_content_sha256 must be exactly 64 hexadecimal characters"
            )
        target_id = (
            _required_text(
                target_attestation_id,
                "target_attestation_id",
                max_length=100,
            )
            if target_attestation_id is not None
            else None
        )

        with self._lock:
            ledger = self._load_ledger()
            proposal = self._find_proposal(ledger, proposal_id_text)
            if proposal["content_sha256"] != expected_digest:
                raise SelfIterationError(
                    "proposal content digest changed; inspect the proposal again"
                )

            now = self._now_utc()
            recorded_at = _isoformat(now)
            receipt = self._server_provenance(recorded_at)
            verifier_identity = self._identity_from_provenance(
                receipt,
                label="verification request",
                required=True,
            )
            proposer_identity = self._identity_from_provenance(
                proposal.get("provenance"),
                label="proposal",
                required=True,
            )
            assert verifier_identity is not None
            assert proposer_identity is not None
            if proposal.get("proposer_identity") != proposer_identity:
                raise SelfIterationError(
                    "proposal proposer identity binding is invalid"
                )
            if verifier_identity == proposer_identity:
                raise SelfIterationError(
                    "the authenticated proposer may not verify its own proposal"
                )

            key = self._resolve_verifier_key(verifier_identity)
            state = self._verification_state(proposal, now=now)
            if state["status"] == "invalid":
                raise SelfIterationError(
                    "proposal contains an invalid attestation; refusing a new challenge"
                )

            expiry: datetime | None
            if verification_decision == "revoke":
                if expires_at is not None:
                    raise SelfIterationError("expires_at is not valid for a revocation")
                if target_id is None:
                    raise SelfIterationError(
                        "target_attestation_id is required for a revocation"
                    )
                target = self._validated_attestation(proposal, target_id, now=now)[
                    "attestation"
                ]
                if target.get("decision") == "revoke":
                    raise SelfIterationError("a revocation may not target a revocation")
                if target.get("verifier_identity") != verifier_identity:
                    raise SelfIterationError(
                        "only the original verifier may revoke an attestation"
                    )
                if target_id in state["revoked_attestation_ids"]:
                    raise SelfIterationError("target attestation is already revoked")
                expiry = None
            else:
                if target_id is not None:
                    raise SelfIterationError(
                        "target_attestation_id is only valid for a revocation"
                    )
                if any(
                    event.get("attestation", {}).get("verifier_identity")
                    == verifier_identity
                    and event.get("attestation", {}).get("attestation_id")
                    in state["active_attestation_ids"]
                    for event in proposal["events"]
                    if isinstance(event.get("attestation"), dict)
                ):
                    raise SelfIterationError(
                        "verifier already has an active verdict; revoke it first"
                    )
                if expires_at is None:
                    expiry = now + timedelta(days=1)
                else:
                    try:
                        expiry = parse_utc_timestamp(expires_at, "expires_at")
                    except VerificationError as exc:
                        raise SelfIterationError(str(exc)) from exc
                if expiry <= now:
                    raise SelfIterationError("expires_at must be in the future")
                if expiry - now > MAX_ATTESTATION_VALIDITY:
                    raise SelfIterationError(
                        "expires_at may be at most seven days after issuance"
                    )

            challenge_id = f"sic-{uuid.uuid4().hex}"
            attestation_id = f"sia-{uuid.uuid4().hex}"
            try:
                attestation = build_attestation(
                    proposal=proposal,
                    verifier_key=key,
                    verifier_identity=verifier_identity,
                    decision=verification_decision,
                    statement=statement,
                    evidence=evidence,
                    issued_at=now,
                    expires_at=expiry,
                    challenge_id=challenge_id,
                    attestation_id=attestation_id,
                    target_attestation_id=target_id,
                )
                signing_bytes = canonical_json_bytes(attestation)
            except VerificationError as exc:
                raise SelfIterationError(str(exc)) from exc
            signing_digest = hashlib.sha256(signing_bytes).hexdigest()
            proposal["events"].append(
                {
                    "type": "verification_challenge_issued",
                    "at": recorded_at,
                    "challenge_id": challenge_id,
                    "attestation": attestation,
                    "signing_sha256": signing_digest,
                    "provenance": receipt,
                    "authority_granted": False,
                }
            )
            self._save_ledger(ledger)

        return {
            "proposal_id": proposal_id_text,
            "challenge_id": challenge_id,
            "attestation_id": attestation_id,
            "attestation": attestation,
            "signing_input_b64": attestation_signing_input_b64(attestation),
            "signing_sha256": signing_digest,
            "signature_algorithm": SIGNATURE_ALGORITHM,
            "challenge_expires_at": attestation["challenge_expires_at"],
            "instructions": (
                "Base64url-decode signing_input_b64, compute HMAC-SHA256 with "
                "the configured verifier key, and submit the lowercase hexadecimal "
                "MAC with record_verification before challenge_expires_at."
            ),
            "authority_granted": False,
        }

    def record_verification(
        self,
        *,
        proposal_id: Any,
        challenge_id: Any,
        signature: Any,
    ) -> dict[str, Any]:
        """Verify a one-time challenge signature and append the signed verdict."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        challenge_id_text = _required_text(challenge_id, "challenge_id", max_length=100)
        if not re.fullmatch(r"sic-[0-9a-f]{32}", challenge_id_text):
            raise SelfIterationError("challenge_id is malformed")
        signature_text = _required_text(signature, "signature", max_length=64).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", signature_text):
            raise SelfIterationError(
                "signature must be exactly 64 hexadecimal characters"
            )

        with self._lock:
            ledger = self._load_ledger()
            proposal = self._find_proposal(ledger, proposal_id_text)
            challenges = [
                event
                for event in proposal["events"]
                if event.get("type") == "verification_challenge_issued"
                and event.get("challenge_id") == challenge_id_text
            ]
            if len(challenges) != 1:
                raise SelfIterationError(
                    "challenge_id does not name one issued verification challenge"
                )
            if any(
                event.get("type") == "verification_attested"
                and event.get("challenge_id") == challenge_id_text
                for event in proposal["events"]
            ):
                raise SelfIterationError("verification challenge has already been used")

            challenge = challenges[0]
            attestation = challenge.get("attestation")
            if not isinstance(attestation, dict):
                raise SelfIterationError("verification challenge is malformed")
            signing_digest = hashlib.sha256(
                canonical_json_bytes(attestation)
            ).hexdigest()
            if challenge.get("signing_sha256") != signing_digest:
                raise SelfIterationError("verification challenge digest is malformed")
            if attestation.get("proposal_id") != proposal_id_text:
                raise SelfIterationError(
                    "verification challenge belongs to another proposal"
                )
            if attestation.get("proposal_content_sha256") != proposal.get(
                "content_sha256"
            ) or proposal_content_sha256(proposal) != proposal.get("content_sha256"):
                raise SelfIterationError(
                    "verification challenge proposal digest is stale"
                )

            now = self._now_utc()
            try:
                challenge_expires_at = parse_utc_timestamp(
                    attestation.get("challenge_expires_at"),
                    "challenge_expires_at",
                )
            except VerificationError as exc:
                raise SelfIterationError(str(exc)) from exc
            if now >= challenge_expires_at:
                raise SelfIterationError("verification challenge has expired")

            recorded_at = _isoformat(now)
            receipt = self._server_provenance(recorded_at)
            verifier_identity = self._identity_from_provenance(
                receipt,
                label="verification request",
                required=True,
            )
            proposer_identity = self._identity_from_provenance(
                proposal.get("provenance"),
                label="proposal",
                required=True,
            )
            assert verifier_identity is not None
            assert proposer_identity is not None
            if attestation.get("verifier_identity") != verifier_identity:
                raise SelfIterationError(
                    "authenticated verifier does not own this challenge"
                )
            if attestation.get("proposer_identity") != proposer_identity:
                raise SelfIterationError("verification challenge proposer is stale")
            if verifier_identity == proposer_identity:
                raise SelfIterationError(
                    "the authenticated proposer may not verify its own proposal"
                )

            key_id = attestation.get("key_id")
            if not isinstance(key_id, str):
                raise SelfIterationError("verification challenge key ID is malformed")
            key = self._resolve_verifier_key(verifier_identity, key_id)
            if not verify_attestation_signature(attestation, signature_text, key):
                raise SelfIterationError("verification signature is invalid")

            state = self._verification_state(proposal, now=now)
            if state["status"] == "invalid":
                raise SelfIterationError(
                    "proposal contains an invalid attestation; refusing a new verdict"
                )
            attestation_decision = attestation.get("decision")
            if attestation_decision in {"verified", "rejected"} and any(
                event.get("attestation", {}).get("verifier_identity")
                == verifier_identity
                and event.get("attestation", {}).get("attestation_id")
                in state["active_attestation_ids"]
                for event in proposal["events"]
                if isinstance(event.get("attestation"), dict)
            ):
                raise SelfIterationError(
                    "verifier already has an active verdict; revoke it first"
                )
            if attestation_decision == "revoke":
                target_id = attestation.get("target_attestation_id")
                if not isinstance(target_id, str):
                    raise SelfIterationError("revocation target is malformed")
                target = self._validated_attestation(proposal, target_id, now=now)[
                    "attestation"
                ]
                if target.get("decision") == "revoke":
                    raise SelfIterationError("a revocation may not target a revocation")
                if target.get("verifier_identity") != verifier_identity:
                    raise SelfIterationError(
                        "only the original verifier may revoke an attestation"
                    )
                if target_id in state["revoked_attestation_ids"]:
                    raise SelfIterationError("target attestation is already revoked")

            event = {
                "type": "verification_attested",
                "at": recorded_at,
                "challenge_id": challenge_id_text,
                "attestation": copy.deepcopy(attestation),
                "signature": {
                    "algorithm": SIGNATURE_ALGORITHM,
                    "key_id": key_id,
                    "value": signature_text,
                    "assurance": "symmetric_mac_server_verifiable",
                },
                "provenance": receipt,
                "authority_granted": False,
            }
            proposal["events"].append(event)
            post_state = self._verification_state(proposal, now=now)
            if post_state["status"] == "invalid":
                proposal["events"].pop()
                raise SelfIterationError(
                    "signed verification failed post-append validation"
                )
            self._save_ledger(ledger)
        return self._public_proposal(proposal, now=now)

    def verification_status(self, *, proposal_id: Any) -> dict[str, Any]:
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        with self._lock:
            ledger = self._load_ledger()
            proposal = self._find_proposal(ledger, proposal_id_text)
            state = self._verification_state(proposal)
        return {
            "proposal_id": proposal_id_text,
            "content_sha256": proposal["content_sha256"],
            "verification_state": state,
            "verification_contract": verification_contract(),
        }

    def construct_patch(
        self,
        *,
        proposal_id: Any,
        expected_content_sha256: Any,
        changes: Any,
    ) -> dict[str, Any]:
        """Construct an inert, proposal-bound artifact outside live source."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        expected_digest = _required_text(
            expected_content_sha256,
            "expected_content_sha256",
            max_length=64,
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise SelfIterationError(
                "expected_content_sha256 must be exactly 64 hexadecimal characters"
            )
        if not isinstance(changes, list) or not changes:
            raise SelfIterationError("changes must be a non-empty list")

        candidate: dict[str, Any] | None = None
        with self._lock:
            try:
                ledger = self._load_ledger()
                proposal = self._find_proposal(ledger, proposal_id_text)
                now = self._now_utc()
                state = self._patch_eligible_state(
                    proposal,
                    expected_content_sha256=expected_digest,
                    now=now,
                )
                source_fingerprint = self._assert_current_proposal_source(proposal)
                proposal_targets = set(proposal["target_paths"])
                normalized_change_paths: list[str] = []
                for change in changes:
                    if not isinstance(change, dict):
                        raise SelfIterationError("each change must be an object")
                    raw_path = change.get("path")
                    if not isinstance(raw_path, str):
                        raise SelfIterationError("change path must be a string")
                    path = _normalize_repo_path(raw_path)
                    if path not in proposal_targets:
                        raise SelfIterationError(
                            f"candidate path was not bound by the proposal: {path}"
                        )
                    boundary = self.classify_target(path)
                    if (
                        boundary["boundary"] != "bounded_candidate"
                        or boundary["auto_implementation_eligible"] is not True
                    ):
                        raise SelfIterationError(
                            f"candidate path is outside the bounded allowlist: {path}"
                        )
                    normalized_change_paths.append(path)

                recorded_at = _isoformat(now)
                receipt = self._server_provenance(recorded_at)
                author_identity = self._identity_from_provenance(
                    receipt,
                    label="patch construction request",
                    required=True,
                )
                proposer_identity = self._identity_from_provenance(
                    proposal.get("provenance"),
                    label="proposal",
                    required=True,
                )
                assert author_identity is not None
                assert proposer_identity is not None
                if (
                    proposal.get("proposer_identity") != proposer_identity
                    or author_identity != proposer_identity
                ):
                    raise SelfIterationError(
                        "only the authenticated proposal author may construct its patch"
                    )

                candidate = self._patch_sandbox.construct(
                    proposal_id=proposal_id_text,
                    proposal_content_sha256=expected_digest,
                    source_fingerprint=source_fingerprint,
                    proposer_identity=proposer_identity,
                    author_identity=author_identity,
                    active_attestation_ids=state["active_attestation_ids"],
                    changes=changes,
                    created_at=recorded_at,
                )
                self._assert_current_proposal_source(proposal)
                event = {
                    "type": "patch_candidate_constructed",
                    "at": recorded_at,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_sha256": candidate["candidate_sha256"],
                    "proposal_content_sha256": expected_digest,
                    "source_fingerprint": source_fingerprint,
                    "author_identity": author_identity,
                    "active_attestation_ids": sorted(state["active_attestation_ids"]),
                    "paths": sorted(normalized_change_paths),
                    "patch_sha256": candidate["patch"]["sha256"],
                    "provenance": receipt,
                    "execution_performed": False,
                    "tests_executed": False,
                    "live_source_writes": False,
                    "authority_granted": False,
                }
                proposal["events"].append(event)
                self._save_ledger(ledger)
            except PatchSandboxError as exc:
                if candidate is not None:
                    try:
                        self._patch_sandbox.discard_candidate(candidate["candidate_id"])
                    except (OSError, PatchSandboxError):
                        pass
                raise SelfIterationError(str(exc)) from exc
            except BaseException:
                if candidate is not None:
                    try:
                        self._patch_sandbox.discard_candidate(candidate["candidate_id"])
                    except (OSError, PatchSandboxError):
                        pass
                raise

        assert candidate is not None
        return {
            "candidate": candidate,
            "sandbox_contract": sandbox_contract(),
            "next_step": (
                "Run evaluate_patch for non-executing static checks. Eligible Python era "
                "candidates may then be submitted to a distinct approver with prepare_execution."
            ),
            "authority_granted": False,
        }

    def evaluate_patch(
        self,
        *,
        proposal_id: Any,
        candidate_id: Any,
        expected_candidate_sha256: Any,
    ) -> dict[str, Any]:
        """Evaluate an artifact without importing or executing candidate code."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        candidate_id_text = _required_text(candidate_id, "candidate_id", max_length=100)
        expected_digest = _required_text(
            expected_candidate_sha256,
            "expected_candidate_sha256",
            max_length=64,
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise SelfIterationError(
                "expected_candidate_sha256 must be exactly 64 hexadecimal characters"
            )

        evaluation: dict[str, Any] | None = None
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
                self._assert_current_proposal_source(proposal)
                candidate = self._patch_sandbox.load_manifest(candidate_id_text)
                if candidate.get("candidate_sha256") != expected_digest:
                    raise SelfIterationError(
                        "candidate digest changed; inspect the candidate again"
                    )
                self._assert_candidate_binding(
                    proposal=proposal,
                    candidate=candidate,
                    state=state,
                )
                recorded_at = _isoformat(now)
                receipt = self._server_provenance(recorded_at)
                requester_identity = self._identity_from_provenance(
                    receipt,
                    label="patch evaluation request",
                    required=True,
                )
                assert requester_identity is not None
                evaluation = self._patch_sandbox.evaluate(
                    candidate_id=candidate_id_text,
                    expected_candidate_sha256=expected_digest,
                    evaluated_at=recorded_at,
                )
                self._assert_current_proposal_source(proposal)
                post_state = self._patch_eligible_state(
                    proposal,
                    expected_content_sha256=proposal["content_sha256"],
                    now=now,
                )
                self._assert_candidate_binding(
                    proposal=proposal,
                    candidate=candidate,
                    state=post_state,
                )
                event = {
                    "type": "patch_candidate_evaluated",
                    "at": recorded_at,
                    "candidate_id": candidate_id_text,
                    "candidate_sha256": expected_digest,
                    "proposal_content_sha256": proposal["content_sha256"],
                    "evaluation_id": evaluation["evaluation_id"],
                    "evaluation_sha256": evaluation["evaluation_sha256"],
                    "evaluator_id": evaluation["evaluator"]["id"],
                    "evaluator_contract_sha256": evaluation["evaluator"][
                        "contract_sha256"
                    ],
                    "requester_identity": requester_identity,
                    "status": evaluation["status"],
                    "eligible_for_external_review": evaluation[
                        "eligible_for_external_review"
                    ],
                    "eligible_for_execution": False,
                    "execution_performed": False,
                    "tests_executed": False,
                    "live_source_writes": False,
                    "authority_granted": False,
                    "provenance": receipt,
                }
                proposal["events"].append(event)
                self._save_ledger(ledger)
            except PatchSandboxError as exc:
                if evaluation is not None:
                    try:
                        self._patch_sandbox.discard_evaluation(
                            candidate_id_text,
                            evaluation["evaluation_id"],
                        )
                    except (OSError, PatchSandboxError):
                        pass
                raise SelfIterationError(str(exc)) from exc
            except BaseException:
                if evaluation is not None:
                    try:
                        self._patch_sandbox.discard_evaluation(
                            candidate_id_text,
                            evaluation["evaluation_id"],
                        )
                    except (OSError, PatchSandboxError):
                        pass
                raise

        assert evaluation is not None
        return {
            "evaluation": evaluation,
            "next_step": (
                "Static success permits external review only. An eligible Python era "
                "candidate still requires a distinct signed prepare_execution approval "
                "before its fixed tests may run in Docker."
            ),
            "authority_granted": False,
        }

    def patch_status(
        self,
        *,
        proposal_id: Any,
        candidate_id: Any,
        include_patch: Any = False,
    ) -> dict[str, Any]:
        """Read one candidate and reconcile artifacts with ledger events."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        candidate_id_text = _required_text(candidate_id, "candidate_id", max_length=100)
        if not isinstance(include_patch, bool):
            raise SelfIterationError("include_patch must be a boolean")
        with self._lock:
            ledger = self._load_ledger()
            proposal = self._find_proposal(ledger, proposal_id_text)
            if include_patch:
                receipt = self._server_provenance(_isoformat(self._now_utc()))
                self._identity_from_provenance(
                    receipt,
                    label="patch content request",
                    required=True,
                )
            try:
                artifact = self._patch_sandbox.status(
                    candidate_id_text,
                    include_patch=include_patch,
                )
            except PatchSandboxError as exc:
                raise SelfIterationError(str(exc)) from exc
            candidate = artifact["candidate"]
            self._assert_candidate_binding(
                proposal=proposal,
                candidate=candidate,
            )
            recorded_events = [
                event
                for event in proposal["events"]
                if event.get("type") == "patch_candidate_evaluated"
                and event.get("candidate_id") == candidate_id_text
            ]
            recorded_by_id = {
                event["evaluation_id"]: event for event in recorded_events
            }
            artifacts_by_id = {
                item["evaluation_id"]: item for item in artifact["evaluations"]
            }
            recorded_evaluations: list[dict[str, Any]] = []
            for evaluation_id, event in recorded_by_id.items():
                evaluation = artifacts_by_id.get(evaluation_id)
                if (
                    evaluation is None
                    or evaluation.get("evaluation_sha256")
                    != event.get("evaluation_sha256")
                    or evaluation.get("candidate_sha256")
                    != candidate.get("candidate_sha256")
                    or evaluation.get("status") != event.get("status")
                ):
                    raise SelfIterationError(
                        "recorded evaluation artifact does not match its ledger event"
                    )
                recorded_evaluations.append(evaluation)
            artifact["evaluations"] = recorded_evaluations
            artifact["unledgered_evaluation_artifact_count"] = len(
                set(artifacts_by_id) - set(recorded_by_id)
            )

            now = self._now_utc()
            verification_state = self._verification_state(proposal, now=now)
            try:
                self._assert_current_proposal_source(proposal)
            except SelfIterationError:
                source_current = False
            else:
                source_current = True
            attestation_binding_current = candidate["proposal_binding"][
                "active_attestation_ids"
            ] == verification_state.get("active_attestation_ids")
            execution_profile_supported = execution_profile(
                "display_era_pytest_v1"
            ).supports([item["path"] for item in candidate["files"]])
            static_pass_recorded = any(
                item.get("status") == "static_checks_passed"
                for item in recorded_evaluations
            )
            base_eligible = bool(
                proposal["status"] == "ready_for_isolated_implementation"
                and proposal.get("risk", {}).get("effective") == "low"
                and verification_state["status"] == "verified"
                and source_current
                and attestation_binding_current
            )
            artifact["current_state"] = {
                "proposal_status": proposal["status"],
                "verification_status": verification_state["status"],
                "source_fingerprint_current": source_current,
                "active_attestation_binding_current": attestation_binding_current,
                "eligible_for_new_static_evaluation": base_eligible,
                "eligible_for_execution_approval": bool(
                    base_eligible
                    and static_pass_recorded
                    and execution_profile_supported
                ),
                "execution_profile_supported": execution_profile_supported,
                "eligible_for_execution": False,
                "authority_granted": False,
            }
            artifact["sandbox_contract"] = sandbox_contract()
            return artifact

    def prepare_execution(
        self,
        *,
        proposal_id: Any,
        candidate_id: Any,
        expected_candidate_sha256: Any,
        evaluation_id: Any,
        expected_evaluation_sha256: Any,
        execution_profile_id: Any = "display_era_pytest_v1",
    ) -> dict[str, Any]:
        """Issue a short-lived plan for one externally approved execution."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        candidate_id_text = _required_text(candidate_id, "candidate_id", max_length=100)
        candidate_digest = _required_text(
            expected_candidate_sha256,
            "expected_candidate_sha256",
            max_length=64,
        ).lower()
        evaluation_id_text = _required_text(
            evaluation_id, "evaluation_id", max_length=100
        )
        evaluation_digest = _required_text(
            expected_evaluation_sha256,
            "expected_evaluation_sha256",
            max_length=64,
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", candidate_digest):
            raise SelfIterationError("expected_candidate_sha256 is malformed")
        if not re.fullmatch(r"sie-[0-9a-f]{32}", evaluation_id_text):
            raise SelfIterationError("evaluation_id is malformed")
        if not re.fullmatch(r"[0-9a-f]{64}", evaluation_digest):
            raise SelfIterationError("expected_evaluation_sha256 is malformed")
        try:
            profile = execution_profile(execution_profile_id)
        except ExecutionError as exc:
            raise SelfIterationError(str(exc)) from exc

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
                if candidate.get("candidate_sha256") != candidate_digest:
                    raise SelfIterationError(
                        "candidate digest changed; inspect the candidate again"
                    )
                self._assert_candidate_binding(
                    proposal=proposal, candidate=candidate, state=state
                )
                self._passing_evaluation(
                    proposal=proposal,
                    candidate=candidate,
                    evaluation_id=evaluation_id_text,
                    expected_evaluation_sha256=evaluation_digest,
                )
                candidate_paths = [item["path"] for item in candidate["files"]]
                if not profile.supports(candidate_paths):
                    raise SelfIterationError(
                        "candidate paths are outside the fixed execution profile"
                    )

                recorded_at = _isoformat(now)
                receipt = self._server_provenance(recorded_at)
                approver_identity = self._identity_from_provenance(
                    receipt,
                    label="execution approval request",
                    required=True,
                )
                assert approver_identity is not None
                proposer_identity = proposal.get("proposer_identity")
                verifier_identities = self._active_verifier_identities(proposal, state)
                if approver_identity == proposer_identity or approver_identity in (
                    verifier_identities
                ):
                    raise SelfIterationError(
                        "execution approver must differ from the proposer and active verifiers"
                    )
                approval_key = self._resolve_verifier_key(approver_identity)

                result_signer_id = os.environ.get(RUNNER_SIGNER_ID_ENV)
                if not isinstance(result_signer_id, str) or not result_signer_id:
                    raise SelfIterationError(
                        f"{RUNNER_SIGNER_ID_ENV} must name a dedicated result signer"
                    )
                participant_ids = {
                    identity.get("id")
                    for identity in [
                        proposer_identity,
                        approver_identity,
                        *verifier_identities,
                    ]
                    if isinstance(identity, dict)
                }
                if result_signer_id in participant_ids:
                    raise SelfIterationError(
                        "execution result signer must differ from all proposal participants"
                    )
                result_signer_identity = {
                    "kind": "service_signer",
                    "id": result_signer_id,
                    "issuer": None,
                }
                result_key = self._resolve_verifier_key(result_signer_identity)

                runner_identity = self._isolation_runner.probe()
                if self.repo_root is None:
                    raise SelfIterationError("source repository not found")
                revision = source_fingerprint.get("revision")
                if not isinstance(revision, str):
                    raise SelfIterationError(
                        "isolated execution requires a committed source revision"
                    )
                snapshot = self._workspace_builder.fingerprint(
                    repo_root=self.repo_root,
                    expected_revision=revision,
                    candidate_manifest=candidate,
                    candidate_contents=contents,
                )
                approval = build_execution_approval(
                    proposal_id=proposal_id_text,
                    proposal_content_sha256=proposal["content_sha256"],
                    source_fingerprint=source_fingerprint,
                    active_attestation_ids=state["active_attestation_ids"],
                    candidate_id=candidate_id_text,
                    candidate_sha256=candidate_digest,
                    evaluation_id=evaluation_id_text,
                    evaluation_sha256=evaluation_digest,
                    approver_identity=approver_identity,
                    approval_key_id=approval_key.key_id,
                    runner_identity=runner_identity,
                    profile=profile,
                    source_snapshot=snapshot,
                    result_signer_id=result_signer_id,
                    result_signer_key_id=result_key.key_id,
                    issued_at=now,
                )
                event = {
                    "type": "execution_approval_challenge_issued",
                    "at": recorded_at,
                    "approval": copy.deepcopy(approval),
                    "approval_sha256": execution_approval_sha256(approval),
                    "requester_identity": approver_identity,
                    "provenance": receipt,
                    "execution_performed": False,
                    "tests_executed": False,
                    "live_source_writes": False,
                    "authority_granted": False,
                }
                proposal["events"].append(event)
                self._save_ledger(ledger)
            except (ExecutionError, PatchSandboxError) as exc:
                raise SelfIterationError(str(exc)) from exc

        return {
            "approval": approval,
            "approval_sha256": execution_approval_sha256(approval),
            "signing_input_b64": approval_signing_input_b64(approval),
            "signature": {
                "algorithm": SIGNATURE_ALGORITHM,
                "key_id": approval_key.key_id,
                "assurance": "symmetric_mac_server_verifiable",
            },
            "next_step": (
                "The authenticated external approver signs these exact bytes, then submits "
                "execute_candidate before expiry. The challenge is consumed before execution."
            ),
            "authority_granted": False,
        }

    def execute_candidate(
        self,
        *,
        proposal_id: Any,
        challenge_id: Any,
        signature: Any,
    ) -> dict[str, Any]:
        """Consume an approval and run its exact candidate once in Docker."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        challenge_id_text = _required_text(challenge_id, "challenge_id", max_length=100)
        signature_text = _required_text(signature, "signature", max_length=64).lower()
        if not re.fullmatch(r"sixc-[0-9a-f]{32}", challenge_id_text):
            raise SelfIterationError("execution challenge_id is malformed")
        if not re.fullmatch(r"[0-9a-f]{64}", signature_text):
            raise SelfIterationError("execution approval signature is malformed")

        with self._lock:
            try:
                ledger = self._load_ledger()
                proposal = self._find_proposal(ledger, proposal_id_text)
                challenge_event = self._execution_challenge_event(
                    proposal, challenge_id_text
                )
                approval = validate_execution_approval(challenge_event["approval"])
                now = self._now_utc()
                issued = parse_utc_timestamp(approval["issued_at"], "issued_at")
                expires = parse_utc_timestamp(
                    approval["challenge_expires_at"], "challenge_expires_at"
                )
                if now < issued - timedelta(seconds=30) or now > expires:
                    raise SelfIterationError(
                        "execution approval is expired or not yet valid"
                    )
                receipt = self._server_provenance(_isoformat(now))
                approver_identity = self._identity_from_provenance(
                    receipt,
                    label="candidate execution request",
                    required=True,
                )
                assert approver_identity is not None
                if approver_identity != approval["approver_identity"]:
                    raise SelfIterationError(
                        "only the authenticated execution approver may consume this challenge"
                    )
                approval_key = self._resolve_verifier_key(
                    approver_identity, approval["approval_key_id"]
                )
                if not verify_execution_approval_signature(
                    approval, signature_text, approval_key
                ):
                    raise SelfIterationError("execution approval signature is invalid")

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
                        "proposal source or verification changed after execution approval"
                    )
                candidate, contents = self._patch_sandbox.execution_material(
                    approval["candidate_id"]
                )
                if candidate.get("candidate_sha256") != approval["candidate_sha256"]:
                    raise SelfIterationError("approved candidate digest is stale")
                self._assert_candidate_binding(
                    proposal=proposal, candidate=candidate, state=state
                )
                self._passing_evaluation(
                    proposal=proposal,
                    candidate=candidate,
                    evaluation_id=approval["evaluation_id"],
                    expected_evaluation_sha256=approval["evaluation_sha256"],
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
                if self._isolation_runner.probe() != approval["runner_identity"]:
                    raise SelfIterationError(
                        "Docker runner identity changed after execution approval"
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
                        "source snapshot changed after execution approval"
                    )

                claimed_at = _isoformat(now)
                claim = self._execution_claim_record(
                    approval=approval,
                    approval_signature=signature_text,
                    approver_identity=approver_identity,
                    claimed_at=claimed_at,
                )
                self._patch_sandbox.claim_execution(
                    approval["candidate_id"], challenge_id_text, claim
                )

                started_at = _isoformat(self._now_utc())
                profile = execution_profile(approval["profile"]["profile_id"])
                with self._workspace_builder.materialize(
                    repo_root=self.repo_root,
                    expected_revision=approval["source_snapshot"]["revision"],
                    candidate_manifest=candidate,
                    candidate_contents=contents,
                    expected_snapshot=approval["source_snapshot"],
                ) as workspace:
                    runner_receipt = self._isolation_runner.run(
                        workspace=workspace,
                        profile=profile,
                        expected_identity=approval["runner_identity"],
                    )
                finished_at = _isoformat(self._now_utc())
                result = build_signed_execution_result(
                    approval=approval,
                    approval_signature=signature_text,
                    approval_key=approval_key,
                    runner_receipt=runner_receipt,
                    started_at=started_at,
                    finished_at=finished_at,
                    signer_key=result_key,
                )
                validate_signed_execution_result(result, self._verifier_key_provider)
                self._patch_sandbox.store_execution_result(
                    approval["candidate_id"], result
                )

                fresh_ledger = self._load_ledger()
                fresh_proposal = self._find_proposal(fresh_ledger, proposal_id_text)
                self._execution_challenge_event(fresh_proposal, challenge_id_text)
                if any(
                    event.get("type") == "isolated_execution_recorded"
                    and event.get("challenge_id") == challenge_id_text
                    for event in fresh_proposal["events"]
                ):
                    raise SelfIterationError(
                        "execution challenge already has a recorded result"
                    )
                result_event = {
                    "type": "isolated_execution_recorded",
                    "at": finished_at,
                    "challenge_id": challenge_id_text,
                    "approval_id": approval["approval_id"],
                    "approval_sha256": execution_approval_sha256(approval),
                    "approval_signature_sha256": hashlib.sha256(
                        canonical_json_bytes(result["approval_signature"])
                    ).hexdigest(),
                    "execution_id": result["execution_id"],
                    "result_sha256": result["result_sha256"],
                    "candidate_id": approval["candidate_id"],
                    "candidate_sha256": approval["candidate_sha256"],
                    "evaluation_id": approval["evaluation_id"],
                    "evaluation_sha256": approval["evaluation_sha256"],
                    "approver_identity": approval["approver_identity"],
                    "result_signer_id": result_signer["id"],
                    "outcome": result["outcome"],
                    "execution_performed": result["execution_performed"],
                    "tests_executed": result["tests_executed"],
                    "cleanup_confirmed": result["cleanup_confirmed"],
                    "eligible_for_external_review": result[
                        "eligible_for_external_review"
                    ],
                    "eligible_for_apply": False,
                    "live_source_writes": False,
                    "authority_granted": False,
                }
                fresh_proposal["events"].append(result_event)
                self._save_ledger(fresh_ledger)
            except (ExecutionError, PatchSandboxError) as exc:
                raise SelfIterationError(str(exc)) from exc

        return {
            "result": result,
            "execution_contract": execution_contract(),
            "next_step": (
                "A passing signed result is eligible for external review only. It does not "
                "apply, commit, merge, deploy, or grant authority."
            ),
            "authority_granted": False,
        }

    def execution_status(
        self,
        *,
        proposal_id: Any,
        candidate_id: Any,
        include_output: Any = False,
    ) -> dict[str, Any]:
        """Reconcile approvals, one-use claims, signed results, and ledger events."""
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        candidate_id_text = _required_text(candidate_id, "candidate_id", max_length=100)
        if not isinstance(include_output, bool):
            raise SelfIterationError("include_output must be a boolean")
        with self._lock:
            ledger = self._load_ledger()
            proposal = self._find_proposal(ledger, proposal_id_text)
            if include_output:
                receipt = self._server_provenance(_isoformat(self._now_utc()))
                self._identity_from_provenance(
                    receipt, label="execution output request", required=True
                )
            try:
                candidate = self._patch_sandbox.load_manifest(candidate_id_text)
                self._assert_candidate_binding(proposal=proposal, candidate=candidate)
                raw_claims = self._patch_sandbox.load_execution_claims(
                    candidate_id_text
                )
                raw_results = self._patch_sandbox.load_execution_results(
                    candidate_id_text
                )
            except PatchSandboxError as exc:
                raise SelfIterationError(str(exc)) from exc

            challenge_events = [
                event
                for event in proposal["events"]
                if event.get("type") == "execution_approval_challenge_issued"
                and event.get("approval", {}).get("candidate_id") == candidate_id_text
            ]
            approvals = {
                event["approval"]["challenge_id"]: event["approval"]
                for event in challenge_events
            }
            claims: dict[str, dict[str, Any]] = {}
            for claim in raw_claims:
                challenge = claim.get("challenge_id")
                if not isinstance(challenge, str):
                    raise SelfIterationError(
                        "execution claim challenge identifier is malformed"
                    )
                approval = approvals.get(challenge)
                if approval is None:
                    raise SelfIterationError(
                        "execution claim has no ledger-recorded approval"
                    )
                claims[challenge] = self._validate_execution_claim(
                    claim, approval=approval
                )
            results: dict[str, dict[str, Any]] = {}
            for raw_result in raw_results:
                try:
                    result = validate_signed_execution_result(
                        raw_result, self._verifier_key_provider
                    )
                except ExecutionError as exc:
                    raise SelfIterationError(
                        "signed execution result artifact is invalid"
                    ) from exc
                challenge = result["challenge_id"]
                approval = approvals.get(challenge)
                if (
                    approval is None
                    or challenge not in claims
                    or result["approval"] != approval
                    or result["approval_signature"]
                    != claims[challenge]["approval_signature"]
                    or result["approval_id"] != approval["approval_id"]
                    or result["approval_sha256"] != execution_approval_sha256(approval)
                    or result["candidate_sha256"] != approval["candidate_sha256"]
                    or result["evaluation_sha256"] != approval["evaluation_sha256"]
                    or result["runner_identity"] != approval["runner_identity"]
                    or result["profile"] != approval["profile"]
                    or result["signature"]["signer_id"]
                    != approval["result_signer"]["id"]
                    or result["signature"]["key_id"]
                    != approval["result_signer"]["key_id"]
                    or challenge in results
                ):
                    raise SelfIterationError(
                        "signed execution result does not match its one-use approval"
                    )
                results[challenge] = result

            result_events = [
                event
                for event in proposal["events"]
                if event.get("type") == "isolated_execution_recorded"
                and event.get("candidate_id") == candidate_id_text
            ]
            recorded_by_challenge = {
                event["challenge_id"]: event for event in result_events
            }
            now = self._now_utc()
            statuses: list[dict[str, Any]] = []
            for challenge_id_text, approval in approvals.items():
                selected_claim = claims.get(challenge_id_text)
                selected_result = results.get(challenge_id_text)
                recorded = recorded_by_challenge.get(challenge_id_text)
                if recorded is not None:
                    expected_recorded = (
                        None
                        if selected_result is None
                        else {
                            "type": "isolated_execution_recorded",
                            "at": selected_result["finished_at"],
                            "challenge_id": challenge_id_text,
                            "approval_id": approval["approval_id"],
                            "approval_sha256": execution_approval_sha256(approval),
                            "approval_signature_sha256": hashlib.sha256(
                                canonical_json_bytes(
                                    selected_result["approval_signature"]
                                )
                            ).hexdigest(),
                            "execution_id": selected_result["execution_id"],
                            "result_sha256": selected_result["result_sha256"],
                            "candidate_id": selected_result["candidate_id"],
                            "candidate_sha256": selected_result["candidate_sha256"],
                            "evaluation_id": selected_result["evaluation_id"],
                            "evaluation_sha256": selected_result["evaluation_sha256"],
                            "approver_identity": approval["approver_identity"],
                            "result_signer_id": selected_result["signature"][
                                "signer_id"
                            ],
                            "outcome": selected_result["outcome"],
                            "execution_performed": selected_result[
                                "execution_performed"
                            ],
                            "tests_executed": selected_result["tests_executed"],
                            "cleanup_confirmed": selected_result["cleanup_confirmed"],
                            "eligible_for_external_review": selected_result[
                                "eligible_for_external_review"
                            ],
                            "eligible_for_apply": False,
                            "live_source_writes": False,
                            "authority_granted": False,
                        }
                    )
                    if recorded != expected_recorded:
                        raise SelfIterationError(
                            "ledger execution event does not match its signed result artifact"
                        )
                if selected_result is not None and recorded is not None:
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
                public_result: dict[str, Any] | None = None
                if selected_result is not None:
                    public_result = copy.deepcopy(selected_result)
                    if not include_output:
                        for stream_name in ("stdout", "stderr"):
                            stream = public_result[stream_name]
                            stream.pop("captured", None)
                        public_result["output_omitted"] = True
                statuses.append(
                    {
                        "challenge_id": challenge_id_text,
                        "approval_id": approval["approval_id"],
                        "approval_sha256": execution_approval_sha256(approval),
                        "issued_at": approval["issued_at"],
                        "challenge_expires_at": approval["challenge_expires_at"],
                        "approver_identity": approval["approver_identity"],
                        "profile_id": approval["profile"]["profile_id"],
                        "state": state,
                        "claim": copy.deepcopy(selected_claim),
                        "result": public_result,
                        "ledger_recorded": recorded is not None,
                        "automatic_retry_allowed": False,
                        "eligible_for_apply": False,
                        "authority_granted": False,
                    }
                )
            return {
                "proposal_id": proposal_id_text,
                "candidate_id": candidate_id_text,
                "candidate_sha256": candidate["candidate_sha256"],
                "executions": statuses,
                "output_included": include_output,
                "execution_contract": execution_contract(),
                "eligible_for_apply": False,
                "authority_granted": False,
            }

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



__all__ = [
    "SelfIterationPipelineMixin",
    "_normalize_repo_path",
    "_required_text",
    "_string_list",
]
