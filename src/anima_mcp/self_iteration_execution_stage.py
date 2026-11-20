"""Isolated container execution stage mixin for self-iteration proposals.

Contains the execution stage controller methods (`prepare_execution`,
`execute_candidate`, `execution_status`) and their execution-specific claim,
challenge, result-event, and passing-result helpers.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from datetime import timedelta
from typing import Any

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

_EXECUTION_CLAIM_SPEC = _StageClaimSpec(
    stage="execution",
    schema="anima.self_iteration.execution_claim.v1",
    stage_id_field="approval_id",
    parent_id_field=None,
    actor_identity_field="approver_identity",
    signature_actor_id_field="approver_id",
    extra_false_fields=(),
    key_registry_unavailable_message=(
        "execution approval signing key registry is unavailable"
    ),
    signature_error_type=ExecutionError,
    approval_sha256=execution_approval_sha256,
    signature_record=approval_signature_record,
    verify_signature=verify_execution_approval_signature,
)


class SelfIterationExecutionStageMixin:
    """Isolated Docker execution stage methods for SelfIterationPipelineMixin."""

    @staticmethod
    def _execution_claim_record(
        *,
        approval: dict[str, Any],
        approval_signature: str,
        approver_identity: dict[str, Any],
        claimed_at: str,
    ) -> dict[str, Any]:
        return _build_stage_claim(
            _EXECUTION_CLAIM_SPEC,
            approval=approval,
            approval_signature=approval_signature,
            actor_identity=approver_identity,
            claimed_at=claimed_at,
        )

    def _validate_execution_claim(
        self, claim: Any, *, approval: dict[str, Any]
    ) -> dict[str, Any]:
        return _validate_stage_claim(
            _EXECUTION_CLAIM_SPEC,
            claim,
            approval=approval,
            verifier_key_provider=self._verifier_key_provider,
        )

    @staticmethod
    def _execution_challenge_event(
        proposal: dict[str, Any], challenge_id: str
    ) -> dict[str, Any]:
        return _stage_challenge_event(
            proposal,
            challenge_id,
            event_type="execution_approval_challenge_issued",
            stage="execution",
        )

    @staticmethod
    def _execution_result_event(result: dict[str, Any]) -> dict[str, Any]:
        approval = result["approval"]
        return {
            "type": "isolated_execution_recorded",
            "at": result["finished_at"],
            "challenge_id": result["challenge_id"],
            "approval_id": approval["approval_id"],
            "approval_sha256": execution_approval_sha256(approval),
            "approval_signature_sha256": hashlib.sha256(
                canonical_json_bytes(result["approval_signature"])
            ).hexdigest(),
            "execution_id": result["execution_id"],
            "result_sha256": result["result_sha256"],
            "candidate_id": result["candidate_id"],
            "candidate_sha256": result["candidate_sha256"],
            "evaluation_id": result["evaluation_id"],
            "evaluation_sha256": result["evaluation_sha256"],
            "approver_identity": approval["approver_identity"],
            "result_signer_id": result["signature"]["signer_id"],
            "outcome": result["outcome"],
            "execution_performed": result["execution_performed"],
            "tests_executed": result["tests_executed"],
            "cleanup_confirmed": result["cleanup_confirmed"],
            "eligible_for_external_review": result["eligible_for_external_review"],
            "eligible_for_apply": False,
            "live_source_writes": False,
            "authority_granted": False,
        }

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
                prior_participant_ids = {
                    identity.get("id")
                    for identity in [proposer_identity, *verifier_identities]
                    if isinstance(identity, dict)
                    and isinstance(identity.get("id"), str)
                }
                if approver_identity["id"] in prior_participant_ids:
                    raise SelfIterationError(
                        "execution approver must differ from the proposer and active verifiers"
                    )
                approval_key = self._resolve_verifier_key(approver_identity)

                result_signer_id = os.environ.get(RUNNER_SIGNER_ID_ENV)
                if not isinstance(result_signer_id, str) or not result_signer_id:
                    raise SelfIterationError(
                        f"{RUNNER_SIGNER_ID_ENV} must name a dedicated result signer"
                    )
                participant_ids = {approver_identity["id"], *prior_participant_ids}
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
                fresh_proposal["events"].append(self._execution_result_event(result))
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
                if recorded is not None and (
                    selected_result is None
                    or recorded != self._execution_result_event(selected_result)
                ):
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


__all__ = ["SelfIterationExecutionStageMixin"]
