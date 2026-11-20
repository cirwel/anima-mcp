"""Stage-lifecycle pipeline mixin for self-iteration proposals and candidates.

Composes the verification and quarantined patch-sandbox controller methods
defined here with the isolated execution (`self_iteration_execution_stage`),
reviewed Git branch application (`self_iteration_application_stage`), and
transient canary supervision (`self_iteration_canary_stage`) mixins.
"""

from __future__ import annotations

import copy
import hashlib
import re
import uuid
from datetime import datetime, timedelta
from typing import Any

from .self_iteration_application_stage import SelfIterationApplicationStageMixin
from .self_iteration_canary_stage import SelfIterationCanaryStageMixin
from .self_iteration_execution import execution_profile
from .self_iteration_execution_stage import SelfIterationExecutionStageMixin
from .self_iteration_ledger import (
    SelfIterationError,
    _isoformat,
)
from .self_iteration_sandbox import (
    PatchSandboxError,
    sandbox_contract,
)
from .self_iteration_stage_helpers import (
    _normalize_repo_path,
    _required_text,
    _string_list,
)
from .self_iteration_verification import (
    MAX_ATTESTATION_VALIDITY,
    SIGNATURE_ALGORITHM,
    VerificationError,
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


class SelfIterationPipelineMixin(
    SelfIterationExecutionStageMixin,
    SelfIterationApplicationStageMixin,
    SelfIterationCanaryStageMixin,
):
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


__all__ = [
    "SelfIterationPipelineMixin",
    "_normalize_repo_path",
    "_required_text",
    "_string_list",
]
