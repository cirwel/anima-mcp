"""Ledger persistence, schema migration, and zero-trust validation for self-iteration.

This module owns the append-only self-iteration proposal ledger on disk,
migrates historical schemas (v1 through v7) without elevating caller claims,
and validates every event against the stage-specific authority contracts.
"""

from __future__ import annotations

import copy
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .atomic_write import atomic_json_write
from .self_iteration_application import (
    ApplicationError,
    application_approval_sha256,
    application_contract,
    validate_application_approval,
)
from .self_iteration_canary import (
    CanaryError,
    canary_approval_sha256,
    canary_contract,
    validate_canary_approval,
)
from .self_iteration_execution import (
    ExecutionError,
    execution_approval_sha256,
    execution_contract,
    validate_execution_approval,
)
from .self_iteration_sandbox import (
    STATIC_EVALUATOR_ID,
    sandbox_contract,
)
from .self_iteration_verification import (
    ATTESTATION_SCHEMA,
    SIGNATURE_ALGORITHM,
    VerificationError,
    VerifierKeyProvider,
    authenticated_identity,
    evaluate_verification,
    proposal_content_sha256,
    proposal_subject_fingerprint,
    verification_contract,
)

SCHEMA_VERSION = 7
APPLICATION_SCHEMA_VERSION = 6
EXECUTION_SCHEMA_VERSION = 5
SANDBOX_SCHEMA_VERSION = 4
VERIFICATION_SCHEMA_VERSION = 3
PROVENANCE_SCHEMA_VERSION = 2
PROVENANCE_SCHEMA = "anima.self_iteration.provenance.v1"

_V3_PROPOSAL_FIELDS = {
    "id",
    "created_at",
    "source_claim",
    "provenance",
    "proposer_identity",
    "trust_policy",
    "status",
    "observation",
    "hypothesis",
    "expected_outcome",
    "evidence",
    "evidence_epistemic_status",
    "target_paths",
    "verification",
    "rollback_plan",
    "risk",
    "boundaries",
    "code_fingerprint",
    "implementation_policy",
    "events",
    "content_sha256",
}

_PROPOSAL_STATUSES = {
    "protected_review_required",
    "human_review_required",
    "ready_for_isolated_implementation",
    "retained",
    "reverted",
    "measurement_inconclusive",
}


class SelfIterationError(ValueError):
    """Raised when an inspection or ledger operation violates its contract."""


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _claim_envelope(value: Any, *, field: str, legacy: bool = False) -> dict[str, Any]:
    return {
        "field": field,
        "value": value,
        "epistemic_status": "caller_claimed_legacy" if legacy else "caller_claimed",
        "verified": False,
        "authority_granted": False,
    }


def _zero_weight_trust_policy(*, legacy: bool = False) -> dict[str, Any]:
    return {
        "effective_weight": 0.0,
        "priority_eligible": False,
        "automation_eligible": False,
        "authority_eligible": False,
        "evidence_status": "caller_asserted_legacy" if legacy else "caller_asserted",
        "reason": (
            "Legacy provenance is unavailable; claims are unverified."
            if legacy
            else "Request provenance identifies receipt context, not claim truth."
        ),
        "upgrade_required": (
            "An independent verifier must append a verification event before "
            "any policy may assign weight or authority."
        ),
    }


def _provenance_contract() -> dict[str, Any]:
    return {
        "schema": PROVENANCE_SCHEMA,
        "caller_labels_are_claims_only": True,
        "server_receipts_verify_request_context_only": True,
        "ledger_receipts_tamper_evident": False,
        "unverified_effective_weight": 0.0,
        "unverified_priority_eligible": False,
        "unverified_automation_eligible": False,
        "unverified_authority_eligible": False,
        "verification_upgrade": "independent_append_only_event_required",
    }


def _legacy_provenance(recorded_at: str | None, *, migrated_at: str) -> dict[str, Any]:
    return {
        "schema": PROVENANCE_SCHEMA,
        "recorded_by": "legacy_v1_migration",
        "recorded_at": recorded_at,
        "recorded_at_verified": False,
        "migration_observed_at": migrated_at,
        "transport": {"kind": "unknown", "server_observed": False},
        "authentication": {"method": "unknown", "verified": False},
        "actor": None,
        "integrity": {
            "tamper_evident": False,
            "cryptographically_signed": False,
            "note": "Schema v1 had no durable provenance attestation.",
        },
        "session": {
            "present": False,
            "identifier_sha256": None,
            "source": "legacy_unknown",
            "verified": False,
            "request_actor_verified": False,
            "note": "No trustworthy session provenance was stored in schema v1.",
        },
        "trust": {
            "classification": "legacy_unverified",
            "actor_authenticated": False,
            "claims_verified": False,
            "evidence_verified": False,
            "weighting_eligible": False,
            "authority_eligible": False,
        },
    }


class SelfIterationLedgerMixin:
    """Mixin providing ledger migration, validation, and persistence for SelfIterationSystem."""

    ledger_path: Path
    _clock: Callable[[], datetime]
    _verifier_key_provider: VerifierKeyProvider
    _lock: threading.RLock

    @staticmethod
    def _identity_from_provenance(
        provenance: Any, *, label: str, required: bool
    ) -> dict[str, str | None] | None:
        trust = provenance.get("trust") if isinstance(provenance, dict) else None
        claims_authenticated_actor = bool(
            isinstance(trust, dict) and trust.get("actor_authenticated") is True
        )
        if not claims_authenticated_actor and not required:
            return None
        try:
            return authenticated_identity(provenance)
        except VerificationError as exc:
            if required:
                raise SelfIterationError(
                    f"{label} lacks authenticated actor provenance"
                ) from exc
            raise SelfIterationError(f"{label} actor provenance is malformed") from exc

    def _ledger_summary(self) -> dict[str, Any]:
        try:
            with self._lock:
                ledger = self._load_ledger()
            legacy_count = sum(
                1
                for proposal in ledger["proposals"]
                if proposal.get("provenance", {}).get("trust", {}).get("classification")
                == "legacy_unverified"
            )
            states = [
                evaluate_verification(
                    proposal,
                    key_provider=self._verifier_key_provider,
                    now=self._clock(),
                )
                for proposal in ledger["proposals"]
            ]
            return {
                "proposal_count": len(ledger["proposals"]),
                "legacy_unverified_count": legacy_count,
                "verified_count": sum(
                    state["status"] == "verified" for state in states
                ),
                "schema_version": ledger["schema_version"],
                "provenance_contract": ledger["provenance_contract"],
                "verification_contract": ledger["verification_contract"],
                "sandbox_contract": ledger["sandbox_contract"],
                "execution_contract": ledger["execution_contract"],
                "application_contract": ledger["application_contract"],
                "canary_contract": ledger["canary_contract"],
            }
        except SelfIterationError as exc:
            return {
                "schema_version": SCHEMA_VERSION,
                "error": str(exc),
                "writes_allowed": False,
            }

    def _empty_ledger(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "provenance_contract": _provenance_contract(),
            "verification_contract": verification_contract(),
            "sandbox_contract": sandbox_contract(),
            "execution_contract": execution_contract(),
            "application_contract": application_contract(),
            "canary_contract": canary_contract(),
            "migrations": [],
            "proposals": [],
        }

    def _migrate_v1(self, data: dict[str, Any]) -> dict[str, Any]:
        """Reclassify schema-v1 attribution as explicit, zero-weight claims."""
        migrated_at = _isoformat(self._clock())
        ledger = copy.deepcopy(data)
        proposals = ledger.get("proposals")
        if not isinstance(proposals, list):
            raise SelfIterationError("self-iteration ledger proposals are malformed")

        for proposal in proposals:
            if not isinstance(proposal, dict):
                raise SelfIterationError(
                    "self-iteration ledger proposals are malformed"
                )
            proposal["source_claim"] = _claim_envelope(
                proposal.pop("source", "unknown"), field="source", legacy=True
            )
            proposal["provenance"] = _legacy_provenance(
                proposal.get("created_at"), migrated_at=migrated_at
            )
            proposal["trust_policy"] = _zero_weight_trust_policy(legacy=True)
            proposal["evidence_epistemic_status"] = "caller_asserted_legacy"

            events = proposal.setdefault("events", [])
            if not isinstance(events, list):
                raise SelfIterationError("self-iteration proposal events are malformed")
            for event in events:
                if not isinstance(event, dict):
                    raise SelfIterationError(
                        "self-iteration proposal events are malformed"
                    )
                if event.get("type") != "outcome_recorded":
                    continue
                event["measurement_source_claim"] = _claim_envelope(
                    event.pop("measurement_source", "unknown"),
                    field="measurement_source",
                    legacy=True,
                )
                event["provenance"] = _legacy_provenance(
                    event.get("at"), migrated_at=migrated_at
                )
                event["trust_policy"] = _zero_weight_trust_policy(legacy=True)
                event["evidence_epistemic_status"] = "caller_asserted_legacy"
            events.append(
                {
                    "type": "provenance_migrated",
                    "at": migrated_at,
                    "from_schema": 1,
                    "to_schema": PROVENANCE_SCHEMA_VERSION,
                    "classification": "legacy_unverified",
                    "authority_granted": False,
                }
            )

        ledger["schema_version"] = PROVENANCE_SCHEMA_VERSION
        ledger["provenance_contract"] = _provenance_contract()
        migrations = ledger.setdefault("migrations", [])
        if not isinstance(migrations, list):
            raise SelfIterationError("self-iteration ledger migrations are malformed")
        migrations.append(
            {
                "type": "schema_migration",
                "at": migrated_at,
                "from_schema": 1,
                "to_schema": PROVENANCE_SCHEMA_VERSION,
                "classification": "legacy_unverified",
            }
        )
        return ledger

    def _migrate_v2(self, data: dict[str, Any]) -> dict[str, Any]:
        """Bind every proposal to immutable content before signed verification."""
        migrated_at = _isoformat(self._clock())
        ledger = copy.deepcopy(data)
        for proposal in ledger["proposals"]:
            proposal["proposer_identity"] = self._identity_from_provenance(
                proposal.get("provenance"),
                label="self-iteration proposal",
                required=False,
            )
            proposal["content_sha256"] = proposal_content_sha256(proposal)
            proposal.setdefault("events", []).append(
                {
                    "type": "verification_schema_migrated",
                    "at": migrated_at,
                    "from_schema": PROVENANCE_SCHEMA_VERSION,
                    "to_schema": VERIFICATION_SCHEMA_VERSION,
                    "verification_status": "unverified",
                    "authority_granted": False,
                }
            )
        ledger["schema_version"] = VERIFICATION_SCHEMA_VERSION
        ledger["verification_contract"] = verification_contract()
        ledger["migrations"].append(
            {
                "type": "schema_migration",
                "at": migrated_at,
                "from_schema": PROVENANCE_SCHEMA_VERSION,
                "to_schema": VERIFICATION_SCHEMA_VERSION,
                "classification": "verification_requires_signed_attestation",
            }
        )
        return ledger

    def _migrate_v3(self, data: dict[str, Any]) -> dict[str, Any]:
        """Declare the quarantined artifact boundary without granting authority."""
        migrated_at = _isoformat(self._clock())
        ledger = copy.deepcopy(data)
        for proposal in ledger["proposals"]:
            proposal.setdefault("events", []).append(
                {
                    "type": "sandbox_schema_migrated",
                    "at": migrated_at,
                    "from_schema": VERIFICATION_SCHEMA_VERSION,
                    "to_schema": SANDBOX_SCHEMA_VERSION,
                    "authority_granted": False,
                }
            )
        ledger["schema_version"] = SANDBOX_SCHEMA_VERSION
        ledger["sandbox_contract"] = sandbox_contract()
        ledger["migrations"].append(
            {
                "type": "schema_migration",
                "at": migrated_at,
                "from_schema": VERIFICATION_SCHEMA_VERSION,
                "to_schema": SANDBOX_SCHEMA_VERSION,
                "classification": "quarantined_patch_static_evaluation_only",
            }
        )
        return ledger

    def _migrate_v4(self, data: dict[str, Any]) -> dict[str, Any]:
        """Add signed isolated execution without adding apply authority."""
        migrated_at = _isoformat(self._clock())
        ledger = copy.deepcopy(data)
        for proposal in ledger["proposals"]:
            proposal.setdefault("events", []).append(
                {
                    "type": "execution_schema_migrated",
                    "at": migrated_at,
                    "from_schema": SANDBOX_SCHEMA_VERSION,
                    "to_schema": EXECUTION_SCHEMA_VERSION,
                    "authority_granted": False,
                }
            )
        ledger["schema_version"] = EXECUTION_SCHEMA_VERSION
        ledger["execution_contract"] = execution_contract()
        ledger["migrations"].append(
            {
                "type": "schema_migration",
                "at": migrated_at,
                "from_schema": SANDBOX_SCHEMA_VERSION,
                "to_schema": EXECUTION_SCHEMA_VERSION,
                "classification": "externally_approved_isolated_execution_only",
            }
        )
        return ledger

    def _migrate_v5(self, data: dict[str, Any]) -> dict[str, Any]:
        """Add reviewed dedicated-ref application without live activation."""
        migrated_at = _isoformat(self._clock())
        ledger = copy.deepcopy(data)
        for proposal in ledger["proposals"]:
            proposal.setdefault("events", []).append(
                {
                    "type": "application_schema_migrated",
                    "at": migrated_at,
                    "from_schema": EXECUTION_SCHEMA_VERSION,
                    "to_schema": APPLICATION_SCHEMA_VERSION,
                    "authority_granted": False,
                }
            )
        ledger["schema_version"] = APPLICATION_SCHEMA_VERSION
        ledger["application_contract"] = application_contract()
        ledger["migrations"].append(
            {
                "type": "schema_migration",
                "at": migrated_at,
                "from_schema": EXECUTION_SCHEMA_VERSION,
                "to_schema": APPLICATION_SCHEMA_VERSION,
                "classification": "reviewed_dedicated_branch_application_only",
            }
        )
        return ledger

    def _migrate_v6(self, data: dict[str, Any]) -> dict[str, Any]:
        """Add signed transient canary evaluation without release authority."""
        migrated_at = _isoformat(self._clock())
        ledger = copy.deepcopy(data)
        for proposal in ledger["proposals"]:
            proposal.setdefault("events", []).append(
                {
                    "type": "canary_schema_migrated",
                    "at": migrated_at,
                    "from_schema": APPLICATION_SCHEMA_VERSION,
                    "to_schema": SCHEMA_VERSION,
                    "authority_granted": False,
                }
            )
        ledger["schema_version"] = SCHEMA_VERSION
        ledger["canary_contract"] = canary_contract()
        ledger["migrations"].append(
            {
                "type": "schema_migration",
                "at": migrated_at,
                "from_schema": APPLICATION_SCHEMA_VERSION,
                "to_schema": SCHEMA_VERSION,
                "classification": "signed_transient_canary_with_mandatory_restore",
            }
        )
        return ledger

    @staticmethod
    def _validate_claim(claim: Any, *, label: str) -> None:
        if (
            not isinstance(claim, dict)
            or claim.get("verified") is not False
            or claim.get("authority_granted") is not False
            or not isinstance(claim.get("field"), str)
            or "value" not in claim
        ):
            raise SelfIterationError(
                f"self-iteration ledger {label} is not a valid unverified claim"
            )

    @staticmethod
    def _validate_provenance(provenance: Any, *, label: str) -> None:
        trust = provenance.get("trust") if isinstance(provenance, dict) else None
        authentication = (
            provenance.get("authentication") if isinstance(provenance, dict) else None
        )
        actor = provenance.get("actor") if isinstance(provenance, dict) else None
        integrity = (
            provenance.get("integrity") if isinstance(provenance, dict) else None
        )
        authenticated_context = bool(
            isinstance(authentication, dict)
            and authentication.get("verified") is True
            and isinstance(actor, dict)
            and actor.get("verified") is True
            and isinstance(actor.get("id"), str)
            and actor.get("id", "").strip()
        )
        if (
            not isinstance(provenance, dict)
            or provenance.get("schema") != PROVENANCE_SCHEMA
            or not isinstance(provenance.get("recorded_by"), str)
            or not isinstance(authentication, dict)
            or not isinstance(trust, dict)
            or trust.get("actor_authenticated") is not authenticated_context
            or trust.get("claims_verified") is not False
            or trust.get("evidence_verified") is not False
            or trust.get("weighting_eligible") is not False
            or trust.get("authority_eligible") is not False
            or not isinstance(integrity, dict)
            or integrity.get("tamper_evident") is not False
            or integrity.get("cryptographically_signed") is not False
        ):
            raise SelfIterationError(
                f"self-iteration ledger {label} provenance violates the zero-trust contract"
            )
        if authenticated_context:
            try:
                authenticated_identity(provenance)
            except VerificationError as exc:
                raise SelfIterationError(
                    f"self-iteration ledger {label} authenticated identity is malformed"
                ) from exc

    @staticmethod
    def _validate_trust_policy(policy: Any, *, label: str) -> None:
        weight = policy.get("effective_weight") if isinstance(policy, dict) else None
        if (
            not isinstance(policy, dict)
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or weight != 0.0
            or policy.get("priority_eligible") is not False
            or policy.get("automation_eligible") is not False
            or policy.get("authority_eligible") is not False
        ):
            raise SelfIterationError(
                f"self-iteration ledger {label} trust policy violates the zero-weight contract"
            )

    def _validate_v2(self, data: dict[str, Any]) -> None:
        if data.get("provenance_contract") != _provenance_contract():
            raise SelfIterationError(
                "self-iteration ledger provenance contract is malformed"
            )
        if not isinstance(data.get("migrations"), list):
            raise SelfIterationError("self-iteration ledger migrations are malformed")

        for proposal in data["proposals"]:
            if not isinstance(proposal, dict):
                raise SelfIterationError(
                    "self-iteration ledger proposals are malformed"
                )
            if "source" in proposal:
                raise SelfIterationError(
                    "self-iteration ledger contains an authoritative-looking source field"
                )
            self._validate_claim(proposal.get("source_claim"), label="proposal source")
            self._validate_provenance(proposal.get("provenance"), label="proposal")
            self._validate_trust_policy(proposal.get("trust_policy"), label="proposal")
            events = proposal.get("events")
            if not isinstance(events, list):
                raise SelfIterationError("self-iteration proposal events are malformed")
            for event in events:
                if not isinstance(event, dict):
                    raise SelfIterationError(
                        "self-iteration proposal events are malformed"
                    )
                if event.get("type") != "outcome_recorded":
                    continue
                if "measurement_source" in event:
                    raise SelfIterationError(
                        "self-iteration outcome contains an authoritative-looking measurement field"
                    )
                self._validate_claim(
                    event.get("measurement_source_claim"),
                    label="outcome measurement source",
                )
                self._validate_provenance(event.get("provenance"), label="outcome")
                self._validate_trust_policy(event.get("trust_policy"), label="outcome")

    def _validate_v3(self, data: dict[str, Any]) -> None:
        self._validate_v2(data)
        if data.get("verification_contract") != verification_contract():
            raise SelfIterationError(
                "self-iteration ledger verification contract is malformed"
            )
        proposal_ids: set[str] = set()
        for proposal in data["proposals"]:
            if not set(proposal).issubset(_V3_PROPOSAL_FIELDS):
                raise SelfIterationError(
                    "self-iteration proposal contains unsupported unbound fields"
                )
            proposal_id = proposal.get("id")
            if (
                not isinstance(proposal_id, str)
                or not proposal_id
                or proposal_id in proposal_ids
            ):
                raise SelfIterationError(
                    "self-iteration proposal identifiers are malformed or duplicated"
                )
            proposal_ids.add(proposal_id)
            if proposal.get("status") not in _PROPOSAL_STATUSES:
                raise SelfIterationError("self-iteration proposal status is malformed")
            expected_digest = proposal_content_sha256(proposal)
            if proposal.get("content_sha256") != expected_digest:
                raise SelfIterationError(
                    "self-iteration proposal content digest is malformed"
                )
            expected_identity = self._identity_from_provenance(
                proposal.get("provenance"),
                label="self-iteration proposal",
                required=False,
            )
            if proposal.get("proposer_identity") != expected_identity:
                raise SelfIterationError(
                    "self-iteration proposal identity binding is malformed"
                )
            for event in proposal["events"]:
                if event.get("type") not in {
                    "verification_challenge_issued",
                    "verification_attested",
                }:
                    continue
                attestation = event.get("attestation")
                if (
                    not isinstance(attestation, dict)
                    or attestation.get("schema") != ATTESTATION_SCHEMA
                ):
                    raise SelfIterationError(
                        "self-iteration verification event is malformed"
                    )
                self._validate_provenance(
                    event.get("provenance"), label="verification event"
                )
                if event.get("authority_granted") is not False:
                    raise SelfIterationError(
                        "self-iteration verification event grants forbidden authority"
                    )
                if event.get("type") == "verification_challenge_issued":
                    if set(event) != {
                        "type",
                        "at",
                        "challenge_id",
                        "attestation",
                        "signing_sha256",
                        "provenance",
                        "authority_granted",
                    }:
                        raise SelfIterationError(
                            "self-iteration verification challenge fields are malformed"
                        )
                    signing_digest = event.get("signing_sha256")
                    if not isinstance(signing_digest, str) or not re.fullmatch(
                        r"[0-9a-f]{64}", signing_digest
                    ):
                        raise SelfIterationError(
                            "self-iteration verification challenge digest is malformed"
                        )
                if event.get("type") == "verification_attested":
                    if set(event) != {
                        "type",
                        "at",
                        "challenge_id",
                        "attestation",
                        "signature",
                        "provenance",
                        "authority_granted",
                    }:
                        raise SelfIterationError(
                            "self-iteration attestation event fields are malformed"
                        )
                    signature = event.get("signature")
                    if (
                        not isinstance(signature, dict)
                        or set(signature)
                        != {"algorithm", "key_id", "value", "assurance"}
                        or signature.get("algorithm") != SIGNATURE_ALGORITHM
                        or signature.get("assurance")
                        != "symmetric_mac_server_verifiable"
                        or not isinstance(signature.get("value"), str)
                    ):
                        raise SelfIterationError(
                            "self-iteration attestation signature is malformed"
                        )

    @staticmethod
    def _validate_hex_digest(value: Any, *, label: str) -> None:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise SelfIterationError(f"self-iteration {label} digest is malformed")

    def _validate_v4(self, data: dict[str, Any]) -> None:
        self._validate_v3(data)
        if data.get("sandbox_contract") != sandbox_contract():
            raise SelfIterationError(
                "self-iteration ledger sandbox contract is malformed"
            )
        for proposal in data["proposals"]:
            constructed: dict[str, dict[str, Any]] = {}
            evaluation_ids: set[str] = set()
            for event in proposal["events"]:
                event_type = event.get("type")
                if event_type == "sandbox_schema_migrated":
                    if (
                        set(event)
                        != {
                            "type",
                            "at",
                            "from_schema",
                            "to_schema",
                            "authority_granted",
                        }
                        or event
                        != {
                            "type": "sandbox_schema_migrated",
                            "at": event.get("at"),
                            "from_schema": VERIFICATION_SCHEMA_VERSION,
                            "to_schema": SANDBOX_SCHEMA_VERSION,
                            "authority_granted": False,
                        }
                        or not isinstance(event.get("at"), str)
                    ):
                        raise SelfIterationError(
                            "self-iteration sandbox migration event is malformed"
                        )
                    continue
                if event_type == "patch_candidate_constructed":
                    if set(event) != {
                        "type",
                        "at",
                        "candidate_id",
                        "candidate_sha256",
                        "proposal_content_sha256",
                        "source_fingerprint",
                        "author_identity",
                        "active_attestation_ids",
                        "paths",
                        "patch_sha256",
                        "provenance",
                        "execution_performed",
                        "tests_executed",
                        "live_source_writes",
                        "authority_granted",
                    }:
                        raise SelfIterationError(
                            "self-iteration patch construction event fields are malformed"
                        )
                    candidate_id = event.get("candidate_id")
                    if (
                        not isinstance(candidate_id, str)
                        or not re.fullmatch(r"sip-[0-9a-f]{32}", candidate_id)
                        or candidate_id in constructed
                    ):
                        raise SelfIterationError(
                            "self-iteration patch candidate identifier is malformed or duplicated"
                        )
                    self._validate_hex_digest(
                        event.get("candidate_sha256"), label="patch candidate"
                    )
                    self._validate_hex_digest(
                        event.get("patch_sha256"), label="patch artifact"
                    )
                    try:
                        subject_fingerprint = proposal_subject_fingerprint(proposal)
                    except VerificationError as exc:
                        raise SelfIterationError(
                            "self-iteration proposal source fingerprint is malformed"
                        ) from exc
                    attestation_ids = event.get("active_attestation_ids")
                    paths = event.get("paths")
                    if (
                        not isinstance(event.get("at"), str)
                        or event.get("proposal_content_sha256")
                        != proposal.get("content_sha256")
                        or event.get("source_fingerprint") != subject_fingerprint
                        or event.get("author_identity")
                        != proposal.get("proposer_identity")
                        or not isinstance(attestation_ids, list)
                        or not attestation_ids
                        or attestation_ids != sorted(attestation_ids)
                        or len(set(attestation_ids)) != len(attestation_ids)
                        or any(
                            not isinstance(item, str)
                            or not re.fullmatch(r"sia-[0-9a-f]{32}", item)
                            for item in attestation_ids
                        )
                        or not isinstance(paths, list)
                        or not paths
                        or paths != sorted(paths)
                        or len(set(paths)) != len(paths)
                        or not set(paths).issubset(
                            set(proposal.get("target_paths", []))
                        )
                        or event.get("execution_performed") is not False
                        or event.get("tests_executed") is not False
                        or event.get("live_source_writes") is not False
                        or event.get("authority_granted") is not False
                    ):
                        raise SelfIterationError(
                            "self-iteration patch construction event violates its boundary"
                        )
                    self._validate_provenance(
                        event.get("provenance"), label="patch construction event"
                    )
                    construction_actor = self._identity_from_provenance(
                        event.get("provenance"),
                        label="patch construction event",
                        required=True,
                    )
                    if construction_actor != event.get("author_identity"):
                        raise SelfIterationError(
                            "self-iteration patch author is not bound to request provenance"
                        )
                    constructed[candidate_id] = event
                    continue
                if event_type == "patch_candidate_evaluated":
                    if set(event) != {
                        "type",
                        "at",
                        "candidate_id",
                        "candidate_sha256",
                        "proposal_content_sha256",
                        "evaluation_id",
                        "evaluation_sha256",
                        "evaluator_id",
                        "evaluator_contract_sha256",
                        "requester_identity",
                        "status",
                        "eligible_for_external_review",
                        "eligible_for_execution",
                        "execution_performed",
                        "tests_executed",
                        "live_source_writes",
                        "authority_granted",
                        "provenance",
                    }:
                        raise SelfIterationError(
                            "self-iteration patch evaluation event fields are malformed"
                        )
                    candidate_id = event.get("candidate_id")
                    construction = constructed.get(candidate_id)
                    status = event.get("status")
                    evaluation_id = event.get("evaluation_id")
                    if (
                        construction is None
                        or not isinstance(event.get("at"), str)
                        or event.get("candidate_sha256")
                        != construction.get("candidate_sha256")
                        or event.get("proposal_content_sha256")
                        != proposal.get("content_sha256")
                        or not isinstance(evaluation_id, str)
                        or not re.fullmatch(r"sie-[0-9a-f]{32}", evaluation_id)
                        or evaluation_id in evaluation_ids
                        or event.get("evaluator_id") != STATIC_EVALUATOR_ID
                        or status not in {"static_checks_passed", "rejected"}
                        or event.get("eligible_for_external_review")
                        is not (status == "static_checks_passed")
                        or event.get("eligible_for_execution") is not False
                        or event.get("execution_performed") is not False
                        or event.get("tests_executed") is not False
                        or event.get("live_source_writes") is not False
                        or event.get("authority_granted") is not False
                    ):
                        raise SelfIterationError(
                            "self-iteration patch evaluation event violates its boundary"
                        )
                    self._validate_hex_digest(
                        event.get("evaluation_sha256"), label="patch evaluation"
                    )
                    self._validate_hex_digest(
                        event.get("evaluator_contract_sha256"),
                        label="patch evaluator contract",
                    )
                    self._validate_provenance(
                        event.get("provenance"), label="patch evaluation event"
                    )
                    requester_identity = self._identity_from_provenance(
                        event.get("provenance"),
                        label="patch evaluation event",
                        required=True,
                    )
                    if requester_identity != event.get("requester_identity"):
                        raise SelfIterationError(
                            "self-iteration patch evaluation requester is not bound to provenance"
                        )
                    evaluation_ids.add(evaluation_id)

    def _validate_v5(self, data: dict[str, Any]) -> None:
        self._validate_v4(data)
        if data.get("execution_contract") != execution_contract():
            raise SelfIterationError(
                "self-iteration ledger execution contract is malformed"
            )
        for proposal in data["proposals"]:
            challenges: dict[str, dict[str, Any]] = {}
            approval_ids: set[str] = set()
            executions: set[str] = set()
            executed_challenges: set[str] = set()
            for event in proposal["events"]:
                event_type = event.get("type")
                if event_type == "execution_schema_migrated":
                    if event != {
                        "type": "execution_schema_migrated",
                        "at": event.get("at"),
                        "from_schema": SANDBOX_SCHEMA_VERSION,
                        "to_schema": EXECUTION_SCHEMA_VERSION,
                        "authority_granted": False,
                    } or not isinstance(event.get("at"), str):
                        raise SelfIterationError(
                            "self-iteration execution migration event is malformed"
                        )
                    continue
                if event_type == "execution_approval_challenge_issued":
                    if set(event) != {
                        "type",
                        "at",
                        "approval",
                        "approval_sha256",
                        "requester_identity",
                        "provenance",
                        "execution_performed",
                        "tests_executed",
                        "live_source_writes",
                        "authority_granted",
                    }:
                        raise SelfIterationError(
                            "self-iteration execution challenge fields are malformed"
                        )
                    try:
                        approval = validate_execution_approval(event.get("approval"))
                        approval_digest = execution_approval_sha256(approval)
                    except ExecutionError as exc:
                        raise SelfIterationError(
                            "self-iteration execution approval is malformed"
                        ) from exc
                    challenge_id = approval["challenge_id"]
                    active_verifiers = [
                        attestation.get("verifier_identity")
                        for recorded in proposal["events"]
                        if recorded.get("type") == "verification_attested"
                        and isinstance(recorded.get("attestation"), dict)
                        and (attestation := recorded["attestation"]).get(
                            "attestation_id"
                        )
                        in approval["active_attestation_ids"]
                        and attestation.get("decision") == "verified"
                    ]
                    construction_matches = [
                        recorded
                        for recorded in proposal["events"]
                        if recorded.get("type") == "patch_candidate_constructed"
                        and recorded.get("candidate_id") == approval["candidate_id"]
                    ]
                    evaluation_matches = [
                        recorded
                        for recorded in proposal["events"]
                        if recorded.get("type") == "patch_candidate_evaluated"
                        and recorded.get("candidate_id") == approval["candidate_id"]
                        and recorded.get("evaluation_id") == approval["evaluation_id"]
                    ]
                    if (
                        challenge_id in challenges
                        or approval["approval_id"] in approval_ids
                        or approval["proposal_id"] != proposal.get("id")
                        or approval["proposal_content_sha256"]
                        != proposal.get("content_sha256")
                        or approval["source_fingerprint"]
                        != proposal_subject_fingerprint(proposal)
                        or len(active_verifiers)
                        != len(approval["active_attestation_ids"])
                        or approval["approver_identity"]
                        in [proposal.get("proposer_identity"), *active_verifiers]
                        or approval["result_signer"]["id"]
                        in {
                            identity.get("id")
                            for identity in [
                                proposal.get("proposer_identity"),
                                approval["approver_identity"],
                                *active_verifiers,
                            ]
                            if isinstance(identity, dict)
                        }
                        or len(construction_matches) != 1
                        or construction_matches[0].get("candidate_sha256")
                        != approval["candidate_sha256"]
                        or construction_matches[0].get("active_attestation_ids")
                        != approval["active_attestation_ids"]
                        or len(evaluation_matches) != 1
                        or evaluation_matches[0].get("evaluation_sha256")
                        != approval["evaluation_sha256"]
                        or evaluation_matches[0].get("status") != "static_checks_passed"
                        or event.get("at") != approval["issued_at"]
                        or event.get("approval_sha256") != approval_digest
                        or event.get("execution_performed") is not False
                        or event.get("tests_executed") is not False
                        or event.get("live_source_writes") is not False
                        or event.get("authority_granted") is not False
                    ):
                        raise SelfIterationError(
                            "self-iteration execution challenge violates its boundary"
                        )
                    self._validate_provenance(
                        event.get("provenance"), label="execution challenge event"
                    )
                    requester = self._identity_from_provenance(
                        event.get("provenance"),
                        label="execution challenge event",
                        required=True,
                    )
                    if requester != event.get("requester_identity"):
                        raise SelfIterationError(
                            "execution challenge requester is not bound to provenance"
                        )
                    challenges[challenge_id] = event
                    approval_ids.add(approval["approval_id"])
                    continue
                if event_type == "isolated_execution_recorded":
                    if set(event) != {
                        "type",
                        "at",
                        "challenge_id",
                        "approval_id",
                        "approval_sha256",
                        "approval_signature_sha256",
                        "execution_id",
                        "result_sha256",
                        "candidate_id",
                        "candidate_sha256",
                        "evaluation_id",
                        "evaluation_sha256",
                        "approver_identity",
                        "result_signer_id",
                        "outcome",
                        "execution_performed",
                        "tests_executed",
                        "cleanup_confirmed",
                        "eligible_for_external_review",
                        "eligible_for_apply",
                        "live_source_writes",
                        "authority_granted",
                    }:
                        raise SelfIterationError(
                            "self-iteration execution result fields are malformed"
                        )
                    challenge = challenges.get(event.get("challenge_id"))
                    approval = challenge.get("approval") if challenge else None
                    challenge_approval_digest = (
                        challenge.get("approval_sha256") if challenge else None
                    )
                    execution_id = event.get("execution_id")
                    outcome = event.get("outcome")
                    if (
                        not isinstance(approval, dict)
                        or not isinstance(execution_id, str)
                        or not re.fullmatch(r"six-[0-9a-f]{32}", execution_id)
                        or execution_id in executions
                        or event.get("challenge_id") in executed_challenges
                        or event.get("approval_id") != approval.get("approval_id")
                        or event.get("approval_sha256") != challenge_approval_digest
                        or not isinstance(event.get("approval_signature_sha256"), str)
                        or event.get("candidate_id") != approval.get("candidate_id")
                        or event.get("candidate_sha256")
                        != approval.get("candidate_sha256")
                        or event.get("evaluation_id") != approval.get("evaluation_id")
                        or event.get("evaluation_sha256")
                        != approval.get("evaluation_sha256")
                        or event.get("approver_identity")
                        != approval.get("approver_identity")
                        or event.get("result_signer_id")
                        != approval.get("result_signer", {}).get("id")
                        or outcome
                        not in {
                            "passed",
                            "failed",
                            "timed_out",
                            "output_limit_exceeded",
                            "runner_error",
                        }
                        or not isinstance(event.get("execution_performed"), bool)
                        or event.get("tests_executed")
                        is not event.get("execution_performed")
                        or not isinstance(event.get("cleanup_confirmed"), bool)
                        or event.get("eligible_for_external_review")
                        is not (outcome == "passed")
                        or (
                            outcome == "passed"
                            and (
                                event.get("execution_performed") is not True
                                or event.get("cleanup_confirmed") is not True
                            )
                        )
                        or (
                            outcome
                            in {
                                "failed",
                                "timed_out",
                                "output_limit_exceeded",
                            }
                            and event.get("execution_performed") is not True
                        )
                        or event.get("eligible_for_apply") is not False
                        or event.get("live_source_writes") is not False
                        or event.get("authority_granted") is not False
                    ):
                        raise SelfIterationError(
                            "self-iteration execution result violates its boundary"
                        )
                    self._validate_hex_digest(
                        event.get("result_sha256"), label="execution result"
                    )
                    self._validate_hex_digest(
                        event.get("approval_signature_sha256"),
                        label="execution approval signature",
                    )
                    executions.add(execution_id)
                    executed_challenges.add(event["challenge_id"])

    def _validate_v6(self, data: dict[str, Any]) -> None:
        self._validate_v5(data)
        if data.get("application_contract") != application_contract():
            raise SelfIterationError(
                "self-iteration ledger application contract is malformed"
            )
        for proposal in data["proposals"]:
            challenges: dict[str, dict[str, Any]] = {}
            application_ids: set[str] = set()
            recorded_challenges: set[str] = set()
            result_ids: set[str] = set()
            for event in proposal["events"]:
                event_type = event.get("type")
                if event_type == "application_schema_migrated":
                    if event != {
                        "type": "application_schema_migrated",
                        "at": event.get("at"),
                        "from_schema": EXECUTION_SCHEMA_VERSION,
                        "to_schema": APPLICATION_SCHEMA_VERSION,
                        "authority_granted": False,
                    } or not isinstance(event.get("at"), str):
                        raise SelfIterationError(
                            "self-iteration application migration event is malformed"
                        )
                    continue
                if event_type == "application_review_challenge_issued":
                    if set(event) != {
                        "type",
                        "at",
                        "approval",
                        "approval_sha256",
                        "requester_identity",
                        "provenance",
                        "branch_created",
                        "live_source_writes",
                        "pushed",
                        "merged",
                        "deployed",
                        "authority_granted",
                    }:
                        raise SelfIterationError(
                            "self-iteration application challenge fields are malformed"
                        )
                    try:
                        approval = validate_application_approval(event.get("approval"))
                        approval_digest = application_approval_sha256(approval)
                    except ApplicationError as exc:
                        raise SelfIterationError(
                            "self-iteration application approval is malformed"
                        ) from exc
                    challenge_id = approval["challenge_id"]
                    active_verifiers = [
                        attestation.get("verifier_identity")
                        for recorded in proposal["events"]
                        if recorded.get("type") == "verification_attested"
                        and isinstance(recorded.get("attestation"), dict)
                        and (attestation := recorded["attestation"]).get(
                            "attestation_id"
                        )
                        in approval["active_attestation_ids"]
                        and attestation.get("decision") == "verified"
                    ]
                    execution_events = [
                        recorded
                        for recorded in proposal["events"]
                        if recorded.get("type") == "isolated_execution_recorded"
                        and recorded.get("execution_id") == approval["execution_id"]
                    ]
                    execution_challenges = [
                        recorded
                        for recorded in proposal["events"]
                        if recorded.get("type") == "execution_approval_challenge_issued"
                        and recorded.get("approval_sha256")
                        == approval["execution_approval_sha256"]
                    ]
                    prior_participant_ids = {
                        identity.get("id")
                        for identity in [
                            proposal.get("proposer_identity"),
                            *active_verifiers,
                            (
                                execution_challenges[0]["approval"].get(
                                    "approver_identity"
                                )
                                if len(execution_challenges) == 1
                                else None
                            ),
                        ]
                        if isinstance(identity, dict)
                    }
                    if len(execution_events) == 1:
                        prior_participant_ids.add(
                            execution_events[0].get("result_signer_id")
                        )
                    reviewer_id = approval["reviewer_identity"]["id"]
                    result_signer_id = approval["result_signer"]["id"]
                    if (
                        challenge_id in challenges
                        or approval["application_id"] in application_ids
                        or approval["proposal_id"] != proposal.get("id")
                        or approval["proposal_content_sha256"]
                        != proposal.get("content_sha256")
                        or approval["source_fingerprint"]
                        != proposal_subject_fingerprint(proposal)
                        or len(active_verifiers)
                        != len(approval["active_attestation_ids"])
                        or len(execution_events) != 1
                        or execution_events[0].get("candidate_id")
                        != approval["candidate_id"]
                        or execution_events[0].get("candidate_sha256")
                        != approval["candidate_sha256"]
                        or execution_events[0].get("result_sha256")
                        != approval["execution_result_sha256"]
                        or execution_events[0].get("approval_sha256")
                        != approval["execution_approval_sha256"]
                        or execution_events[0].get("at")
                        != approval["execution_finished_at"]
                        or execution_events[0].get("outcome") != "passed"
                        or execution_events[0].get("execution_performed") is not True
                        or execution_events[0].get("cleanup_confirmed") is not True
                        or execution_events[0].get("eligible_for_external_review")
                        is not True
                        or len(execution_challenges) != 1
                        or reviewer_id in prior_participant_ids
                        or result_signer_id in {reviewer_id, *prior_participant_ids}
                        or event.get("at") != approval["issued_at"]
                        or event.get("approval_sha256") != approval_digest
                        or event.get("branch_created") is not False
                        or event.get("live_source_writes") is not False
                        or event.get("pushed") is not False
                        or event.get("merged") is not False
                        or event.get("deployed") is not False
                        or event.get("authority_granted") is not False
                    ):
                        raise SelfIterationError(
                            "self-iteration application challenge violates its boundary"
                        )
                    self._validate_provenance(
                        event.get("provenance"), label="application challenge event"
                    )
                    requester = self._identity_from_provenance(
                        event.get("provenance"),
                        label="application challenge event",
                        required=True,
                    )
                    if (
                        requester != event.get("requester_identity")
                        or requester != approval["reviewer_identity"]
                    ):
                        raise SelfIterationError(
                            "application reviewer is not bound to request provenance"
                        )
                    challenges[challenge_id] = event
                    application_ids.add(approval["application_id"])
                    continue
                if event_type == "reviewed_application_recorded":
                    if set(event) != {
                        "type",
                        "at",
                        "challenge_id",
                        "application_id",
                        "approval_sha256",
                        "approval_signature_sha256",
                        "application_result_id",
                        "result_sha256",
                        "candidate_id",
                        "candidate_sha256",
                        "execution_id",
                        "execution_result_sha256",
                        "reviewer_identity",
                        "result_signer_id",
                        "target_ref",
                        "commit_oid",
                        "tree_oid",
                        "branch_created",
                        "live_source_writes",
                        "pushed",
                        "merged",
                        "deployed",
                        "eligible_for_canary_review",
                        "eligible_for_live_activation",
                        "authority_granted",
                    }:
                        raise SelfIterationError(
                            "self-iteration application result fields are malformed"
                        )
                    result_challenge = challenges.get(event.get("challenge_id"))
                    recorded_approval = (
                        result_challenge.get("approval") if result_challenge else None
                    )
                    result_approval_digest = (
                        result_challenge.get("approval_sha256")
                        if result_challenge
                        else None
                    )
                    result_id = event.get("application_result_id")
                    commit_oid = event.get("commit_oid")
                    tree_oid = event.get("tree_oid")
                    if (
                        not isinstance(recorded_approval, dict)
                        or event.get("challenge_id") in recorded_challenges
                        or not isinstance(result_id, str)
                        or not re.fullmatch(r"siar-[0-9a-f]{32}", result_id)
                        or result_id in result_ids
                        or event.get("application_id")
                        != recorded_approval.get("application_id")
                        or event.get("approval_sha256") != result_approval_digest
                        or event.get("candidate_id")
                        != recorded_approval.get("candidate_id")
                        or event.get("candidate_sha256")
                        != recorded_approval.get("candidate_sha256")
                        or event.get("execution_id")
                        != recorded_approval.get("execution_id")
                        or event.get("execution_result_sha256")
                        != recorded_approval.get("execution_result_sha256")
                        or event.get("reviewer_identity")
                        != recorded_approval.get("reviewer_identity")
                        or event.get("result_signer_id")
                        != recorded_approval.get("result_signer", {}).get("id")
                        or event.get("target_ref")
                        != recorded_approval.get("target_ref")
                        or not isinstance(commit_oid, str)
                        or not re.fullmatch(
                            r"[0-9a-f]{40}(?:[0-9a-f]{24})?", commit_oid
                        )
                        or not isinstance(tree_oid, str)
                        or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", tree_oid)
                        or len(commit_oid)
                        != len(recorded_approval["expected_parent_revision"])
                        or len(tree_oid) != len(commit_oid)
                        or event.get("branch_created") is not True
                        or event.get("live_source_writes") is not False
                        or event.get("pushed") is not False
                        or event.get("merged") is not False
                        or event.get("deployed") is not False
                        or event.get("eligible_for_canary_review") is not True
                        or event.get("eligible_for_live_activation") is not False
                        or event.get("authority_granted") is not False
                    ):
                        raise SelfIterationError(
                            "self-iteration application result violates its boundary"
                        )
                    self._validate_hex_digest(
                        event.get("approval_signature_sha256"),
                        label="application approval signature",
                    )
                    self._validate_hex_digest(
                        event.get("result_sha256"), label="application result"
                    )
                    recorded_challenges.add(event["challenge_id"])
                    result_ids.add(result_id)

    def _validate_v7(self, data: dict[str, Any]) -> None:
        self._validate_v6(data)
        if data.get("canary_contract") != canary_contract():
            raise SelfIterationError(
                "self-iteration ledger canary contract is malformed"
            )
        for proposal in data["proposals"]:
            challenges: dict[str, dict[str, Any]] = {}
            canary_ids: set[str] = set()
            recorded_challenges: set[str] = set()
            result_ids: set[str] = set()
            for event in proposal["events"]:
                event_type = event.get("type")
                if event_type == "canary_schema_migrated":
                    if event != {
                        "type": "canary_schema_migrated",
                        "at": event.get("at"),
                        "from_schema": APPLICATION_SCHEMA_VERSION,
                        "to_schema": SCHEMA_VERSION,
                        "authority_granted": False,
                    } or not isinstance(event.get("at"), str):
                        raise SelfIterationError(
                            "self-iteration canary migration event is malformed"
                        )
                    continue
                if event_type == "canary_review_challenge_issued":
                    if set(event) != {
                        "type",
                        "at",
                        "approval",
                        "approval_sha256",
                        "requester_identity",
                        "provenance",
                        "activation_performed",
                        "baseline_restore_attempted",
                        "persistent_activation_retained",
                        "authority_granted",
                    }:
                        raise SelfIterationError(
                            "self-iteration canary challenge fields are malformed"
                        )
                    try:
                        approval = validate_canary_approval(event.get("approval"))
                        approval_digest = canary_approval_sha256(approval)
                    except CanaryError as exc:
                        raise SelfIterationError(
                            "self-iteration canary approval is malformed"
                        ) from exc
                    challenge_id = approval["challenge_id"]
                    application_results = [
                        recorded
                        for recorded in proposal["events"]
                        if recorded.get("type") == "reviewed_application_recorded"
                        and recorded.get("application_result_id")
                        == approval["application_result_id"]
                    ]
                    application_challenges = [
                        recorded
                        for recorded in proposal["events"]
                        if recorded.get("type") == "application_review_challenge_issued"
                        and recorded.get("approval", {}).get("application_id")
                        == approval["application_id"]
                    ]
                    app_approval = (
                        application_challenges[0].get("approval")
                        if len(application_challenges) == 1
                        else None
                    )
                    execution_challenges = [
                        recorded
                        for recorded in proposal["events"]
                        if isinstance(app_approval, dict)
                        and recorded.get("type")
                        == "execution_approval_challenge_issued"
                        and recorded.get("approval_sha256")
                        == app_approval.get("execution_approval_sha256")
                    ]
                    active_verifiers = [
                        attestation.get("verifier_identity")
                        for recorded in proposal["events"]
                        if isinstance(app_approval, dict)
                        and recorded.get("type") == "verification_attested"
                        and isinstance(recorded.get("attestation"), dict)
                        and (attestation := recorded["attestation"]).get(
                            "attestation_id"
                        )
                        in app_approval.get("active_attestation_ids", [])
                        and attestation.get("decision") == "verified"
                    ]
                    prior_ids = {
                        identity.get("id")
                        for identity in [
                            proposal.get("proposer_identity"),
                            *active_verifiers,
                            (
                                execution_challenges[0]["approval"].get(
                                    "approver_identity"
                                )
                                if len(execution_challenges) == 1
                                else None
                            ),
                            (
                                app_approval.get("reviewer_identity")
                                if isinstance(app_approval, dict)
                                else None
                            ),
                        ]
                        if isinstance(identity, dict)
                    }
                    if isinstance(app_approval, dict):
                        prior_ids.add(app_approval.get("result_signer", {}).get("id"))
                    execution_results = [
                        recorded
                        for recorded in proposal["events"]
                        if isinstance(app_approval, dict)
                        and recorded.get("type") == "isolated_execution_recorded"
                        and recorded.get("execution_id")
                        == app_approval.get("execution_id")
                    ]
                    if len(execution_results) == 1:
                        prior_ids.add(execution_results[0].get("result_signer_id"))
                    reviewer_id = approval["reviewer_identity"]["id"]
                    supervisor_id = approval["supervisor_identity"]["result_signer"][
                        "id"
                    ]
                    if (
                        challenge_id in challenges
                        or approval["canary_id"] in canary_ids
                        or approval["proposal_id"] != proposal.get("id")
                        or approval["proposal_content_sha256"]
                        != proposal.get("content_sha256")
                        or len(application_results) != 1
                        or application_results[0].get("candidate_id")
                        != approval["candidate_id"]
                        or application_results[0].get("application_id")
                        != approval["application_id"]
                        or application_results[0].get("result_sha256")
                        != approval["application_result_sha256"]
                        or application_results[0].get("approval_sha256")
                        != approval["application_approval_sha256"]
                        or application_results[0].get("target_ref")
                        != approval["target_ref"]
                        or application_results[0].get("commit_oid")
                        != approval["candidate_commit_oid"]
                        or len(application_challenges) != 1
                        or not isinstance(app_approval, dict)
                        or app_approval.get("expected_parent_revision")
                        != approval["baseline_revision"]
                        or approval["application_reviewer_identity"]
                        != app_approval.get("reviewer_identity")
                        or approval["application_result_signer_id"]
                        != app_approval.get("result_signer", {}).get("id")
                        or len(active_verifiers)
                        != len(app_approval.get("active_attestation_ids", []))
                        or len(execution_challenges) != 1
                        or len(execution_results) != 1
                        or reviewer_id in prior_ids
                        or supervisor_id in {reviewer_id, *prior_ids}
                        or event.get("at") != approval["issued_at"]
                        or event.get("approval_sha256") != approval_digest
                        or event.get("activation_performed") is not False
                        or event.get("baseline_restore_attempted") is not False
                        or event.get("persistent_activation_retained") is not False
                        or event.get("authority_granted") is not False
                    ):
                        raise SelfIterationError(
                            "self-iteration canary challenge violates its boundary"
                        )
                    self._validate_provenance(
                        event.get("provenance"), label="canary challenge event"
                    )
                    requester = self._identity_from_provenance(
                        event.get("provenance"),
                        label="canary challenge event",
                        required=True,
                    )
                    if (
                        requester != event.get("requester_identity")
                        or requester != approval["reviewer_identity"]
                    ):
                        raise SelfIterationError(
                            "canary reviewer is not bound to request provenance"
                        )
                    challenges[challenge_id] = event
                    canary_ids.add(approval["canary_id"])
                    continue
                if event_type == "transient_canary_recorded":
                    if set(event) != {
                        "type",
                        "at",
                        "challenge_id",
                        "canary_id",
                        "approval_sha256",
                        "approval_signature_sha256",
                        "canary_result_id",
                        "result_sha256",
                        "candidate_id",
                        "application_result_id",
                        "application_result_sha256",
                        "reviewer_identity",
                        "supervisor_signer_id",
                        "outcome",
                        "activation_performed",
                        "health_checks_sha256",
                        "baseline_restore_attempted",
                        "baseline_restored",
                        "persistent_activation_retained",
                        "recommended_decision",
                        "eligible_for_merge_review",
                        "eligible_for_live_activation",
                        "authority_granted",
                    }:
                        raise SelfIterationError(
                            "self-iteration canary result fields are malformed"
                        )
                    challenge = challenges.get(event.get("challenge_id"))
                    recorded_approval = challenge.get("approval") if challenge else None
                    result_id = event.get("canary_result_id")
                    outcome = event.get("outcome")
                    baseline_restored = event.get("baseline_restored")
                    expected_decision = (
                        "keep_candidate_for_merge_review"
                        if outcome == "passed" and baseline_restored is True
                        else (
                            "operator_recovery_required"
                            if outcome == "rollback_failed"
                            or baseline_restored is False
                            else "reject_candidate"
                        )
                    )
                    eligible = outcome == "passed" and baseline_restored is True
                    if (
                        not isinstance(recorded_approval, dict)
                        or event.get("challenge_id") in recorded_challenges
                        or not isinstance(result_id, str)
                        or not re.fullmatch(r"sicr-[0-9a-f]{32}", result_id)
                        or result_id in result_ids
                        or event.get("canary_id") != recorded_approval.get("canary_id")
                        or event.get("approval_sha256")
                        != canary_approval_sha256(recorded_approval)
                        or event.get("candidate_id")
                        != recorded_approval.get("candidate_id")
                        or event.get("application_result_id")
                        != recorded_approval.get("application_result_id")
                        or event.get("application_result_sha256")
                        != recorded_approval.get("application_result_sha256")
                        or event.get("reviewer_identity")
                        != recorded_approval.get("reviewer_identity")
                        or event.get("supervisor_signer_id")
                        != recorded_approval.get("supervisor_identity", {})
                        .get("result_signer", {})
                        .get("id")
                        or outcome
                        not in {
                            "passed",
                            "failed",
                            "activation_failed",
                            "timed_out",
                            "supervisor_error",
                            "rollback_failed",
                        }
                        or not isinstance(event.get("activation_performed"), bool)
                        or not isinstance(event.get("baseline_restore_attempted"), bool)
                        or not isinstance(baseline_restored, bool)
                        or event.get("persistent_activation_retained") is not False
                        or event.get("recommended_decision") != expected_decision
                        or event.get("eligible_for_merge_review") is not eligible
                        or event.get("eligible_for_live_activation") is not False
                        or event.get("authority_granted") is not False
                    ):
                        raise SelfIterationError(
                            "self-iteration canary result violates its boundary"
                        )
                    for field, label in (
                        ("approval_signature_sha256", "canary approval signature"),
                        ("health_checks_sha256", "canary health checks"),
                        ("result_sha256", "canary result"),
                    ):
                        self._validate_hex_digest(event.get(field), label=label)
                    recorded_challenges.add(event["challenge_id"])
                    result_ids.add(result_id)

    def _load_ledger(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            return self._empty_ledger()
        try:
            data = json.loads(self.ledger_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SelfIterationError(
                "self-iteration ledger is unreadable; refusing to overwrite it"
            ) from exc
        if not isinstance(data, dict):
            raise SelfIterationError("self-iteration ledger has an unsupported schema")
        if not isinstance(data.get("proposals"), list):
            raise SelfIterationError("self-iteration ledger proposals are malformed")
        version = data.get("schema_version", 1)
        migrated = False
        if version == 1:
            data = self._migrate_v1(data)
            self._validate_v2(data)
            version = PROVENANCE_SCHEMA_VERSION
            migrated = True
        if version == PROVENANCE_SCHEMA_VERSION:
            self._validate_v2(data)
            data = self._migrate_v2(data)
            self._validate_v3(data)
            version = VERIFICATION_SCHEMA_VERSION
            migrated = True
        if version == VERIFICATION_SCHEMA_VERSION:
            self._validate_v3(data)
            data = self._migrate_v3(data)
            self._validate_v4(data)
            version = SANDBOX_SCHEMA_VERSION
            migrated = True
        if version == SANDBOX_SCHEMA_VERSION:
            self._validate_v4(data)
            data = self._migrate_v4(data)
            self._validate_v5(data)
            version = EXECUTION_SCHEMA_VERSION
            migrated = True
        if version == EXECUTION_SCHEMA_VERSION:
            self._validate_v5(data)
            data = self._migrate_v5(data)
            self._validate_v6(data)
            version = APPLICATION_SCHEMA_VERSION
            migrated = True
        if version == APPLICATION_SCHEMA_VERSION:
            self._validate_v6(data)
            data = self._migrate_v6(data)
            self._validate_v7(data)
            version = SCHEMA_VERSION
            migrated = True
        elif version == SCHEMA_VERSION:
            self._validate_v7(data)
        else:
            raise SelfIterationError("self-iteration ledger has an unsupported schema")
        if version != SCHEMA_VERSION:
            self._validate_v7(data)
            raise SelfIterationError("self-iteration ledger has an unsupported schema")
        if migrated:
            self._save_ledger(data)
        return data

    def _save_ledger(self, ledger: dict[str, Any]) -> None:
        atomic_json_write(self.ledger_path, ledger, indent=2)

    @staticmethod
    def _find_proposal(ledger: dict[str, Any], proposal_id: str) -> dict[str, Any]:
        proposal = next(
            (item for item in ledger["proposals"] if item.get("id") == proposal_id),
            None,
        )
        if proposal is None:
            raise SelfIterationError(f"proposal not found: {proposal_id}")
        return proposal
