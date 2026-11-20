"""Shared pure validation and claim-lifecycle helpers for self-iteration stages.

Provides path/string validators plus the parameterized stage-claim record,
validation, and challenge-lookup helpers shared across the execution,
application, and canary stage mixins.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .self_iteration_ledger import SelfIterationError
from .self_iteration_verification import (
    VerificationError,
    VerifierKey,
    parse_utc_timestamp,
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


@dataclass(frozen=True)
class _StageClaimSpec:
    """Declarative specification for a one-use stage approval claim."""

    stage: str
    schema: str
    stage_id_field: str
    parent_id_field: str | None
    actor_identity_field: str
    signature_actor_id_field: str
    extra_false_fields: tuple[str, ...]
    key_registry_unavailable_message: str
    signature_error_type: type[Exception]
    approval_sha256: Callable[[dict[str, Any]], str]
    signature_record: Callable[[dict[str, Any], Any], dict[str, Any]]
    verify_signature: Callable[[dict[str, Any], str, VerifierKey], bool]


def _build_stage_claim(
    spec: _StageClaimSpec,
    *,
    approval: dict[str, Any],
    approval_signature: str,
    actor_identity: dict[str, Any],
    claimed_at: str,
) -> dict[str, Any]:
    claim: dict[str, Any] = {
        "schema": spec.schema,
        "challenge_id": approval["challenge_id"],
        spec.stage_id_field: approval[spec.stage_id_field],
        "approval_sha256": spec.approval_sha256(approval),
        "approval_signature": spec.signature_record(approval, approval_signature),
        "proposal_id": approval["proposal_id"],
        "candidate_id": approval["candidate_id"],
    }
    if spec.parent_id_field is not None:
        claim[spec.parent_id_field] = approval[spec.parent_id_field]
    claim[spec.actor_identity_field] = copy.deepcopy(actor_identity)
    claim["claimed_at"] = claimed_at
    claim["automatic_retry_allowed"] = False
    for flag in spec.extra_false_fields:
        claim[flag] = False
    claim["authority_granted"] = False
    return claim


def _validate_stage_claim(
    spec: _StageClaimSpec,
    claim: Any,
    *,
    approval: dict[str, Any],
    verifier_key_provider: Any,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "challenge_id",
        spec.stage_id_field,
        "approval_sha256",
        "approval_signature",
        "proposal_id",
        "candidate_id",
        spec.actor_identity_field,
        "claimed_at",
        "automatic_retry_allowed",
        *spec.extra_false_fields,
        "authority_granted",
    }
    if spec.parent_id_field is not None:
        expected_keys.add(spec.parent_id_field)
    if not isinstance(claim, dict) or set(claim) != expected_keys:
        raise SelfIterationError(f"{spec.stage} claim fields are malformed")
    try:
        parse_utc_timestamp(claim.get("claimed_at"), "claimed_at")
    except VerificationError as exc:
        raise SelfIterationError(f"{spec.stage} claim timestamp is malformed") from exc
    raw_signature = claim.get("approval_signature")
    try:
        expected_signature = spec.signature_record(
            approval,
            raw_signature.get("value") if isinstance(raw_signature, dict) else None,
        )
    except spec.signature_error_type as exc:
        raise SelfIterationError(f"{spec.stage} claim signature is malformed") from exc
    binding_ok = (
        claim.get("schema") == spec.schema
        and claim.get("challenge_id") == approval["challenge_id"]
        and claim.get(spec.stage_id_field) == approval[spec.stage_id_field]
        and claim.get("approval_sha256") == spec.approval_sha256(approval)
        and raw_signature == expected_signature
        and claim.get("proposal_id") == approval["proposal_id"]
        and claim.get("candidate_id") == approval["candidate_id"]
        and (
            spec.parent_id_field is None
            or claim.get(spec.parent_id_field) == approval[spec.parent_id_field]
        )
        and claim.get(spec.actor_identity_field)
        == approval[spec.actor_identity_field]
        and claim.get("automatic_retry_allowed") is False
        and all(claim.get(flag) is False for flag in spec.extra_false_fields)
        and claim.get("authority_granted") is False
    )
    if not binding_ok:
        raise SelfIterationError(f"{spec.stage} claim binding is malformed")
    signature = claim["approval_signature"]
    try:
        key = verifier_key_provider(
            signature[spec.signature_actor_id_field], signature["key_id"]
        )
    except Exception as exc:
        raise SelfIterationError(spec.key_registry_unavailable_message) from exc
    if not isinstance(key, VerifierKey) or not spec.verify_signature(
        approval, signature["value"], key
    ):
        raise SelfIterationError(f"{spec.stage} claim approval signature is invalid")
    return copy.deepcopy(claim)


def _stage_challenge_event(
    proposal: dict[str, Any],
    challenge_id: str,
    *,
    event_type: str,
    stage: str,
) -> dict[str, Any]:
    matches = [
        event
        for event in proposal["events"]
        if event.get("type") == event_type
        and isinstance(event.get("approval"), dict)
        and event["approval"].get("challenge_id") == challenge_id
    ]
    if len(matches) != 1:
        raise SelfIterationError(f"challenge_id does not name one {stage} approval")
    return matches[0]


__all__ = [
    "_StageClaimSpec",
    "_build_stage_claim",
    "_normalize_repo_path",
    "_required_text",
    "_stage_challenge_event",
    "_string_list",
    "_validate_stage_claim",
]
