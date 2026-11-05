"""Read-only attention projection for self-iteration proposals and candidates.

This module projects the integrity-checked ledger and reconciled stage
artifacts into bounded, non-authoritative attention records for agents and
operators. Attention records never include signatures, signing payloads,
patch contents, or actions that consume an approval.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime
from typing import Any, Callable

from .self_iteration_ledger import SelfIterationError, _isoformat
from .self_iteration_verification import canonical_json_bytes

ATTENTION_SCHEMA = "anima.self_iteration.attention.v1"

_ATTENTION_PRIORITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


class SelfIterationAttentionMixin:
    """Mixin providing read-only attention projection for SelfIterationSystem."""

    list_proposals: Callable[..., dict[str, Any]]
    patch_status: Callable[..., dict[str, Any]]
    execution_status: Callable[..., dict[str, Any]]
    application_status: Callable[..., dict[str, Any]]
    canary_status: Callable[..., dict[str, Any]]
    _now_utc: Callable[[], datetime]

    @staticmethod
    def _attention_item(
        *,
        proposal: dict[str, Any],
        stage: str,
        state: str,
        priority: str,
        summary: str,
        next_action: str,
        required_role: str | None,
        status_action: str,
        candidate_id: str | None = None,
        reference_id: str | None = None,
        occurred_at: str | None = None,
        active: bool = True,
        detail: str | None = None,
    ) -> dict[str, Any]:
        """Build one stable, non-authoritative attention record.

        Attention records deliberately contain only read-only status-query
        coordinates. They never contain signatures, signing payloads, patch
        contents, or an action that can consume an approval.
        """
        if priority not in _ATTENTION_PRIORITY_ORDER:
            raise SelfIterationError("self-iteration attention priority is malformed")

        proposal_id = proposal.get("id")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise SelfIterationError("self-iteration attention proposal is malformed")
        reference = reference_id or str(proposal.get("content_sha256") or proposal_id)
        identity = {
            "proposal_id": proposal_id,
            "candidate_id": candidate_id,
            "stage": stage,
            "state": state,
            "reference_id": reference,
        }
        attention_id = (
            "si-attn-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:24]
        )
        arguments: dict[str, Any] = {
            "action": status_action,
            "proposal_id": proposal_id,
        }
        if candidate_id is not None:
            arguments["candidate_id"] = candidate_id

        source_claim_raw = proposal.get("source_claim")
        source_claim: dict[str, Any] = (
            source_claim_raw if isinstance(source_claim_raw, dict) else {}
        )
        provenance_raw = proposal.get("provenance")
        provenance: dict[str, Any] = (
            provenance_raw if isinstance(provenance_raw, dict) else {}
        )
        trust_raw = provenance.get("trust")
        trust: dict[str, Any] = trust_raw if isinstance(trust_raw, dict) else {}
        verification_raw = proposal.get("verification_state")
        verification: dict[str, Any] = (
            verification_raw if isinstance(verification_raw, dict) else {}
        )

        return {
            "attention_id": attention_id,
            "proposal_id": proposal_id,
            "candidate_id": candidate_id,
            "stage": stage,
            "state": state,
            "priority": priority,
            "active": active,
            "summary": summary,
            "next_action": next_action,
            "required_role": required_role,
            "reference_id": reference,
            "occurred_at": occurred_at or proposal.get("created_at"),
            "target_paths": copy.deepcopy(proposal.get("target_paths", [])),
            "claim_provenance": {
                "source_epistemic_status": source_claim.get(
                    "epistemic_status", "unknown"
                ),
                "request_trust_classification": trust.get("classification", "unknown"),
                "request_actor_authenticated": trust.get("actor_authenticated") is True,
                "claims_verified_by_request_provenance": trust.get("claims_verified")
                is True,
                "independent_verification_status": verification.get(
                    "status", "unknown"
                ),
                "effective_weight": verification.get("effective_weight", 0.0),
                "authority_granted": False,
            },
            "status_query": {
                "tool": "self_iteration",
                "arguments": arguments,
                "read_only": True,
            },
            "detail": detail,
            "acknowledgement_is_approval": False,
            "signed_approval_required": required_role
            in {
                "independent_verifier",
                "execution_approver",
                "application_reviewer",
                "canary_reviewer",
            },
            "authority_granted": False,
        }

    @staticmethod
    def _latest_attention_status(records: Any) -> dict[str, Any] | None:
        if not isinstance(records, list):
            return None
        valid = [item for item in records if isinstance(item, dict)]
        if not valid:
            return None
        return max(
            valid,
            key=lambda item: str(
                item.get("issued_at") or item.get("occurred_at") or ""
            ),
        )

    def _proposal_attention(self, proposal: dict[str, Any]) -> list[dict[str, Any]]:
        proposal_id = proposal["id"]
        status = proposal.get("status")
        terminal = {
            "retained": "The measured change was retained.",
            "reverted": "The measured change was reverted.",
            "measurement_inconclusive": "The measured outcome was inconclusive.",
        }
        if status in terminal:
            outcome_events = [
                event
                for event in proposal.get("events", [])
                if isinstance(event, dict) and event.get("type") == "outcome_recorded"
            ]
            latest = outcome_events[-1] if outcome_events else {}
            return [
                self._attention_item(
                    proposal=proposal,
                    stage="outcome",
                    state=str(status),
                    priority="low",
                    summary=terminal[str(status)],
                    next_action="No self-iteration action is pending.",
                    required_role=None,
                    status_action="list",
                    reference_id=str(
                        latest.get("at") or proposal.get("content_sha256")
                    ),
                    occurred_at=latest.get("at"),
                    active=False,
                )
            ]

        if status in {"protected_review_required", "human_review_required"}:
            return [
                self._attention_item(
                    proposal=proposal,
                    stage="proposal",
                    state=str(status),
                    priority="high",
                    summary=f"Proposal {proposal_id} requires caretaker review.",
                    next_action=(
                        "Review the proposal and its protected boundary; do not treat "
                        "this notification as approval."
                    ),
                    required_role="caretaker",
                    status_action="verification_status",
                )
            ]

        verification = proposal.get("verification_state")
        verification_state = (
            verification.get("status") if isinstance(verification, dict) else "invalid"
        )
        if verification_state == "invalid":
            return [
                self._attention_item(
                    proposal=proposal,
                    stage="verification",
                    state="verification_integrity_failed",
                    priority="critical",
                    summary=f"Proposal {proposal_id} has invalid verification evidence.",
                    next_action="Inspect the signed verification ledger before proceeding.",
                    required_role="operator_recovery",
                    status_action="verification_status",
                )
            ]
        if verification_state in {"rejected", "conflicted"}:
            return [
                self._attention_item(
                    proposal=proposal,
                    stage="verification",
                    state=str(verification_state),
                    priority="high",
                    summary=(
                        f"Proposal {proposal_id} verification is {verification_state}."
                    ),
                    next_action="Resolve the independent verdict before constructing code.",
                    required_role="caretaker",
                    status_action="verification_status",
                )
            ]
        if verification_state != "verified":
            return [
                self._attention_item(
                    proposal=proposal,
                    stage="verification",
                    state=str(verification_state),
                    priority="high",
                    summary=f"Proposal {proposal_id} awaits independent verification.",
                    next_action=(
                        "A distinct authenticated verifier must inspect and sign an exact "
                        "verification challenge."
                    ),
                    required_role="independent_verifier",
                    status_action="verification_status",
                )
            ]

        return [
            self._attention_item(
                proposal=proposal,
                stage="candidate",
                state="ready_for_candidate_construction",
                priority="medium",
                summary=f"Verified proposal {proposal_id} is ready for a quarantined candidate.",
                next_action=(
                    "The authenticated proposer may construct a bounded patch artifact; "
                    "the live worktree must remain unchanged."
                ),
                required_role="authenticated_proposer",
                status_action="verification_status",
            )
        ]

    def _candidate_attention(
        self, proposal: dict[str, Any], candidate_id: str
    ) -> list[dict[str, Any]]:
        proposal_id = proposal["id"]
        queries: tuple[tuple[str, str, Callable[..., dict[str, Any]]], ...] = (
            ("patch", "patch_status", self.patch_status),
            ("execution", "execution_status", self.execution_status),
            ("application", "application_status", self.application_status),
            ("canary", "canary_status", self.canary_status),
        )
        snapshots: dict[str, dict[str, Any]] = {}
        for stage, action, query in queries:
            try:
                snapshots[stage] = query(
                    proposal_id=proposal_id,
                    candidate_id=candidate_id,
                )
            except Exception as exc:
                return [
                    self._attention_item(
                        proposal=proposal,
                        candidate_id=candidate_id,
                        stage=stage,
                        state=f"{stage}_integrity_check_failed",
                        priority="critical",
                        summary=(
                            f"Candidate {candidate_id} failed {stage} integrity reconciliation."
                        ),
                        next_action=(
                            "An operator must inspect the ledger and artifacts; automatic "
                            "retry is not allowed."
                        ),
                        required_role="operator_recovery",
                        status_action=action,
                        detail=str(exc),
                    )
                ]

        patch = snapshots["patch"]
        if patch.get("unledgered_evaluation_artifact_count", 0):
            return [
                self._attention_item(
                    proposal=proposal,
                    candidate_id=candidate_id,
                    stage="patch",
                    state="unledgered_evaluation_artifact",
                    priority="critical",
                    summary=f"Candidate {candidate_id} has an unledgered evaluation artifact.",
                    next_action="Inspect the artifact and ledger before any further review.",
                    required_role="operator_recovery",
                    status_action="patch_status",
                    reference_id=str(patch.get("unledgered_evaluation_artifact_count")),
                )
            ]

        stage_records = (
            (
                "execution",
                "execution_status",
                snapshots["execution"].get("executions", []),
            ),
            (
                "application",
                "application_status",
                snapshots["application"].get("applications", []),
            ),
            ("canary", "canary_status", snapshots["canary"].get("canaries", [])),
        )
        critical_states = {
            "claimed_result_indeterminate",
            "signed_result_unledgered",
            "ref_integrity_failed",
            "recorded_recovery_required",
        }
        anomalies: list[dict[str, Any]] = []
        for stage, action, records in stage_records:
            for record in records if isinstance(records, list) else []:
                state = record.get("state") if isinstance(record, dict) else None
                if state not in critical_states:
                    continue
                challenge_id = str(record.get("challenge_id") or candidate_id)
                anomalies.append(
                    self._attention_item(
                        proposal=proposal,
                        candidate_id=candidate_id,
                        stage=stage,
                        state=str(state),
                        priority="critical",
                        summary=(
                            f"Candidate {candidate_id} requires operator recovery at the "
                            f"{stage} stage ({state})."
                        ),
                        next_action=(
                            "Establish actual state and reconcile signed artifacts before "
                            "creating any new review path."
                        ),
                        required_role="operator_recovery",
                        status_action=action,
                        reference_id=challenge_id,
                        occurred_at=record.get("issued_at"),
                    )
                )
        if anomalies:
            return anomalies

        canary = self._latest_attention_status(snapshots["canary"].get("canaries"))
        if canary is not None:
            state = str(canary.get("state"))
            challenge_id = str(canary.get("challenge_id") or candidate_id)
            if state == "awaiting_signature":
                return [
                    self._attention_item(
                        proposal=proposal,
                        candidate_id=candidate_id,
                        stage="canary",
                        state=state,
                        priority="high",
                        summary=f"Candidate {candidate_id} awaits signed canary review.",
                        next_action=(
                            "The distinct canary reviewer must sign the exact transient plan."
                        ),
                        required_role="canary_reviewer",
                        status_action="canary_status",
                        reference_id=challenge_id,
                        occurred_at=canary.get("issued_at"),
                    )
                ]
            if state == "expired_unclaimed":
                return [
                    self._attention_item(
                        proposal=proposal,
                        candidate_id=candidate_id,
                        stage="canary",
                        state=state,
                        priority="medium",
                        summary=f"Candidate {candidate_id} has an expired canary review.",
                        next_action="A distinct reviewer may prepare a new canary challenge.",
                        required_role="canary_reviewer",
                        status_action="canary_status",
                        reference_id=challenge_id,
                        occurred_at=canary.get("issued_at"),
                    )
                ]
            result_raw = canary.get("result")
            result: dict[str, Any] = result_raw if isinstance(result_raw, dict) else {}
            if canary.get("eligible_for_merge_review") is True:
                return [
                    self._attention_item(
                        proposal=proposal,
                        candidate_id=candidate_id,
                        stage="merge_review",
                        state="eligible_for_human_merge_review",
                        priority="high",
                        summary=(
                            f"Candidate {candidate_id} passed its restored transient canary."
                        ),
                        next_action=(
                            "A human may review the dedicated branch for merge; the candidate "
                            "is not live and this notice grants no authority."
                        ),
                        required_role="human_merge_reviewer",
                        status_action="canary_status",
                        reference_id=str(
                            result.get("canary_result_id") or challenge_id
                        ),
                        occurred_at=result.get("finished_at")
                        or canary.get("issued_at"),
                    )
                ]
            return [
                self._attention_item(
                    proposal=proposal,
                    candidate_id=candidate_id,
                    stage="canary",
                    state="candidate_rejected_by_canary",
                    priority="medium",
                    summary=f"Candidate {candidate_id} did not pass its transient canary.",
                    next_action="Review the signed result and record a measured outcome.",
                    required_role="caretaker",
                    status_action="canary_status",
                    reference_id=str(result.get("canary_result_id") or challenge_id),
                    occurred_at=result.get("finished_at") or canary.get("issued_at"),
                )
            ]

        application = self._latest_attention_status(
            snapshots["application"].get("applications")
        )
        if application is not None:
            state = str(application.get("state"))
            challenge_id = str(application.get("challenge_id") or candidate_id)
            if state == "awaiting_signature":
                summary = f"Candidate {candidate_id} awaits signed application review."
                next_action = (
                    "The distinct application reviewer must sign the exact plan."
                )
                priority = "high"
            elif state == "expired_unclaimed":
                summary = f"Candidate {candidate_id} has an expired application review."
                next_action = (
                    "A distinct reviewer may prepare a new application challenge."
                )
                priority = "medium"
            else:
                summary = (
                    f"Candidate {candidate_id} is ready for transient canary review."
                )
                next_action = "A new distinct reviewer may prepare the fixed external-supervisor canary."
                priority = "high"
            return [
                self._attention_item(
                    proposal=proposal,
                    candidate_id=candidate_id,
                    stage="application",
                    state=state,
                    priority=priority,
                    summary=summary,
                    next_action=next_action,
                    required_role=(
                        "application_reviewer"
                        if state in {"awaiting_signature", "expired_unclaimed"}
                        else "canary_reviewer"
                    ),
                    status_action="application_status",
                    reference_id=challenge_id,
                    occurred_at=application.get("issued_at"),
                )
            ]

        execution = self._latest_attention_status(
            snapshots["execution"].get("executions")
        )
        if execution is not None:
            state = str(execution.get("state"))
            challenge_id = str(execution.get("challenge_id") or candidate_id)
            if state == "awaiting_signature":
                summary = f"Candidate {candidate_id} awaits signed isolated execution."
                next_action = (
                    "The distinct execution approver must sign the exact plan."
                )
                priority = "high"
                role = "execution_approver"
            elif state == "expired_unclaimed":
                summary = f"Candidate {candidate_id} has an expired execution approval."
                next_action = (
                    "A distinct approver may prepare a new execution challenge."
                )
                priority = "medium"
                role = "execution_approver"
            else:
                result_raw = execution.get("result")
                result = result_raw if isinstance(result_raw, dict) else {}
                if result.get("outcome") == "passed":
                    summary = f"Candidate {candidate_id} passed isolated execution."
                    next_action = (
                        "A new distinct reviewer may prepare application review."
                    )
                    priority = "high"
                    role = "application_reviewer"
                else:
                    summary = f"Candidate {candidate_id} failed isolated execution."
                    next_action = (
                        "Review the signed result and revise or reject the candidate."
                    )
                    priority = "medium"
                    role = "caretaker"
            return [
                self._attention_item(
                    proposal=proposal,
                    candidate_id=candidate_id,
                    stage="execution",
                    state=state,
                    priority=priority,
                    summary=summary,
                    next_action=next_action,
                    required_role=role,
                    status_action="execution_status",
                    reference_id=challenge_id,
                    occurred_at=execution.get("issued_at"),
                )
            ]

        current = patch.get("current_state")
        current = current if isinstance(current, dict) else {}
        if current.get("source_fingerprint_current") is False:
            return [
                self._attention_item(
                    proposal=proposal,
                    candidate_id=candidate_id,
                    stage="patch",
                    state="source_fingerprint_stale",
                    priority="high",
                    summary=f"Candidate {candidate_id} no longer matches the running source.",
                    next_action="Create a new proposal and candidate against current source.",
                    required_role="authenticated_proposer",
                    status_action="patch_status",
                )
            ]
        evaluations = patch.get("evaluations")
        evaluations = evaluations if isinstance(evaluations, list) else []
        evaluation = evaluations[-1] if evaluations else None
        if evaluation is None:
            return [
                self._attention_item(
                    proposal=proposal,
                    candidate_id=candidate_id,
                    stage="patch",
                    state="awaiting_static_evaluation",
                    priority="medium",
                    summary=f"Candidate {candidate_id} awaits non-executing static evaluation.",
                    next_action="Run the fixed static evaluator; do not execute candidate code.",
                    required_role="static_evaluator",
                    status_action="patch_status",
                )
            ]
        evaluation_id = str(evaluation.get("evaluation_id") or candidate_id)
        if (
            evaluation.get("status") == "static_checks_passed"
            and current.get("eligible_for_execution_approval") is True
        ):
            return [
                self._attention_item(
                    proposal=proposal,
                    candidate_id=candidate_id,
                    stage="execution",
                    state="ready_for_execution_approval",
                    priority="high",
                    summary=f"Candidate {candidate_id} passed static evaluation.",
                    next_action=(
                        "A distinct execution approver may prepare the fixed Docker test plan."
                    ),
                    required_role="execution_approver",
                    status_action="patch_status",
                    reference_id=evaluation_id,
                    occurred_at=evaluation.get("evaluated_at"),
                )
            ]
        return [
            self._attention_item(
                proposal=proposal,
                candidate_id=candidate_id,
                stage="patch",
                state="static_evaluation_rejected",
                priority="medium",
                summary=f"Candidate {candidate_id} did not pass static evaluation.",
                next_action="Review the findings and construct a new candidate if warranted.",
                required_role="authenticated_proposer",
                status_action="patch_status",
                reference_id=evaluation_id,
                occurred_at=evaluation.get("evaluated_at"),
            )
        ]

    def attention(self, *, limit: int = 20) -> dict[str, Any]:
        """Return bounded, stable attention records for agents and observers.

        This is a server-derived, read-only projection of an integrity-checked
        ledger plus reconciled artifacts. Caller claims retain their explicit
        epistemic status; only separate review artifacts are signed. The
        projection grants no authority, and acknowledgement of a record is
        never equivalent to a review signature.
        """
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 50
        ):
            raise SelfIterationError("attention limit must be between 1 and 50")

        listed = self.list_proposals(limit=limit)
        items: list[dict[str, Any]] = []
        remaining_candidate_budget = limit
        unexamined_candidate_count = 0
        for proposal in listed["proposals"]:
            if proposal.get("status") in {
                "retained",
                "reverted",
                "measurement_inconclusive",
            }:
                items.extend(self._proposal_attention(proposal))
                continue
            candidate_ids: list[str] = []
            for event in proposal.get("events", []):
                if not isinstance(event, dict):
                    continue
                candidate_id = event.get("candidate_id")
                if (
                    event.get("type") == "patch_candidate_constructed"
                    and isinstance(candidate_id, str)
                    and candidate_id not in candidate_ids
                ):
                    candidate_ids.append(candidate_id)
            if candidate_ids:
                unexamined_candidate_count += max(
                    0, len(candidate_ids) - remaining_candidate_budget
                )
                selected_candidates = (
                    candidate_ids[-remaining_candidate_budget:]
                    if remaining_candidate_budget > 0
                    else []
                )
                remaining_candidate_budget -= len(selected_candidates)
                for candidate_id in reversed(selected_candidates):
                    items.extend(self._candidate_attention(proposal, candidate_id))
            else:
                items.extend(self._proposal_attention(proposal))

        # Stable sorts keep critical work first and favor newer records within
        # one priority. The public limit applies to returned records as well as
        # proposal/candidate examination, so downstream polling is bounded.
        items.sort(
            key=lambda item: (
                "" if item.get("occurred_at") is None else str(item["occurred_at"]),
                item["attention_id"],
            ),
            reverse=True,
        )
        items.sort(key=lambda item: (_ATTENTION_PRIORITY_ORDER[item["priority"]],))
        total_item_count = len(items)
        total_active_count = sum(item["active"] is True for item in items)
        items = items[:limit]
        active = [item for item in items if item["active"] is True]
        return {
            "schema": ATTENTION_SCHEMA,
            "generated_at": _isoformat(self._now_utc()),
            "projection_provenance": {
                "derivation": (
                    "server_computed_from_integrity_checked_ledger_and_"
                    "reconciled_artifacts"
                ),
                "caller_claims_preserve_epistemic_status": True,
                "request_provenance_verifies_claim_truth": False,
                "signed_review_artifacts_reconciled_where_present": True,
                "authority_granted": False,
            },
            "proposal_count": listed["count"],
            "item_count": len(items),
            "active_count": len(active),
            "total_item_count": total_item_count,
            "total_active_count": total_active_count,
            "unexamined_candidate_count": unexamined_candidate_count,
            "truncated": (
                total_item_count > len(items) or unexamined_candidate_count > 0
            ),
            "counts": {
                priority: sum(item["priority"] == priority for item in active)
                for priority in _ATTENTION_PRIORITY_ORDER
            },
            "items": items,
            "acknowledgement_is_approval": False,
            "authority_granted": False,
        }
