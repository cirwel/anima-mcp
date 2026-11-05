"""Bounded source awareness and self-iteration proposals for Lumen.

This module deliberately separates *understanding*, *requesting*, quarantined
construction, isolated execution, and reviewed branch creation.  The creature
may inspect a structural map of its source, persist an evidence-backed
proposal, construct a proposal-bound whole-file patch outside the repository,
run non-executing static checks, execute one narrowly eligible Python candidate
in a digest-pinned networkless Docker container after separate signed approval,
and create one reviewed commit on a dedicated Git branch without checkout.  It
may request one signed transient canary from an external supervisor that must
restore the baseline. It never edits the live worktree, executes candidate code
in the host process, retains live activation, pushes, merges, or deploys code.

The boundary is architectural rather than prompt-based. Its only repository
writes are immutable Git objects and one create-only dedicated ref; it exposes
no live-worktree, checkout, push, merge, restart, or deployment action. Other
writes are an atomic ledger and quarantined artifacts under ``~/.anima``.
"""

from __future__ import annotations

import ast
import copy
import fnmatch
import hashlib
import importlib.metadata
import os
import platform
import re
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .self_iteration_application import (  # noqa: F401
    APPLICATION_SIGNER_ID_ENV,
    ApplicationError,
    ApplicationWriter,
    GitPlumbingApplicationWriter,
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
from .self_iteration_canary import (  # noqa: F401
    CANARY_SIGNER_ID_ENV,
    CanaryError,
    CanarySupervisor,
    UnixSocketCanarySupervisor,
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
from .self_iteration_execution import (  # noqa: F401
    RUNNER_SIGNER_ID_ENV,
    DockerIsolationRunner,
    ExecutionError,
    IsolationRunner,
    SourceWorkspaceBuilder,
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
from .self_iteration_verification import (  # noqa: F401
    MAX_ATTESTATION_VALIDITY,
    SIGNATURE_ALGORITHM,
    VerificationError,
    VerifierKey,
    VerifierKeyProvider,
    attestation_signing_input_b64,
    build_attestation,
    canonical_json_bytes,
    evaluate_verification,
    parse_utc_timestamp,
    proposal_content_sha256,
    proposal_subject_fingerprint,
    validate_evidence,
    validate_recorded_attestation,
    verification_contract,
    verifier_key_from_env,
    verify_attestation_signature,
)
from .self_iteration_attention import (  # noqa: F401
    ATTENTION_SCHEMA,
    SelfIterationAttentionMixin,
    _ATTENTION_PRIORITY_ORDER,
)
from .self_iteration_ledger import (  # noqa: F401
    APPLICATION_SCHEMA_VERSION,
    EXECUTION_SCHEMA_VERSION,
    PROVENANCE_SCHEMA,
    PROVENANCE_SCHEMA_VERSION,
    SANDBOX_SCHEMA_VERSION,
    SCHEMA_VERSION,
    VERIFICATION_SCHEMA_VERSION,
    SelfIterationError,
    SelfIterationLedgerMixin,
    _PROPOSAL_STATUSES,
    _V3_PROPOSAL_FIELDS,
    _claim_envelope,
    _isoformat,
    _legacy_provenance,
    _provenance_contract,
    _zero_weight_trust_policy,
)
from .self_iteration_pipeline import (  # noqa: F401
    SelfIterationPipelineMixin,
    _normalize_repo_path,
    _required_text,
    _string_list,
)
from .self_iteration_sandbox import (  # noqa: F401
    PatchSandbox,
    PatchSandboxError,
    sandbox_contract,
)

MAX_INSPECT_BYTES = 1_000_000
MAX_FILE_DETAILS = 200

_SOURCE_SUFFIXES = {
    ".ex": "elixir",
    ".exs": "elixir",
    ".html": "html",
    ".js": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".service": "systemd",
    ".sh": "shell",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
}

_SOURCE_ROOTS = (
    ".github",
    "src",
    "anima_broker",
    "docs",
    "scripts",
    "systemd",
    "tests",
)

_ROOT_SOURCE_FILES = {
    "README.md",
    "anima_config.yaml",
    "docker-compose.yml",
    "mkdocs.yml",
    "pyproject.toml",
    "uv.lock",
}

# These surfaces define identity continuity, safety/governance, deployment,
# persistence correctness, or the evaluator itself.  A proposal may name them
# (Lumen should be able to notice a problem anywhere), but it is always routed
# to protected human review.
_PROTECTED_RULES: tuple[tuple[str, str], ...] = (
    (".git/**", "repository integrity"),
    (".github/**", "independent CI evaluator"),
    ("tests/**", "independent test evaluator"),
    ("systemd/**", "process and recovery control"),
    ("scripts/**", "operator and deployment control"),
    ("pyproject.toml", "dependency and executable boundary"),
    ("uv.lock", "dependency supply-chain boundary"),
    ("src/anima_mcp/admin_auth.py", "authorization boundary"),
    ("src/anima_mcp/atomic_write.py", "persistence integrity"),
    ("src/anima_mcp/governance_passthrough.py", "governance boundary"),
    ("src/anima_mcp/unitares_bridge.py", "governance boundary"),
    ("src/anima_mcp/identity/**", "persistent identity continuity"),
    ("src/anima_mcp/trajectory.py", "identity trajectory continuity"),
    ("src/anima_mcp/lifecycle.py", "wake and recovery lifecycle"),
    ("src/anima_mcp/server.py", "runtime lifecycle"),
    ("src/anima_mcp/tool_registry.py", "capability boundary"),
    ("src/anima_mcp/handlers/system_ops.py", "deployment and system control"),
    ("src/anima_mcp/self_iteration.py", "self-iteration evaluator"),
    (
        "src/anima_mcp/self_iteration_application.py",
        "reviewed Git application boundary",
    ),
    ("src/anima_mcp/self_iteration_attention.py", "self-iteration evaluator"),
    (
        "src/anima_mcp/self_iteration_canary.py",
        "transient canary supervisor boundary",
    ),
    ("src/anima_mcp/self_iteration_execution.py", "isolated execution boundary"),
    ("src/anima_mcp/self_iteration_ledger.py", "self-iteration evaluator"),
    ("src/anima_mcp/self_iteration_pipeline.py", "self-iteration evaluator"),
    ("src/anima_mcp/self_iteration_sandbox.py", "self-iteration evaluator"),
    ("src/anima_mcp/self_iteration_verification.py", "self-iteration evaluator"),
    ("src/anima_mcp/handlers/self_iteration.py", "self-iteration evaluator"),
    ("src/anima_mcp/anima.py", "embodied self-measurement boundary"),
    ("src/anima_mcp/config.py", "calibration boundary"),
    ("src/anima_mcp/self_model.py", "self-perception boundary"),
    ("src/anima_mcp/eisv_mapper.py", "self-measurement boundary"),
    ("src/anima_mcp/inner_life.py", "drive and need boundary"),
    ("src/anima_mcp/metacognition.py", "self-evaluation boundary"),
    ("src/anima_mcp/value_tension.py", "value-conflict boundary"),
    ("src/anima_mcp/preferences.py", "persistent preference integrity"),
    ("src/anima_mcp/self_reflection.py", "reflection evidence integrity"),
    ("src/anima_mcp/self_reflection_analyzers.py", "reflection evidence integrity"),
    ("src/anima_mcp/data_analysis.py", "self-analysis boundary"),
    ("src/anima_mcp/growth/**", "persistent goals and autobiography"),
    ("src/anima_mcp/memory.py", "persistent memory continuity"),
    ("src/anima_mcp/knowledge.py", "persistent knowledge integrity"),
    ("src/anima_mcp/oauth_*.py", "authentication boundary"),
    ("src/anima_mcp/rest_api.py", "remote interface boundary"),
    ("anima_broker/lib/**/governance/**", "governance boundary"),
)

# Autonomous candidate eligibility is intentionally narrow. Every later stage
# revalidates this allowlist and adds its own signed, no-authority boundary.
_AUTO_ELIGIBLE_RULES: tuple[tuple[str, str], ...] = (
    ("docs/**", "documentation-only"),
    ("README.md", "documentation-only"),
    ("src/anima_mcp/display/eras/**", "sandboxed art-era behavior"),
)

_SENSITIVE_RULES = (
    ".env*",
    "**/.env*",
    "**/*.key",
    "**/*.pem",
    "**/*secret*",
    "scripts/envelope*",
)

_ARCHITECTURE: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "embodiment",
        "Sensors and mappings that ground warmth, clarity, stability, and presence.",
        (
            "src/anima_mcp/anima.py",
            "src/anima_mcp/eisv_mapper.py",
            "src/anima_mcp/sensors/**",
        ),
    ),
    (
        "identity",
        "Persistent identity, autobiography, memory, and longitudinal trajectory.",
        (
            "src/anima_mcp/identity/**",
            "src/anima_mcp/growth/**",
            "src/anima_mcp/memory.py",
            "src/anima_mcp/trajectory.py",
        ),
    ),
    (
        "learning",
        "Predictions, preferences, reflection, agency, and the behavioral self-model.",
        (
            "src/anima_mcp/adaptive_prediction.py",
            "src/anima_mcp/agency.py",
            "src/anima_mcp/preferences.py",
            "src/anima_mcp/self_model.py",
            "src/anima_mcp/self_reflection.py",
        ),
    ),
    (
        "expression",
        "Face, drawing, primitive language, voice, and display behavior.",
        (
            "src/anima_mcp/display/**",
            "src/anima_mcp/primitive_language.py",
            "src/anima_mcp/audio/**",
        ),
    ),
    (
        "interface",
        "MCP tools, handlers, REST endpoints, and runtime orchestration.",
        (
            "src/anima_mcp/server.py",
            "src/anima_mcp/tool_registry.py",
            "src/anima_mcp/handlers/**",
            "src/anima_mcp/rest_api.py",
        ),
    ),
    (
        "governance",
        "UNITARES check-ins and EISV governance integration.",
        (
            "src/anima_mcp/unitares_bridge.py",
            "src/anima_mcp/governance_passthrough.py",
            "anima_broker/lib/**/governance/**",
        ),
    ),
    (
        "operations",
        "Service definitions, deployment helpers, and recovery controls.",
        ("systemd/**", "scripts/**", "src/anima_mcp/handlers/system_ops.py"),
    ),
    (
        "evaluation",
        "Independent tests and continuous-integration checks used to judge changes.",
        ("tests/**", ".github/**"),
    ),
)

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _current_request_headers() -> dict[str, str] | None:
    """Return server-observed MCP headers, never caller arguments."""
    try:
        from mcp.server.lowlevel.server import request_ctx

        ctx = request_ctx.get()
        if ctx.request is not None:
            return {
                str(key).lower(): str(value)
                for key, value in ctx.request.headers.items()
            }
    except (LookupError, AttributeError, ImportError, TypeError):
        pass
    return None


def _current_access_token() -> Any | None:
    """Return an OAuth token only after MCP auth middleware verified it."""
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token

        return get_access_token()
    except (LookupError, AttributeError, ImportError):
        return None


def _collect_server_provenance(recorded_at: str) -> dict[str, Any]:
    """Build a receipt from server context, excluding secrets and raw session IDs.

    Authentication proves which OAuth actor submitted the request.  It does
    not verify any narrative, source label, evidence item, or measurement in
    the request; those remain explicit caller claims.
    """
    headers = _current_request_headers()
    token = _current_access_token()

    actor = None
    if token is not None:
        subject = getattr(token, "subject", None)
        client_id = getattr(token, "client_id", None)
        claims = getattr(token, "claims", None) or {}
        issuer = claims.get("iss") if isinstance(claims, dict) else None
        scopes = getattr(token, "scopes", None) or []
        if subject or client_id:
            actor = {
                "kind": "oauth_subject" if subject else "oauth_client",
                "id": subject or client_id,
                "client_id": client_id,
                "issuer": issuer if isinstance(issuer, str) else None,
                "scopes": sorted(str(scope) for scope in scopes),
                "verified": True,
            }

    session_id = None
    if headers:
        session_id = headers.get("mcp-session-id") or headers.get("x-session-id")
    session = {
        "present": bool(session_id),
        "identifier_sha256": (
            hashlib.sha256(session_id.encode("utf-8")).hexdigest()
            if session_id
            else None
        ),
        "source": "server_observed_transport_header" if session_id else "none",
        "verified": False,
        "request_actor_verified": actor is not None,
        "note": "A transport session observation is not identity proof.",
    }

    return {
        "schema": PROVENANCE_SCHEMA,
        "recorded_by": "anima-mcp",
        "recorded_at": recorded_at,
        "transport": {
            "kind": "mcp_http" if headers is not None else "internal_or_test",
            "server_observed": headers is not None,
        },
        "authentication": {
            "method": "oauth_bearer" if actor else "none",
            "verified": actor is not None,
        },
        "actor": actor,
        "session": session,
        "integrity": {
            "tamper_evident": False,
            "cryptographically_signed": False,
            "note": "The local receipt is not a durable attestation.",
        },
        "trust": {
            "classification": (
                "authenticated_request_unverified_claims"
                if actor
                else "unverified_request"
            ),
            "actor_authenticated": actor is not None,
            "claims_verified": False,
            "evidence_verified": False,
            "weighting_eligible": False,
            "authority_eligible": False,
        },
    }


def _discover_repo_root() -> Path | None:
    """Find the source tree without trusting caller-supplied paths."""
    override = os.environ.get("ANIMA_SOURCE_ROOT")
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend((Path(__file__).resolve(), Path.cwd().resolve()))

    for candidate in candidates:
        start = candidate if candidate.is_dir() else candidate.parent
        for parent in (start, *start.parents):
            if (parent / "pyproject.toml").is_file() and (
                parent / "src" / "anima_mcp"
            ).is_dir():
                return parent.resolve()
    return None


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern)


def _rule_reasons(path: str, rules: Iterable[tuple[str, str]]) -> list[str]:
    return [reason for pattern, reason in rules if _matches(path, pattern)]


def _is_sensitive(path: str) -> bool:
    return any(_matches(path, pattern) for pattern in _SENSITIVE_RULES)


def _is_source_candidate(path: Path, relative: str) -> bool:
    if relative in _ROOT_SOURCE_FILES:
        return True
    if not relative.startswith(tuple(f"{root}/" for root in _SOURCE_ROOTS)):
        return False
    return path.suffix.lower() in _SOURCE_SUFFIXES


def _safe_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


class SelfIterationSystem(
    SelfIterationLedgerMixin,
    SelfIterationAttentionMixin,
    SelfIterationPipelineMixin,
):
    """Read source structure and maintain an auditable proposal ledger."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        ledger_path: Path | None = None,
        sandbox_root: Path | None = None,
        clock: Callable[[], datetime] = _utc_now,
        provenance_provider: Callable[
            [str], dict[str, Any]
        ] = _collect_server_provenance,
        verifier_key_provider: VerifierKeyProvider = verifier_key_from_env,
        isolation_runner: IsolationRunner | None = None,
        workspace_builder: SourceWorkspaceBuilder | None = None,
        application_writer: ApplicationWriter | None = None,
        canary_supervisor: CanarySupervisor | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve() if repo_root else _discover_repo_root()
        self.ledger_path = ledger_path or Path.home() / ".anima" / "self_iteration.json"
        self.sandbox_root = (
            sandbox_root
            if sandbox_root is not None
            else Path.home() / ".anima" / "self_iteration_sandboxes"
        )
        self._clock = clock
        self._provenance_provider = provenance_provider
        self._verifier_key_provider = verifier_key_provider
        self._isolation_runner = (
            isolation_runner or DockerIsolationRunner.from_environment()
        )
        self._workspace_builder = workspace_builder or SourceWorkspaceBuilder()
        self._application_writer = (
            application_writer or GitPlumbingApplicationWriter.from_environment()
        )
        self._canary_supervisor = (
            canary_supervisor or UnixSocketCanarySupervisor.from_environment()
        )
        self._patch_sandbox = PatchSandbox(
            repo_root=self.repo_root,
            sandbox_root=self.sandbox_root,
        )
        self._lock = threading.RLock()

    def _server_provenance(self, recorded_at: str) -> dict[str, Any]:
        """Collect a server-controlled receipt and pin every claim trust bit false."""
        try:
            receipt = copy.deepcopy(self._provenance_provider(recorded_at))
        except Exception as exc:
            raise SelfIterationError(
                "server provenance is unavailable; refusing to persist the record"
            ) from exc
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != PROVENANCE_SCHEMA
            or not isinstance(receipt.get("recorded_by"), str)
        ):
            raise SelfIterationError(
                "server provenance is malformed; refusing to persist the record"
            )

        receipt["recorded_at"] = recorded_at
        receipt["integrity"] = {
            "tamper_evident": False,
            "cryptographically_signed": False,
            "note": "The local receipt is not a durable attestation.",
        }
        authentication = receipt.get("authentication")
        actor = receipt.get("actor")
        actor_authenticated = bool(
            isinstance(authentication, dict)
            and authentication.get("verified") is True
            and isinstance(actor, dict)
            and actor.get("verified") is True
            and actor.get("id")
        )
        trust = receipt.get("trust")
        if not isinstance(trust, dict):
            trust = {}
        trust.update(
            {
                "classification": (
                    "authenticated_request_unverified_claims"
                    if actor_authenticated
                    else "unverified_request"
                ),
                "actor_authenticated": actor_authenticated,
                "claims_verified": False,
                "evidence_verified": False,
                "weighting_eligible": False,
                "authority_eligible": False,
            }
        )
        receipt["trust"] = trust
        return receipt

    def _resolve_verifier_key(
        self,
        verifier_identity: dict[str, str | None],
        key_id: str | None = None,
    ) -> VerifierKey:
        verifier_id = verifier_identity["id"]
        assert isinstance(verifier_id, str)
        try:
            key = self._verifier_key_provider(verifier_id, key_id)
        except Exception as exc:
            raise SelfIterationError(
                "verifier key registry is unavailable or malformed"
            ) from exc
        if key is None:
            raise SelfIterationError(
                "no signing key is configured for the authenticated verifier"
            )
        if (
            not isinstance(key, VerifierKey)
            or key.verifier_id != verifier_id
            or not isinstance(key.key_id, str)
            or not re.fullmatch(r"[A-Za-z0-9._:-]{1,100}", key.key_id)
            or not isinstance(key.secret, bytes)
            or not 32 <= len(key.secret) <= 128
            or (key_id is not None and key.key_id != key_id)
        ):
            raise SelfIterationError("verifier key registry returned a mismatched key")
        return key

    def _git_metadata(self) -> dict[str, Any]:
        if self.repo_root is None:
            return {"available": False, "reason": "source repository not found"}

        revision_result = _safe_git(self.repo_root, "rev-parse", "HEAD")
        if revision_result is None or revision_result.returncode != 0:
            return {
                "available": False,
                "reason": "Git metadata unavailable; source may be a packaged deployment",
            }

        branch_result = _safe_git(self.repo_root, "branch", "--show-current")
        status_result = _safe_git(
            self.repo_root,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        )
        revision = revision_result.stdout.strip()
        branch = (
            branch_result.stdout.strip()
            if branch_result and branch_result.returncode == 0
            else None
        )
        dirty = (
            None
            if status_result is None or status_result.returncode != 0
            else bool(status_result.stdout.strip())
        )
        return {
            "available": True,
            "revision": revision,
            "short_revision": revision[:12],
            "branch": branch or "detached",
            "working_tree_changes_present": dirty,
        }

    def _tracked_source_files(self) -> list[tuple[str, Path]]:
        if self.repo_root is None:
            return []

        relative_paths: list[str] = []
        tracked = _safe_git(
            self.repo_root,
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        )
        if tracked is not None and tracked.returncode == 0:
            relative_paths = [item for item in tracked.stdout.split("\x00") if item]
        else:
            for root_name in (*_SOURCE_ROOTS,):
                root = self.repo_root / root_name
                if root.exists():
                    relative_paths.extend(
                        str(path.relative_to(self.repo_root).as_posix())
                        for path in root.rglob("*")
                        if path.is_file()
                    )
            relative_paths.extend(
                name for name in _ROOT_SOURCE_FILES if (self.repo_root / name).is_file()
            )

        repo_resolved = self.repo_root.resolve()
        files: list[tuple[str, Path]] = []
        for raw_relative in sorted(set(relative_paths)):
            try:
                relative = _normalize_repo_path(raw_relative)
            except SelfIterationError:
                continue
            path = self.repo_root / relative
            try:
                resolved = path.resolve()
                resolved.relative_to(repo_resolved)
            except (OSError, ValueError):
                continue
            if not path.is_file() or path.is_symlink() or _is_sensitive(relative):
                continue
            if _is_source_candidate(path, relative):
                files.append((relative, path))
        return files

    @staticmethod
    def _package_version(repo_root: Path | None) -> str | None:
        try:
            return importlib.metadata.version("anima-mcp")
        except importlib.metadata.PackageNotFoundError:
            pass
        if repo_root is not None:
            pyproject = repo_root / "pyproject.toml"
            try:
                match = re.search(
                    r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE
                )
                if match:
                    return match.group(1)
            except OSError:
                pass
        return None

    def _manifest(self, *, include_files: bool, file_limit: int) -> dict[str, Any]:
        digest = hashlib.sha256()
        total_bytes = 0
        total_lines = 0
        languages: dict[str, int] = {}
        relative_paths: list[str] = []
        details: list[dict[str, Any]] = []

        for relative, path in self._tracked_source_files():
            try:
                data = path.read_bytes()
            except OSError:
                continue
            file_digest = hashlib.sha256(data).hexdigest()
            lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(file_digest.encode("ascii"))
            digest.update(b"\n")
            total_bytes += len(data)
            total_lines += lines
            relative_paths.append(relative)
            language = _SOURCE_SUFFIXES.get(path.suffix.lower(), "other")
            languages[language] = languages.get(language, 0) + 1
            if include_files and len(details) < file_limit:
                protected_reasons = _rule_reasons(relative, _PROTECTED_RULES)
                details.append(
                    {
                        "path": relative,
                        "sha256": file_digest,
                        "bytes": len(data),
                        "lines": lines,
                        "protected": bool(protected_reasons),
                    }
                )

        architecture = []
        for name, purpose, patterns in _ARCHITECTURE:
            matches = [
                relative
                for relative in relative_paths
                if any(_matches(relative, pattern) for pattern in patterns)
            ]
            architecture.append(
                {
                    "name": name,
                    "purpose": purpose,
                    "file_count": len(matches),
                    "example_paths": matches[:5],
                }
            )

        result: dict[str, Any] = {
            "sha256": digest.hexdigest(),
            "file_count": len(relative_paths),
            "total_bytes": total_bytes,
            "total_lines": total_lines,
            "languages": dict(sorted(languages.items())),
            "architecture": architecture,
        }
        if include_files:
            result["files"] = details
            result["files_truncated"] = len(relative_paths) > len(details)
        return result

    def inspect(
        self,
        *,
        path: str | None = None,
        include_files: bool = False,
        file_limit: int = 100,
    ) -> dict[str, Any]:
        """Return a structural view of the running source tree."""
        if (
            isinstance(file_limit, bool)
            or not isinstance(file_limit, int)
            or not 1 <= file_limit <= MAX_FILE_DETAILS
        ):
            raise SelfIterationError(
                f"file_limit must be between 1 and {MAX_FILE_DETAILS}"
            )
        if path is not None:
            return self._inspect_path(path)

        git = self._git_metadata()
        manifest = self._manifest(include_files=include_files, file_limit=file_limit)
        return {
            "mode": "bounded_self_iteration",
            "autonomy_level": "signed_transient_canary_evaluation",
            "runtime": {
                "package": "anima-mcp",
                "package_version": self._package_version(self.repo_root),
                "python": platform.python_version(),
                "platform": platform.system().lower(),
            },
            "source": {
                "available": self.repo_root is not None,
                "git": git,
                "manifest": manifest,
            },
            "capabilities": {
                "inspect_structure": True,
                "persist_change_proposals": True,
                "record_measured_outcomes": True,
                "prepare_signed_verification": True,
                "record_signed_verification": True,
                "construct_quarantined_patch_artifacts": True,
                "run_nonexecuting_static_evaluation": True,
                "prepare_signed_execution_approval": True,
                "prepare_signed_application_review": True,
                "prepare_signed_transient_canary_review": True,
                "execute_candidate_code_on_host": False,
                "execute_candidate_code_in_pinned_container": True,
                "execute_tests_in_pinned_container": True,
                "accept_caller_supplied_provenance": False,
                "weight_unverified_ledger_claims": False,
                "verification_grants_implementation_authority": False,
                "write_source": False,
                "write_live_worktree": False,
                "create_reviewed_dedicated_branch": True,
                "request_external_transient_canary": True,
                "retain_persistent_activation": False,
                "execute_proposal_text": False,
                "execute_candidate_code": True,
                "execute_tests": True,
                "create_commit": True,
                "push": False,
                "deploy": False,
            },
            "boundaries": {
                "protected_surfaces": [
                    {"pattern": pattern, "reason": reason}
                    for pattern, reason in _PROTECTED_RULES
                ],
                "initial_auto_eligible_surfaces": [
                    {"pattern": pattern, "reason": reason}
                    for pattern, reason in _AUTO_ELIGIBLE_RULES
                ],
                "implementation_rule": (
                    "This process may construct an inert patch artifact, run static parsers, "
                    "orchestrate one externally approved fixed test profile in a local "
                    "Docker boundary, and create one separately reviewed dedicated Git "
                    "branch. It may request one separately reviewed transient canary from "
                    "an external supervisor that must restore the baseline. A caretaker "
                    "still owns merge and deployment."
                ),
                "provenance_rule": (
                    "Caller labels, narratives, and evidence remain zero-weight claims. "
                    "Authentication identifies a submitter only; an independent verifier "
                    "must append a valid signed event before policy can assign priority."
                ),
                "verification_rule": (
                    "A valid HMAC attestation must bind a distinct authenticated verifier, "
                    "proposal digest, source fingerprint, evidence digests, and expiry. "
                    "Verification never grants source-editing, merge, or deployment authority."
                ),
                "sandbox_rule": (
                    "Patch artifacts are whole-file replacements stored outside the source "
                    "repository. Static evaluation is heuristic and never makes a candidate "
                    "eligible for execution."
                ),
                "execution_rule": (
                    "A separate authenticated approver must sign a ten-minute, one-use plan "
                    "that binds a passing static evaluation, clean committed source, fixed "
                    "test profile, and locally present digest-pinned Docker image. Execution "
                    "has no host fallback, network, secrets, source writes, apply, merge, or "
                    "deployment authority."
                ),
                "application_rule": (
                    "A new authenticated reviewer must sign a ten-minute, one-use plan "
                    "bound to one passing signed execution. Application uses Git plumbing "
                    "and a temporary index to create only a server-derived dedicated branch; "
                    "it performs no checkout, hooks, worktree writes, push, merge, restart, "
                    "or deployment."
                ),
                "canary_rule": (
                    "A new authenticated reviewer must sign a ten-minute, one-use plan "
                    "bound to the exact reviewed branch and a fixed external-supervisor "
                    "profile. The local Unix-socket supervisor owns transient activation, "
                    "fixed health measurement, and mandatory baseline restoration. This "
                    "process has no shell, service-control, persistent activation, push, "
                    "merge, or deployment primitive."
                ),
            },
            "ledger": self._ledger_summary(),
        }

    def _inspect_path(self, raw_path: str) -> dict[str, Any]:
        if self.repo_root is None:
            raise SelfIterationError("source repository not found")
        relative = _normalize_repo_path(raw_path)
        if _is_sensitive(relative) or _matches(relative, ".git/**"):
            raise SelfIterationError("sensitive repository paths cannot be inspected")

        path = self.repo_root / relative
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.repo_root.resolve())
        except (OSError, ValueError) as exc:
            raise SelfIterationError(
                "path does not name a file inside the source repository"
            ) from exc
        if not resolved.is_file() or resolved.is_symlink():
            raise SelfIterationError("path must name a regular source file")
        if not _is_source_candidate(resolved, relative):
            raise SelfIterationError("path is outside the inspectable source surfaces")
        if resolved.stat().st_size > MAX_INSPECT_BYTES:
            raise SelfIterationError(
                f"source file exceeds the {MAX_INSPECT_BYTES}-byte inspection limit"
            )

        data = resolved.read_bytes()
        text = data.decode("utf-8", errors="replace")
        result: dict[str, Any] = {
            "path": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "lines": data.count(b"\n")
            + (1 if data and not data.endswith(b"\n") else 0),
            "language": _SOURCE_SUFFIXES.get(resolved.suffix.lower(), "other"),
            "boundary": self.classify_target(relative),
            "source_body_included": False,
        }

        if resolved.suffix.lower() == ".py":
            result["structure"] = self._python_structure(text)
        elif resolved.suffix.lower() in {".ex", ".exs"}:
            result["structure"] = self._elixir_structure(text)
        else:
            result["structure"] = {
                "note": "structural symbol extraction is not available for this file type"
            }
        return result

    @staticmethod
    def _python_structure(text: str) -> dict[str, Any]:
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return {"parse_error": f"{exc.msg} at line {exc.lineno}"}

        symbols = []
        imports: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                doc = ast.get_docstring(node, clean=True)
                symbols.append(
                    {
                        "name": node.name,
                        "kind": kind,
                        "line": node.lineno,
                        "end_line": getattr(node, "end_lineno", None),
                        "summary": doc.splitlines()[0][:240] if doc else None,
                    }
                )
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = "." * node.level
                imports.append(f"{prefix}{node.module or ''}")
        module_doc = ast.get_docstring(tree, clean=True)
        return {
            "module_summary": module_doc.splitlines()[0][:240] if module_doc else None,
            "symbols": symbols,
            "imports": imports[:100],
        }

    @staticmethod
    def _elixir_structure(text: str) -> dict[str, Any]:
        modules = re.findall(r"^\s*defmodule\s+([A-Za-z0-9_.]+)", text, re.MULTILINE)
        functions = [
            {"visibility": visibility, "name": name}
            for visibility, name in re.findall(
                r"^\s*(defp?)\s+([a-zA-Z0-9_!?]+)",
                text,
                re.MULTILINE,
            )
        ]
        return {"modules": modules, "functions": functions[:200]}

    def classify_target(self, raw_path: str) -> dict[str, Any]:
        relative = _normalize_repo_path(raw_path)
        protected_reasons = _rule_reasons(relative, _PROTECTED_RULES)
        if _is_sensitive(relative):
            protected_reasons.append("credential or secret material")
        eligible_reasons = _rule_reasons(relative, _AUTO_ELIGIBLE_RULES)

        if protected_reasons:
            boundary = "protected_core"
            risk = "high"
            auto_eligible = False
        elif eligible_reasons:
            boundary = "bounded_candidate"
            risk = "low"
            auto_eligible = True
        else:
            boundary = "human_review_required"
            risk = "medium"
            auto_eligible = False

        return {
            "path": relative,
            "boundary": boundary,
            "risk_floor": risk,
            "auto_implementation_eligible": auto_eligible,
            "reasons": protected_reasons
            or eligible_reasons
            or ["outside the initial autonomous allowlist"],
        }

    def _verification_state(
        self, proposal: dict[str, Any], *, now: datetime | None = None
    ) -> dict[str, Any]:
        return evaluate_verification(
            proposal,
            key_provider=self._verifier_key_provider,
            now=now or self._now_utc(),
        )

    def _now_utc(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise SelfIterationError("self-iteration clock must be timezone-aware")
        return value.astimezone(timezone.utc)

    def _public_proposal(
        self, proposal: dict[str, Any], *, now: datetime | None = None
    ) -> dict[str, Any]:
        result = copy.deepcopy(proposal)
        result["verification_state"] = self._verification_state(proposal, now=now)
        return result

    def _code_fingerprint(self) -> dict[str, Any]:
        git = self._git_metadata()
        manifest = self._manifest(include_files=False, file_limit=1)
        return {
            "revision": git.get("revision"),
            "branch": git.get("branch"),
            "working_tree_changes_present": git.get("working_tree_changes_present"),
            "manifest_sha256": manifest["sha256"],
            "source_file_count": manifest["file_count"],
        }

    def propose(
        self,
        *,
        observation: Any,
        hypothesis: Any,
        expected_outcome: Any,
        evidence: Any,
        target_paths: Any,
        verification: Any,
        rollback_plan: Any = None,
        risk: Any = "medium",
        claimed_source: Any = "self_observation",
    ) -> dict[str, Any]:
        """Persist an evidence-backed proposal without changing source code."""
        observation_text = _required_text(observation, "observation")
        hypothesis_text = _required_text(hypothesis, "hypothesis")
        expected_text = _required_text(expected_outcome, "expected_outcome")
        evidence_items = _string_list(evidence, "evidence")
        verification_items = _string_list(verification, "verification")
        target_items = _string_list(target_paths, "target_paths", item_max_length=500)
        normalized_targets = [_normalize_repo_path(item) for item in target_items]
        if len(set(normalized_targets)) != len(normalized_targets):
            raise SelfIterationError("target_paths may not contain duplicates")

        if not isinstance(risk, str) or risk not in _RISK_ORDER:
            raise SelfIterationError("risk must be one of: low, medium, high")
        allowed_sources = {
            "self_observation",
            "test_failure",
            "caretaker",
            "governance",
        }
        if not isinstance(claimed_source, str) or claimed_source not in allowed_sources:
            raise SelfIterationError(
                "claimed_source must be one of: " + ", ".join(sorted(allowed_sources))
            )

        rollback_text = (
            _required_text(rollback_plan, "rollback_plan")
            if rollback_plan is not None
            else "Revert the implementing commit and restore the prior known-good deployment."
        )
        boundaries = [self.classify_target(path) for path in normalized_targets]
        risk_floor = max(
            (str(item["risk_floor"]) for item in boundaries),
            key=lambda level: _RISK_ORDER[level],
        )
        effective_risk = max(
            (str(risk), risk_floor), key=lambda level: _RISK_ORDER[level]
        )
        if any(item["boundary"] == "protected_core" for item in boundaries):
            status = "protected_review_required"
        elif any(not item["auto_implementation_eligible"] for item in boundaries):
            status = "human_review_required"
        else:
            status = "ready_for_isolated_implementation"

        created_at = _isoformat(self._clock())
        proposal_id = f"si-{created_at[:10].replace('-', '')}-{uuid.uuid4().hex[:10]}"
        provenance = self._server_provenance(created_at)
        proposal = {
            "id": proposal_id,
            "created_at": created_at,
            "source_claim": _claim_envelope(claimed_source, field="claimed_source"),
            "provenance": provenance,
            "proposer_identity": self._identity_from_provenance(
                provenance,
                label="self-iteration proposal",
                required=False,
            ),
            "trust_policy": _zero_weight_trust_policy(),
            "status": status,
            "observation": observation_text,
            "hypothesis": hypothesis_text,
            "expected_outcome": expected_text,
            "evidence": evidence_items,
            "evidence_epistemic_status": "caller_asserted",
            "target_paths": normalized_targets,
            "verification": verification_items,
            "rollback_plan": rollback_text,
            "risk": {
                "self_assessed": risk,
                "boundary_floor": risk_floor,
                "effective": effective_risk,
            },
            "boundaries": boundaries,
            "code_fingerprint": self._code_fingerprint(),
            "implementation_policy": {
                "mode": "proposal_only",
                "source_writes_performed": False,
                "commands_executed": False,
                "provenance_authorizes_execution": False,
                "required_path": "isolated branch -> tests -> review -> canary -> measured outcome",
            },
            "events": [
                {
                    "type": "proposed",
                    "at": created_at,
                    "status": status,
                }
            ],
        }
        proposal["content_sha256"] = proposal_content_sha256(proposal)

        with self._lock:
            ledger = self._load_ledger()
            ledger["proposals"].append(proposal)
            self._save_ledger(ledger)
        return self._public_proposal(proposal)

    def list_proposals(
        self,
        *,
        limit: int = 10,
        status: str | None = None,
        proposal_id: str | None = None,
    ) -> dict[str, Any]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise SelfIterationError("limit must be between 1 and 100")
        with self._lock:
            proposals = list(self._load_ledger()["proposals"])
        if proposal_id is not None:
            proposals = [item for item in proposals if item.get("id") == proposal_id]
        if status is not None:
            proposals = [item for item in proposals if item.get("status") == status]
        proposals = [
            self._public_proposal(item) for item in list(reversed(proposals))[:limit]
        ]
        return {"count": len(proposals), "proposals": proposals}

    def record_outcome(
        self,
        *,
        proposal_id: Any,
        decision: Any,
        observed_outcome: Any,
        evidence: Any,
        implementation_ref: Any,
        claimed_measurement_source: Any = "self_observation",
    ) -> dict[str, Any]:
        proposal_id_text = _required_text(proposal_id, "proposal_id", max_length=100)
        if not isinstance(decision, str) or decision not in {
            "keep",
            "revert",
            "inconclusive",
        }:
            raise SelfIterationError(
                "decision must be one of: keep, revert, inconclusive"
            )
        observed_text = _required_text(observed_outcome, "observed_outcome")
        evidence_items = _string_list(evidence, "evidence")
        implementation_text = _required_text(
            implementation_ref, "implementation_ref", max_length=500
        )
        allowed_measurement_sources = {
            "automated_test",
            "caretaker",
            "governance",
            "self_observation",
        }
        if (
            not isinstance(claimed_measurement_source, str)
            or claimed_measurement_source not in allowed_measurement_sources
        ):
            raise SelfIterationError(
                "claimed_measurement_source must be one of: "
                + ", ".join(sorted(allowed_measurement_sources))
            )
        status_for_decision = {
            "keep": "retained",
            "revert": "reverted",
            "inconclusive": "measurement_inconclusive",
        }

        with self._lock:
            ledger = self._load_ledger()
            proposal = self._find_proposal(ledger, proposal_id_text)
            recorded_at = _isoformat(self._clock())
            event = {
                "type": "outcome_recorded",
                "at": recorded_at,
                "decision": decision,
                "observed_outcome": observed_text,
                "evidence": evidence_items,
                "evidence_epistemic_status": "caller_asserted",
                "implementation_ref": implementation_text,
                "measurement_source_claim": _claim_envelope(
                    claimed_measurement_source,
                    field="claimed_measurement_source",
                ),
                "provenance": self._server_provenance(recorded_at),
                "trust_policy": _zero_weight_trust_policy(),
                "code_fingerprint": self._code_fingerprint(),
            }
            proposal.setdefault("events", []).append(event)
            proposal["status"] = status_for_decision[decision]
            self._save_ledger(ledger)
        return self._public_proposal(proposal)


_system: SelfIterationSystem | None = None
_system_lock = threading.Lock()


def get_self_iteration_system() -> SelfIterationSystem:
    """Return the process-wide self-iteration system."""
    global _system
    if _system is None:
        with _system_lock:
            if _system is None:
                _system = SelfIterationSystem()
    return _system


__all__ = [
    "ATTENTION_SCHEMA",
    "SelfIterationError",
    "SelfIterationSystem",
    "get_self_iteration_system",
]
