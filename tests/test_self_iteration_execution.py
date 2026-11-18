"""Tests for externally approved, fail-closed isolated self-iteration."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from anima_mcp.self_iteration import (
    PROVENANCE_SCHEMA,
    SelfIterationError,
    SelfIterationSystem,
)
from anima_mcp.self_iteration_execution import (
    APPROVAL_VALIDITY,
    MAX_CAPTURE_BYTES,
    CapturedProcess,
    DISPLAY_ERA_TEST_PROFILE,
    DockerIsolationRunner,
    ExecutionError,
    SourceWorkspaceBuilder,
    _run_bounded_process,
    approval_signing_input_b64,
    build_execution_approval,
    build_signed_execution_result,
    execution_approval_sha256,
    isolation_contract_sha256,
    sign_execution_approval,
    validate_execution_approval,
    validate_signed_execution_result,
    verify_execution_approval_signature,
)
from anima_mcp.self_iteration_sandbox import MAX_EXECUTION_RESULT_BYTES
from anima_mcp.self_iteration_verification import (
    VerifierKey,
    sign_attestation,
)
from conftest import parse_result


@dataclass
class MutableClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: int) -> None:
        self.value += timedelta(**kwargs)


class KeyRegistry:
    def __init__(self, *keys: VerifierKey) -> None:
        self.keys = {(key.verifier_id, key.key_id): key for key in keys}
        self.active = {key.verifier_id: key.key_id for key in keys}

    def __call__(
        self, verifier_id: str, requested_key_id: str | None
    ) -> VerifierKey | None:
        key_id = requested_key_id or self.active.get(verifier_id)
        return self.keys.get((verifier_id, key_id))


def _identity(actor_id: str) -> dict[str, str]:
    return {
        "kind": "oauth_subject",
        "id": actor_id,
        "issuer": "https://issuer.example.test",
    }


def _provenance_provider(actor_id: str | None):
    def provider(recorded_at: str) -> dict:
        authenticated = actor_id is not None
        return {
            "schema": PROVENANCE_SCHEMA,
            "recorded_by": "anima-mcp-execution-test",
            "recorded_at": recorded_at,
            "transport": {"kind": "test", "server_observed": True},
            "authentication": {
                "method": "oauth_bearer" if authenticated else "none",
                "verified": authenticated,
            },
            "actor": (
                {**_identity(actor_id), "verified": True}
                if authenticated and actor_id is not None
                else None
            ),
            "session": {"present": False, "verified": False},
            "trust": {},
        }

    return provider


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit_repo(repo: Path) -> str:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Phase Four Test")
    _git(repo, "config", "user.email", "phase4@example.test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return _git(repo, "rev-parse", "HEAD")


def _runner_identity(marker: str = "b") -> dict:
    image = "example/anima-runner@sha256:" + "a" * 64
    return {
        "schema": "anima.self_iteration.docker_runner.v1",
        "backend": "docker",
        "image_reference": image,
        "image_id": "sha256:" + marker * 64,
        "repo_digest": image,
        "os": "linux",
        "architecture": "arm64",
        "declared_volumes": [],
        "healthcheck_policy": "disabled",
        "runner_contract_sha256": isolation_contract_sha256(),
    }


def _empty_stream() -> dict:
    return {
        "captured": "",
        "bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "truncated": False,
    }


def _approval(now: datetime) -> tuple[dict, VerifierKey, VerifierKey]:
    approver_key = VerifierKey("approver-1", "key-a", b"a" * 32)
    signer_key = VerifierKey("runner-signer", "key-r", b"r" * 32)
    approval = build_execution_approval(
        proposal_id="si-20260811-0123456789",
        proposal_content_sha256="1" * 64,
        source_fingerprint={
            "revision": "2" * 40,
            "manifest_sha256": "3" * 64,
        },
        active_attestation_ids=["sia-" + "4" * 32],
        candidate_id="sip-" + "5" * 32,
        candidate_sha256="6" * 64,
        evaluation_id="sie-" + "7" * 32,
        evaluation_sha256="8" * 64,
        approver_identity=_identity("approver-1"),
        approval_key_id=approver_key.key_id,
        runner_identity=_runner_identity(),
        profile=DISPLAY_ERA_TEST_PROFILE,
        source_snapshot={
            "revision": "2" * 40,
            "tracked_file_count": 4,
            "tracked_total_bytes": 200,
            "baseline_tree_sha256": "9" * 64,
            "candidate_tree_sha256": "a" * 64,
            "candidate_paths": [
                "src/anima_mcp/display/eras/example.py",
            ],
        },
        result_signer_id=signer_key.verifier_id,
        result_signer_key_id=signer_key.key_id,
        issued_at=now,
    )
    return approval, approver_key, signer_key


class TestExecutionArtifacts:
    def test_approval_signature_binds_every_execution_input(self):
        now = datetime(2026, 8, 11, 22, 0, tzinfo=timezone.utc)
        approval, approver_key, _signer_key = _approval(now)
        signature = sign_execution_approval(approval, approver_key)

        assert validate_execution_approval(approval) == approval
        assert (
            execution_approval_sha256(approval)
            == hashlib.sha256(
                json.dumps(
                    approval, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
            ).hexdigest()
        )
        assert verify_execution_approval_signature(approval, signature, approver_key)
        assert base64.urlsafe_b64decode(
            approval_signing_input_b64(approval)
        ).startswith(b"anima.self_iteration.execution_approval.v1\x00")
        assert (
            datetime.fromisoformat(
                approval["challenge_expires_at"].replace("Z", "+00:00")
            )
            - now
            == APPROVAL_VALIDITY
        )

        tampered = copy.deepcopy(approval)
        tampered["profile"]["arguments"].append("tests/attacker.py")
        with pytest.raises(ExecutionError, match="profile binding"):
            validate_execution_approval(tampered)
        assert not verify_execution_approval_signature(
            {**approval, "candidate_sha256": "f" * 64}, signature, approver_key
        )

    def test_signed_result_is_server_verifiable_and_never_grants_apply(self):
        now = datetime(2026, 8, 11, 22, 0, tzinfo=timezone.utc)
        approval, approver_key, signer_key = _approval(now)
        receipt = {
            "outcome": "passed",
            "container_started": True,
            "timed_out": False,
            "output_limit_exceeded": False,
            "exit_code": 0,
            "oom_killed": False,
            "duration_ms": 12,
            "stdout": _empty_stream(),
            "stderr": _empty_stream(),
            "cleanup_confirmed": True,
            "runner_identity": approval["runner_identity"],
            "isolation_contract_sha256": isolation_contract_sha256(),
        }
        result = build_signed_execution_result(
            approval=approval,
            approval_signature=sign_execution_approval(approval, approver_key),
            approval_key=approver_key,
            runner_receipt=receipt,
            started_at=now.isoformat(),
            finished_at=(now + timedelta(seconds=1)).isoformat(),
            signer_key=signer_key,
        )
        registry = KeyRegistry(approver_key, signer_key)

        assert validate_signed_execution_result(result, registry) == result
        assert result["eligible_for_external_review"] is True
        assert result["approval_signature"]["approver_id"] == "approver-1"
        assert result["eligible_for_apply"] is False
        assert result["authority_granted"] is False

        tampered = copy.deepcopy(result)
        tampered["eligible_for_apply"] = True
        with pytest.raises(ExecutionError):
            validate_signed_execution_result(tampered, registry)
        with pytest.raises(ExecutionError, match="invalid approval signature"):
            build_signed_execution_result(
                approval=approval,
                approval_signature="0" * 64,
                approval_key=approver_key,
                runner_receipt=receipt,
                started_at=now.isoformat(),
                finished_at=(now + timedelta(seconds=1)).isoformat(),
                signer_key=signer_key,
            )

    def test_binary_capture_fits_the_immutable_result_artifact_limit(self):
        now = datetime(2026, 8, 11, 22, 0, tzinfo=timezone.utc)
        approval, approver_key, signer_key = _approval(now)
        captured = "\x00" * MAX_CAPTURE_BYTES
        raw = captured.encode()
        stream = {
            "captured": captured,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "truncated": False,
        }
        result = build_signed_execution_result(
            approval=approval,
            approval_signature=sign_execution_approval(approval, approver_key),
            approval_key=approver_key,
            runner_receipt={
                "outcome": "passed",
                "container_started": True,
                "timed_out": False,
                "output_limit_exceeded": False,
                "exit_code": 0,
                "oom_killed": False,
                "duration_ms": 12,
                "stdout": stream,
                "stderr": stream,
                "cleanup_confirmed": True,
                "runner_identity": approval["runner_identity"],
                "isolation_contract_sha256": isolation_contract_sha256(),
            },
            started_at=now.isoformat(),
            finished_at=(now + timedelta(seconds=1)).isoformat(),
            signer_key=signer_key,
        )

        encoded = json.dumps(
            result, indent=2, ensure_ascii=False, allow_nan=False
        ).encode()
        assert len(encoded) <= MAX_EXECUTION_RESULT_BYTES


class TestDockerEnvelope:
    def test_create_command_has_fixed_fail_closed_isolation(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ANIMA_PHASE4_CANARY_SECRET", "never-forward-this")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = DockerIsolationRunner(
            image_reference=_runner_identity()["image_reference"],
            docker_socket="/tmp/test-docker.sock",
            docker_binary="/bin/echo",
            socket_check=False,
        )
        command = runner.build_create_command(
            workspace=workspace,
            profile=DISPLAY_ERA_TEST_PROFILE,
            image_identity=_runner_identity(),
            container_name="anima-si-" + "c" * 24,
        )

        for required in (
            "--pull=never",
            "--network=none",
            "--ipc=none",
            "--cgroupns=private",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges=true",
            "--user=65534:65534",
            "--log-driver=none",
            "--no-healthcheck",
            "--entrypoint=/usr/local/bin/python",
        ):
            assert required in command
        mount = next(item for item in command if item.startswith("--mount="))
        assert mount.endswith(",dst=/workspace,readonly")
        assert not any("docker.sock" in item for item in command[4:])
        assert not any("ANIMA_PHASE4_CANARY_SECRET" in item for item in command)
        assert command[-len(DISPLAY_ERA_TEST_PROFILE.arguments) - 1] == (
            _runner_identity()["image_reference"]
        )

    def test_runner_has_no_unconfigured_or_host_fallback(self):
        runner = DockerIsolationRunner(
            image_reference=None,
            docker_socket="/tmp/test-docker.sock",
            docker_binary="/bin/echo",
            socket_check=False,
        )
        with pytest.raises(ExecutionError, match="digest-pinned local image"):
            runner.probe()

    def test_image_declared_writable_volumes_are_refused(self):
        image = _runner_identity()["image_reference"]
        payload = json.dumps(
            [
                {
                    "Id": "sha256:" + "b" * 64,
                    "RepoDigests": [image],
                    "Os": "linux",
                    "Architecture": "arm64",
                    "Config": {"Volumes": {"/state": {}}},
                }
            ]
        )

        def process_runner(*_args, **_kwargs):
            return CapturedProcess(
                returncode=0,
                timed_out=False,
                output_limit_exceeded=False,
                stdout=payload,
                stderr="",
                stdout_bytes=len(payload.encode()),
                stderr_bytes=0,
                stdout_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                stderr_sha256=hashlib.sha256(b"").hexdigest(),
                stdout_truncated=False,
                stderr_truncated=False,
            )

        runner = DockerIsolationRunner(
            image_reference=image,
            docker_socket="/tmp/test-docker.sock",
            docker_binary="/bin/echo",
            socket_check=False,
            process_runner=process_runner,
        )
        with pytest.raises(ExecutionError, match="writable volumes"):
            runner.probe()

    def test_process_output_is_bounded_and_terminated(self):
        result = _run_bounded_process(
            [sys.executable, "-c", "import os; os.write(1, b'x' * 1048576)"],
            timeout=5,
            capture_limit=1024,
            output_limit=4096,
            environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )

        assert result.output_limit_exceeded is True
        assert result.returncode is None
        assert len(result.stdout.encode()) <= 1024
        assert result.stdout_bytes + result.stderr_bytes == 4096
        assert result.stdout_truncated is True
        assert result.stderr_truncated is True


class TestCommittedWorkspace:
    def test_snapshot_uses_git_blobs_and_materializes_candidate_read_only(
        self, tmp_path
    ):
        repo = tmp_path / "repo"
        target = repo / "src/anima_mcp/display/eras/example.py"
        target.parent.mkdir(parents=True)
        target.write_text("VALUE = 'base'\n")
        (repo / "README.md").write_text("# Fixture\n")
        revision = _commit_repo(repo)
        replacement = b"VALUE = 'candidate'\n"
        manifest = {
            "files": [
                {
                    "path": target.relative_to(repo).as_posix(),
                    "base_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "candidate_sha256": hashlib.sha256(replacement).hexdigest(),
                }
            ]
        }
        contents = {target.relative_to(repo).as_posix(): replacement}
        builder = SourceWorkspaceBuilder()
        snapshot = builder.fingerprint(
            repo_root=repo,
            expected_revision=revision,
            candidate_manifest=manifest,
            candidate_contents=contents,
        )

        with builder.materialize(
            repo_root=repo,
            expected_revision=revision,
            candidate_manifest=manifest,
            candidate_contents=contents,
            expected_snapshot=snapshot,
        ) as workspace:
            materialized = workspace / target.relative_to(repo)
            assert materialized.read_bytes() == replacement
            assert (workspace / "README.md").read_text() == "# Fixture\n"
            assert materialized.stat().st_mode & 0o222 == 0
            assert workspace.stat().st_mode & 0o222 == 0
        assert not workspace.exists()

        (repo / "untracked.txt").write_text("not approved\n")
        with pytest.raises(ExecutionError, match="clean source worktree"):
            builder.fingerprint(
                repo_root=repo,
                expected_revision=revision,
                candidate_manifest=manifest,
                candidate_contents=contents,
            )

    def test_committed_symlink_is_refused(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "regular.txt").write_text("regular\n")
        (repo / "linked.txt").symlink_to("regular.txt")
        revision = _commit_repo(repo)

        with pytest.raises(ExecutionError, match="regular files only"):
            SourceWorkspaceBuilder().fingerprint(
                repo_root=repo,
                expected_revision=revision,
                candidate_manifest={"files": []},
                candidate_contents={},
            )


class FakeIsolationRunner:
    def __init__(self) -> None:
        self.identity = _runner_identity()
        self.fail_run = False
        self.seen_candidate: bytes | None = None
        self.run_count = 0

    def probe(self) -> dict:
        return copy.deepcopy(self.identity)

    def run(self, *, workspace: Path, profile, expected_identity: dict) -> dict:
        self.run_count += 1
        if self.fail_run:
            raise ExecutionError("simulated isolated runner failure")
        assert expected_identity == self.identity
        assert profile == DISPLAY_ERA_TEST_PROFILE
        target = workspace / "src/anima_mcp/display/eras/test_era.py"
        self.seen_candidate = target.read_bytes()
        return {
            "outcome": "passed",
            "container_started": True,
            "timed_out": False,
            "output_limit_exceeded": False,
            "exit_code": 0,
            "oom_killed": False,
            "duration_ms": 5,
            "stdout": _empty_stream(),
            "stderr": _empty_stream(),
            "cleanup_confirmed": True,
            "runner_identity": copy.deepcopy(self.identity),
            "isolation_contract_sha256": isolation_contract_sha256(),
        }


@pytest.fixture
def execution_setup(tmp_path, monkeypatch):
    repo = tmp_path / "anima-mcp"
    files = {
        "pyproject.toml": '[project]\nname = "anima-mcp"\nversion = "9.9.9"\n',
        "README.md": "# Test creature\n",
        "src/anima_mcp/display/eras/test_era.py": "def draw():\n    return None\n",
        "tests/test_placeholder.py": "def test_placeholder():\n    assert True\n",
    }
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _commit_repo(repo)

    clock = MutableClock(datetime(2026, 8, 11, 22, 30, tzinfo=timezone.utc))
    keys = {
        "proposer": VerifierKey("proposer-1", "key-p", b"p" * 32),
        "verifier": VerifierKey("verifier-1", "key-v", b"v" * 32),
        "approver": VerifierKey("approver-1", "key-a", b"a" * 32),
        "other": VerifierKey("other-1", "key-o", b"o" * 32),
        "signer": VerifierKey("runner-signer", "key-r", b"r" * 32),
        "reviewer": VerifierKey("reviewer-1", "key-w", b"w" * 32),
        "app_signer": VerifierKey("application-signer", "key-s", b"s" * 32),
    }
    registry = KeyRegistry(*keys.values())
    runner = FakeIsolationRunner()
    ledger = tmp_path / "state" / "self_iteration.json"
    sandbox_root = tmp_path / "quarantine"
    monkeypatch.setenv("ANIMA_SELF_ITERATION_RUNNER_SIGNER_ID", "runner-signer")
    monkeypatch.setenv("ANIMA_SELF_ITERATION_APPLIER_SIGNER_ID", "application-signer")

    def system(actor_id: str | None) -> SelfIterationSystem:
        return SelfIterationSystem(
            repo_root=repo,
            ledger_path=ledger,
            sandbox_root=sandbox_root,
            clock=clock,
            provenance_provider=_provenance_provider(actor_id),
            verifier_key_provider=registry,
            isolation_runner=runner,
        )

    return SimpleNamespace(
        repo=repo,
        ledger=ledger,
        sandbox_root=sandbox_root,
        clock=clock,
        keys=keys,
        registry=registry,
        runner=runner,
        proposer=system("proposer-1"),
        verifier=system("verifier-1"),
        approver=system("approver-1"),
        reviewer=system("reviewer-1"),
        other=system("other-1"),
        unauthenticated=system(None),
    )


def _prepared_lifecycle(setup):
    proposal = setup.proposer.propose(
        observation="The era repeats a converged mark.",
        hypothesis="A bounded variation will reduce repetition.",
        expected_outcome="The fixed era tests remain green.",
        evidence=["8 of 20 recent drawings repeated the mark"],
        target_paths=["src/anima_mcp/display/eras/test_era.py"],
        verification=["Run the fixed display-era test profile"],
        risk="low",
    )
    verification = setup.verifier.prepare_verification(
        proposal_id=proposal["id"],
        verification_decision="verified",
        verification_statement="Independent evidence supports a bounded test.",
        verification_evidence=[
            {
                "kind": "canary",
                "uri": "artifact://canary/phase4.json",
                "sha256": "d" * 64,
            }
        ],
        expected_content_sha256=proposal["content_sha256"],
    )
    setup.verifier.record_verification(
        proposal_id=proposal["id"],
        challenge_id=verification["challenge_id"],
        signature=sign_attestation(verification["attestation"], setup.keys["verifier"]),
    )
    target = setup.repo / "src/anima_mcp/display/eras/test_era.py"
    candidate = setup.proposer.construct_patch(
        proposal_id=proposal["id"],
        expected_content_sha256=proposal["content_sha256"],
        changes=[
            {
                "path": "src/anima_mcp/display/eras/test_era.py",
                "expected_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "content": "def draw():\n    return 'varied'\n",
            }
        ],
    )["candidate"]
    evaluation = setup.verifier.evaluate_patch(
        proposal_id=proposal["id"],
        candidate_id=candidate["candidate_id"],
        expected_candidate_sha256=candidate["candidate_sha256"],
    )["evaluation"]
    prepared = setup.approver.prepare_execution(
        proposal_id=proposal["id"],
        candidate_id=candidate["candidate_id"],
        expected_candidate_sha256=candidate["candidate_sha256"],
        evaluation_id=evaluation["evaluation_id"],
        expected_evaluation_sha256=evaluation["evaluation_sha256"],
    )
    return proposal, candidate, evaluation, prepared


class TestIsolatedExecutionLifecycle:
    @pytest.mark.asyncio
    async def test_handler_exposes_approved_execution_lifecycle(
        self, execution_setup, monkeypatch
    ):
        setup = execution_setup
        proposal, candidate, _evaluation, prepared = _prepared_lifecycle(setup)
        approval = prepared["approval"]
        monkeypatch.setattr(
            "anima_mcp.handlers.self_iteration.get_self_iteration_system",
            lambda: setup.approver,
        )
        from anima_mcp.handlers.self_iteration import handle_self_iteration

        executed = parse_result(
            await handle_self_iteration(
                {
                    "action": "execute_candidate",
                    "proposal_id": proposal["id"],
                    "challenge_id": approval["challenge_id"],
                    "signature": sign_execution_approval(
                        approval, setup.keys["approver"]
                    ),
                }
            )
        )
        status = parse_result(
            await handle_self_iteration(
                {
                    "action": "execution_status",
                    "proposal_id": proposal["id"],
                    "candidate_id": candidate["candidate_id"],
                }
            )
        )

        assert executed["success"] is True
        assert executed["result"]["outcome"] == "passed"
        assert status["executions"][0]["state"] == "recorded"
        assert status["authority_granted"] is False

    def test_distinct_approval_executes_once_and_records_signed_result(
        self, execution_setup
    ):
        setup = execution_setup
        before = {
            path.relative_to(setup.repo).as_posix(): path.read_bytes()
            for path in setup.repo.rglob("*")
            if path.is_file() and ".git" not in path.parts
        }
        proposal, candidate, evaluation, prepared = _prepared_lifecycle(setup)
        approval = prepared["approval"]
        signature = sign_execution_approval(approval, setup.keys["approver"])

        executed = setup.approver.execute_candidate(
            proposal_id=proposal["id"],
            challenge_id=approval["challenge_id"],
            signature=signature,
        )
        result = executed["result"]

        assert setup.runner.run_count == 1
        assert setup.runner.seen_candidate == b"def draw():\n    return 'varied'\n"
        assert result["candidate_sha256"] == candidate["candidate_sha256"]
        assert result["evaluation_sha256"] == evaluation["evaluation_sha256"]
        assert result["signature"]["signer_id"] == "runner-signer"
        assert validate_signed_execution_result(result, setup.registry) == result
        assert result["eligible_for_external_review"] is True
        assert result["eligible_for_apply"] is False
        assert result["authority_granted"] is False
        after = {
            path.relative_to(setup.repo).as_posix(): path.read_bytes()
            for path in setup.repo.rglob("*")
            if path.is_file() and ".git" not in path.parts
        }
        assert after == before

        status = setup.unauthenticated.execution_status(
            proposal_id=proposal["id"], candidate_id=candidate["candidate_id"]
        )
        assert status["executions"][0]["state"] == "recorded"
        assert status["executions"][0]["result"]["output_omitted"] is True
        assert "captured" not in status["executions"][0]["result"]["stdout"]
        full_status = setup.approver.execution_status(
            proposal_id=proposal["id"],
            candidate_id=candidate["candidate_id"],
            include_output=True,
        )
        assert full_status["executions"][0]["result"]["stdout"]["captured"] == ""
        with pytest.raises(SelfIterationError, match="authenticated actor"):
            setup.unauthenticated.execution_status(
                proposal_id=proposal["id"],
                candidate_id=candidate["candidate_id"],
                include_output=True,
            )

        with pytest.raises(SelfIterationError, match="already claimed"):
            setup.approver.execute_candidate(
                proposal_id=proposal["id"],
                challenge_id=approval["challenge_id"],
                signature=signature,
            )
        assert setup.runner.run_count == 1

    @pytest.mark.parametrize("actor", ["proposer", "verifier"])
    def test_proposer_and_active_verifier_cannot_approve(self, execution_setup, actor):
        setup = execution_setup
        proposal, candidate, evaluation, _prepared = _prepared_lifecycle(setup)
        with pytest.raises(SelfIterationError, match="must differ"):
            getattr(setup, actor).prepare_execution(
                proposal_id=proposal["id"],
                candidate_id=candidate["candidate_id"],
                expected_candidate_sha256=candidate["candidate_sha256"],
                evaluation_id=evaluation["evaluation_id"],
                expected_evaluation_sha256=evaluation["evaluation_sha256"],
            )

    @pytest.mark.parametrize(
        ("actor_id", "kind", "issuer"),
        [
            ("proposer-1", "oauth_client", None),
            ("proposer-1", "oauth_subject", "https://other-issuer.example.test"),
            ("verifier-1", "oauth_client", None),
            ("verifier-1", "oauth_subject", "https://other-issuer.example.test"),
        ],
    )
    def test_same_participant_id_with_different_kind_or_issuer_cannot_approve(
        self, execution_setup, actor_id, kind, issuer
    ):
        setup = execution_setup
        proposal, candidate, evaluation, _prepared = _prepared_lifecycle(setup)

        def alternate_provenance(recorded_at: str) -> dict:
            return {
                "schema": PROVENANCE_SCHEMA,
                "recorded_by": "anima-mcp-execution-test",
                "recorded_at": recorded_at,
                "transport": {"kind": "test", "server_observed": True},
                "authentication": {"method": "oauth_bearer", "verified": True},
                "actor": {
                    "kind": kind,
                    "id": actor_id,
                    "issuer": issuer,
                    "verified": True,
                },
                "session": {"present": False, "verified": False},
                "trust": {},
            }

        aliased_actor = SelfIterationSystem(
            repo_root=setup.repo,
            ledger_path=setup.ledger,
            sandbox_root=setup.sandbox_root,
            clock=setup.clock,
            provenance_provider=alternate_provenance,
            verifier_key_provider=setup.registry,
            isolation_runner=setup.runner,
        )
        with pytest.raises(SelfIterationError, match="must differ"):
            aliased_actor.prepare_execution(
                proposal_id=proposal["id"],
                candidate_id=candidate["candidate_id"],
                expected_candidate_sha256=candidate["candidate_sha256"],
                evaluation_id=evaluation["evaluation_id"],
                expected_evaluation_sha256=evaluation["evaluation_sha256"],
            )

    def test_bad_or_expired_signature_never_consumes_challenge(self, execution_setup):
        setup = execution_setup
        proposal, candidate, _evaluation, prepared = _prepared_lifecycle(setup)
        approval = prepared["approval"]

        with pytest.raises(SelfIterationError, match="signature is invalid"):
            setup.approver.execute_candidate(
                proposal_id=proposal["id"],
                challenge_id=approval["challenge_id"],
                signature="0" * 64,
            )
        status = setup.approver.execution_status(
            proposal_id=proposal["id"], candidate_id=candidate["candidate_id"]
        )
        assert status["executions"][0]["state"] == "awaiting_signature"

        setup.clock.advance(minutes=11)
        with pytest.raises(SelfIterationError, match="expired"):
            setup.approver.execute_candidate(
                proposal_id=proposal["id"],
                challenge_id=approval["challenge_id"],
                signature=sign_execution_approval(approval, setup.keys["approver"]),
            )
        status = setup.approver.execution_status(
            proposal_id=proposal["id"], candidate_id=candidate["candidate_id"]
        )
        assert status["executions"][0]["state"] == "expired_unclaimed"
        assert setup.runner.run_count == 0

    def test_runner_change_fails_before_claim(self, execution_setup):
        setup = execution_setup
        proposal, candidate, _evaluation, prepared = _prepared_lifecycle(setup)
        approval = prepared["approval"]
        setup.runner.identity = _runner_identity("c")

        with pytest.raises(SelfIterationError, match="identity changed"):
            setup.approver.execute_candidate(
                proposal_id=proposal["id"],
                challenge_id=approval["challenge_id"],
                signature=sign_execution_approval(approval, setup.keys["approver"]),
            )
        status = setup.approver.execution_status(
            proposal_id=proposal["id"], candidate_id=candidate["candidate_id"]
        )
        assert status["executions"][0]["state"] == "awaiting_signature"

    def test_failure_after_claim_is_indeterminate_and_not_retried(
        self, execution_setup
    ):
        setup = execution_setup
        proposal, candidate, _evaluation, prepared = _prepared_lifecycle(setup)
        approval = prepared["approval"]
        signature = sign_execution_approval(approval, setup.keys["approver"])
        setup.runner.fail_run = True

        with pytest.raises(SelfIterationError, match="simulated"):
            setup.approver.execute_candidate(
                proposal_id=proposal["id"],
                challenge_id=approval["challenge_id"],
                signature=signature,
            )
        status = setup.approver.execution_status(
            proposal_id=proposal["id"], candidate_id=candidate["candidate_id"]
        )
        assert status["executions"][0]["state"] == "claimed_result_indeterminate"
        assert status["executions"][0]["automatic_retry_allowed"] is False

        setup.runner.fail_run = False
        with pytest.raises(SelfIterationError, match="already claimed"):
            setup.approver.execute_candidate(
                proposal_id=proposal["id"],
                challenge_id=approval["challenge_id"],
                signature=signature,
            )
        assert setup.runner.run_count == 1

    def test_tampered_signed_result_fails_closed(self, execution_setup):
        setup = execution_setup
        proposal, candidate, _evaluation, prepared = _prepared_lifecycle(setup)
        approval = prepared["approval"]
        executed = setup.approver.execute_candidate(
            proposal_id=proposal["id"],
            challenge_id=approval["challenge_id"],
            signature=sign_execution_approval(approval, setup.keys["approver"]),
        )
        result_path = (
            setup.sandbox_root
            / candidate["candidate_id"]
            / "executions"
            / f"{executed['result']['execution_id']}.json"
        )
        tampered = json.loads(result_path.read_text())
        tampered["outcome"] = "failed"
        result_path.write_text(json.dumps(tampered))

        with pytest.raises(SelfIterationError, match="signed execution result"):
            setup.approver.execution_status(
                proposal_id=proposal["id"], candidate_id=candidate["candidate_id"]
            )

    def test_tampered_durable_approval_signature_fails_closed(self, execution_setup):
        setup = execution_setup
        proposal, candidate, _evaluation, prepared = _prepared_lifecycle(setup)
        approval = prepared["approval"]
        setup.approver.execute_candidate(
            proposal_id=proposal["id"],
            challenge_id=approval["challenge_id"],
            signature=sign_execution_approval(approval, setup.keys["approver"]),
        )
        claim_path = (
            setup.sandbox_root
            / candidate["candidate_id"]
            / "execution_claims"
            / f"{approval['challenge_id']}.json"
        )
        tampered = json.loads(claim_path.read_text())
        tampered["approval_signature"]["value"] = "0" * 64
        claim_path.write_text(json.dumps(tampered))

        with pytest.raises(SelfIterationError, match="approval signature is invalid"):
            setup.approver.execution_status(
                proposal_id=proposal["id"], candidate_id=candidate["candidate_id"]
            )

    def test_schema_v4_migrates_to_execution_contract(self, execution_setup):
        setup = execution_setup
        proposal = setup.proposer.propose(
            observation="Migration fixture.",
            hypothesis="The schema migration remains no-authority.",
            expected_outcome="A v5 execution contract is added.",
            evidence=["local ledger"],
            target_paths=["src/anima_mcp/display/eras/test_era.py"],
            verification=["inspect migration event"],
            risk="low",
        )
        ledger = json.loads(setup.ledger.read_text())
        ledger["schema_version"] = 4
        ledger.pop("execution_contract")
        setup.ledger.write_text(json.dumps(ledger))

        migrated = setup.proposer.list_proposals(proposal_id=proposal["id"])[
            "proposals"
        ][0]
        on_disk = json.loads(setup.ledger.read_text())
        assert migrated["events"][-3] == {
            "type": "execution_schema_migrated",
            "at": "2026-08-11T22:30:00Z",
            "from_schema": 4,
            "to_schema": 5,
            "authority_granted": False,
        }
        assert migrated["events"][-2] == {
            "type": "application_schema_migrated",
            "at": "2026-08-11T22:30:00Z",
            "from_schema": 5,
            "to_schema": 6,
            "authority_granted": False,
        }
        assert migrated["events"][-1] == {
            "type": "canary_schema_migrated",
            "at": "2026-08-11T22:30:00Z",
            "from_schema": 6,
            "to_schema": 7,
            "authority_granted": False,
        }
        assert on_disk["schema_version"] == 7
        assert on_disk["execution_contract"]["automatic_apply"] is False
        assert on_disk["application_contract"]["push_allowed"] is False
        assert on_disk["canary_contract"]["persistent_activation_allowed"] is False


@pytest.mark.skipif(
    os.environ.get("ANIMA_SELF_ITERATION_RUN_DOCKER_TESTS") != "1",
    reason="set ANIMA_SELF_ITERATION_RUN_DOCKER_TESTS=1 with a configured image",
)
def test_real_docker_runner_security_envelope(tmp_path, monkeypatch):
    """Optional operator test against the configured local digest-pinned image."""
    monkeypatch.setenv("ANIMA_PHASE4_CANARY_SECRET", "must-not-cross-boundary")
    workspace = tmp_path / "workspace"
    target = workspace / "src/anima_mcp/display/eras/example.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 'candidate'\n")
    tests = workspace / "tests"
    tests.mkdir()
    security_test = """
import os
import socket
from pathlib import Path

import pytest


def test_boundary():
    assert os.environ.get("ANIMA_PHASE4_CANARY_SECRET") is None
    with pytest.raises(OSError):
        Path("/workspace/forbidden-write").write_text("no")
    with pytest.raises(OSError):
        socket.create_connection(("1.1.1.1", 53), timeout=0.1)
"""
    fixed_tests = [
        "test_art_era.py",
        "test_era_registry.py",
        "test_gestural_era.py",
        "test_resonance_era.py",
        "test_drawing_earned_completion.py",
    ]
    for index, name in enumerate(fixed_tests):
        (tests / name).write_text(
            security_test
            if index == 0
            else "def test_placeholder():\n    assert True\n"
        )
    for path in sorted(
        workspace.rglob("*"), key=lambda item: len(item.parts), reverse=True
    ):
        path.chmod(0o555 if path.is_dir() else 0o444)
    workspace.chmod(0o555)

    runner = DockerIsolationRunner.from_environment()
    identity = runner.probe()
    receipt = runner.run(
        workspace=workspace,
        profile=DISPLAY_ERA_TEST_PROFILE,
        expected_identity=identity,
    )
    assert receipt["outcome"] == "passed"
    assert receipt["cleanup_confirmed"] is True
