"""Deployment wiring must preserve the single-owner runtime architecture."""

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_restore_template_and_warning_include_outbound_heartbeat():
    template = (ROOT / "config" / "anima.env.example").read_text()
    restore = (ROOT / "scripts" / "restore_lumen.sh").read_text()

    assert "ANIMA_HEARTBEAT_URL=" in template
    assert "ANIMA_HEARTBEAT_URL is empty" in restore
    assert "dead-man's switch is INERT" in restore
    assert "No service restart is required" in restore


def test_systemd_units_pin_single_owners():
    server = (ROOT / "systemd" / "anima.service").read_text()
    broker = (ROOT / "systemd" / "anima-broker.service").read_text()
    elixir_broker = (ROOT / "anima_broker" / "systemd" / "anima-broker-ex.service").read_text()

    assert 'Environment="ANIMA_SENSORS_BACKEND=shm"' in server
    assert 'Environment="ANIMA_CONFIG=/home/unitares-anima/.anima/anima_config.json"' in server
    assert 'Environment="ANIMA_CONFIG=/home/unitares-anima/.anima/anima_config.json"' in broker
    assert 'Environment="ANIMA_BROKER_AGENCY_ENABLED=false"' in broker
    assert 'Environment="ANIMA_BROKER_VOICE_ENABLED=false"' in broker
    assert "Requires=anima-restore.service" in broker
    assert "PartOf=anima.service" not in broker
    assert "UMask=0077" in server
    assert "UMask=0077" in broker
    assert "UMask=0077" in elixir_broker


def test_deploy_syncs_core_units_before_restart():
    script = (ROOT / "deploy.sh").read_text()

    sync_at = script.index("Syncing core systemd units")
    restart_at = script.index("Restarting anima service")
    assert sync_at < restart_at
    assert "anima-restore.service anima-broker.service anima.service" in script
    assert '"set -e; changed=0;' in script
    assert "sudo install -m 0644" in script
    assert "sudo systemctl daemon-reload" in script
    assert "sudo systemctl enable anima-restore.service" in script
    assert "anima-storage-maintenance.service" in script
    assert "anima-storage-maintenance.timer" in script
    assert "enable --now anima-storage-maintenance.timer" in script
    assert "scripts/deploy_elixir_broker.sh" in script
    assert "anima-broker-ex" in script
    assert "--exclude='anima_broker/_build'" in script
    assert "--exclude='anima_broker/deps'" in script
    assert "--exclude='.ruff_cache'" in script


def test_deploy_records_the_exact_clean_commit_after_rsync():
    script = (ROOT / "deploy.sh").read_text()
    aligner = (ROOT / "scripts" / "align_deployed_checkout.sh").read_text()

    clean_at = script.index("refusing deploy from a dirty source checkout")
    sync_at = script.index("Syncing code")
    align_at = script.index("Aligning deployed Git revision")
    restart_at = script.index("Restarting anima service")

    assert clean_at < sync_at < align_at < restart_at
    assert 'DEPLOYED_REF="$(git rev-parse --verify HEAD^{commit}' in script
    assert "bash $PI_PATH/scripts/align_deployed_checkout.sh '$DEPLOYED_REF'" in script
    assert "status --porcelain --untracked-files=all" in aligner
    tracking_fetch = "fetch --quiet --no-tags origin\n"
    exact_fetch = 'fetch --quiet --no-tags origin "$DEPLOYED_REF"'
    assert tracking_fetch in aligner
    assert aligner.index(tracking_fetch) < aligner.index(exact_fetch)
    assert 'reset --mixed "$DEPLOYED_REF"' in aligner
    assert 'diff-index --quiet "$DEPLOYED_REF" --' in aligner
    assert "Git HEAD, index, and deployed files agree" in script
    assert "Services were not restarted" in script


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _rsynced_checkout(tmp_path):
    """A checkout at commit A after rsync copied commit C over it without --delete.

    B edits stale.txt. C deletes it and gone.txt, renames moved.txt to
    renamed.txt and edits keep.txt. Returns the checkout and C's SHA.
    """
    source = tmp_path / "source"
    _git(tmp_path, "init", "-q", "-b", "main", "source")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    (source / "scripts").mkdir()
    aligner = ROOT / "scripts" / "align_deployed_checkout.sh"
    (source / "scripts" / aligner.name).write_text(aligner.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("keep.txt", "gone.txt", "moved.txt", "stale.txt"):
        (source / name).write_text(f"{name} at A\n", encoding="utf-8")
    _git(source, "add", "-A")
    _git(source, "commit", "-q", "-m", "A")
    first = _git(source, "rev-parse", "HEAD")
    (source / "stale.txt").write_text("stale.txt at B\n", encoding="utf-8")
    _git(source, "commit", "-q", "-am", "B")
    _git(source, "rm", "-q", "gone.txt", "stale.txt")
    _git(source, "mv", "moved.txt", "renamed.txt")
    (source / "keep.txt").write_text("keep.txt at C\n", encoding="utf-8")
    _git(source, "commit", "-q", "-am", "C")
    deployed = _git(source, "rev-parse", "HEAD")

    _git(tmp_path, "clone", "-q", "--bare", "source", "origin.git")
    _git(tmp_path, "clone", "-q", "origin.git", "pi")
    pi = tmp_path / "pi"
    _git(pi, "reset", "-q", "--hard", first)
    for name in ("keep.txt", "renamed.txt"):
        (pi / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    return pi, deployed


def _align(checkout, ref):
    return subprocess.run(
        ["bash", str(checkout / "scripts" / "align_deployed_checkout.sh"), ref],
        capture_output=True, text=True, timeout=60,
    )


def test_aligner_removes_leftovers_of_files_the_deployed_history_deleted(tmp_path):
    pi, deployed = _rsynced_checkout(tmp_path)

    result = _align(pi, deployed)

    assert result.returncode == 0, result.stderr
    assert "Removed leftover gone.txt" in result.stdout
    assert not (pi / "gone.txt").exists()
    # A rename leaves its old path behind too.
    assert not (pi / "moved.txt").exists()
    # The Pi missed B, so its copy matches A, not the version C deleted.
    assert not (pi / "stale.txt").exists()
    assert _git(pi, "rev-parse", "HEAD") == deployed
    assert _git(pi, "status", "--porcelain", "--untracked-files=all") == ""


def test_aligner_keeps_and_names_files_git_cannot_restore(tmp_path):
    pi, deployed = _rsynced_checkout(tmp_path)
    (pi / "moved.txt").write_text("edited on the Pi\n", encoding="utf-8")
    (pi / "notes.txt").write_text("never tracked\n", encoding="utf-8")
    # Git stored the file, never a link, even one whose target has its bytes.
    (pi / "stale-copy.txt").write_text("stale.txt at A\n", encoding="utf-8")
    (pi / "stale.txt").unlink()
    (pi / "stale.txt").symlink_to("stale-copy.txt")
    # A glob in a name must not match the deleted gone.txt.
    (pi / "gon?.txt").symlink_to("missing-target")

    result = _align(pi, deployed)

    assert result.returncode == 1
    assert not (pi / "gone.txt").exists()
    assert (pi / "moved.txt").read_text(encoding="utf-8") == "edited on the Pi\n"
    assert (pi / "notes.txt").exists()
    assert (pi / "stale.txt").is_symlink()
    assert (pi / "gon?.txt").is_symlink()
    for name in ("moved.txt", "notes.txt", "stale.txt", "stale-copy.txt", "gon?.txt"):
        assert f"?? {name}" in result.stderr


def test_deploy_fails_closed_on_permissive_rest_and_hardens_state_modes():
    script = (ROOT / "deploy.sh").read_text()

    security_at = script.index("Checking runtime security")
    sync_at = script.index("Syncing code")
    assert security_at < sync_at
    assert "ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN" in script
    assert "ANIMA_OAUTH_DYNAMIC_REGISTRATION" in script
    assert "permissive-no-token" in script
    assert "registration_endpoint" in script
    assert 'chmod 700 "$HOME/.anima"' in script
    assert "chmod 600" in script
    assert "closed OAuth registration verified" in script


def test_deploy_fails_closed_on_backup_restart_or_runtime_verification():
    script = (ROOT / "deploy.sh").read_text()

    assert script.index("Migrating persistent calibration") < script.index("Backing up Pi state")
    assert "anima_config.json" in script
    assert "--skip-backup" in script
    assert 'python3 scripts/sync_state.py "${SYNC_ARGS[@]}"' in script
    assert "Verified state backup failed; deployment aborted" in script
    assert "sudo systemctl restart anima-broker anima" in script
    assert "systemctl is-active --quiet anima-broker" in script
    assert "ANIMA_SENSORS_BACKEND=shm" in script
    assert "/dev/shm/anima_state.json" in script
    assert "ServerAliveInterval=10" in script
    assert "for _attempt in 1 2 3" in script
    assert "Could not capture pre-restart service PIDs" in script
    assert "Restart connection interrupted; verifying changed service PIDs" in script
    assert "new_broker_pid" in script
    assert r'test \"\$new_broker_pid\" != ' in script
    assert r'test \"\$new_anima_pid\" != ' in script
    assert "'$OLD_BROKER_PID'" in script
    assert "'$OLD_ANIMA_PID'" in script
    assert "Restart or post-deploy verification failed" in script


def test_restore_quiesces_writers_and_restores_durable_event_inbox():
    script = (ROOT / "scripts" / "restore_lumen.sh").read_text()

    preflight_at = script.index("Recovery set verified and frozen before mutation")
    stop_at = script.index("Quiescing Lumen before restore")
    deploy_at = script.index("Deploying code")
    restore_at = script.index("Restoring Lumen data")
    assert preflight_at < stop_at < deploy_at < restore_at
    assert "anima_config.json preferences.json self_model.json patterns.json metacognition_baselines.json" in script
    assert "lumen-restore.XXXXXX" in script
    assert "--no-restart --skip-backup" in script
    assert "systemd/anima-restore.service" in script
    assert "scripts/deploy_elixir_broker.sh" in script
    assert "anima-broker-ex" in script
    assert "/dev/shm/anima_state.shadow.json" in script
    assert ".restore-anima.db" in script
    assert "PRAGMA integrity_check" in script
    assert '"$BACKUP/learning_inbox/"' in script
    assert "--delete" in script
    assert "refusing a silent fresh identity" in script
    assert "fresh shared state verified" in script


def test_nightly_backup_publishes_learned_state_only_from_staging():
    script = (ROOT / "scripts" / "backup_lumen_from_mac.sh").read_text()

    assert ".anima_data.staging." in script
    assert "anima_data.previous" in script
    assert "--exclude='*.db*'" in script
    assert "--exclude='anima.env*'" in script
    assert "this run is not a complete restore point" in script
    assert "MIRROR_CAPTURED -eq 1" in script
    assert "required restore state was incomplete" in script
    assert "lumen_recovery_${DAY_NAME}.tar" in script
    assert "hf_upload_with_watchdog" in script
    assert '"anima_${DATE}.db" "$(basename "$STATE_ARCHIVE")"' in script
    assert "PI_BACKUP_MAX_AGE_MINUTES" in script
    assert "recent Pi snapshot failed integrity_check" in script
    assert "~/.anima/oauth.db" in script
    assert "--exclude='./oauth.db'" in script
    assert "anima.offdevice-recovery.v1" in script
    assert "restore_bundle_verified" in script
    assert "Forensic archives mirrored off-device" in script
    assert "ANIMA_DB_RETAIN" in script


def test_storage_maintenance_is_pressure_aware_and_restorable():
    maintenance = (ROOT / "scripts" / "storage_maintenance.py").read_text()
    service = (ROOT / "systemd" / "anima-storage-maintenance.service").read_text()
    timer = (ROOT / "systemd" / "anima-storage-maintenance.timer").read_text()
    restore = (ROOT / "scripts" / "restore_lumen.sh").read_text()
    state_backup = (ROOT / "scripts" / "backup_state.sh").read_text()

    assert 'PressurePolicy("warning", 75.0' in maintenance
    assert 'PressurePolicy("action", 80.0' in maintenance
    assert 'PressurePolicy("urgent", 85.0' in maintenance
    assert "newest incident always remains unpacked" in maintenance
    assert "offdevice-recovery-receipt.json" in maintenance
    assert "verify_forensic_archive" in maintenance
    assert "--apply" in service
    assert "User=unitares-anima" in service
    assert "OnCalendar=*-*-* *:45:00" in timer
    assert "backup_db.sh" in restore
    assert "anima-storage-maintenance.timer" in restore
    assert "anima_config.json" in state_backup


def test_predeploy_snapshot_uses_sqlite_backup_and_separate_bundle():
    script = (ROOT / "scripts" / "sync_state.py").read_text()

    assert "src.backup(dst)" in script
    assert "PRAGMA integrity_check" in script
    assert 'staging / "anima_data"' in script
    assert 'backup_root / "latest"' in script
    assert "LEGACY_LOCAL_DB" in script  # read-only compatibility for guarded push
    assert "shutil.copy2" not in script


def test_boot_restore_gates_silent_identity_replacement():
    script = (ROOT / "scripts" / "auto_restore.sh").read_text()
    unit = (ROOT / "systemd" / "anima-restore.service").read_text()

    assert "PRAGMA integrity_check" in script
    assert "learned_state_ok" in script
    assert "ANIMA_ALLOW_FRESH_START" in script
    assert "Refusing silent fresh start" in script
    assert ".auto-restore-state." in script
    assert ".auto-restore-db." in script
    assert "anima_data.previous" in script
    assert "learning_inbox/***" in script
    assert "find \"$ANIMA_DIR\" -maxdepth 1 -type f -name '*.json' -delete" in script
    assert ".pre-restore-anima.db" in script
    assert "anima_config.json preferences.json self_model.json patterns.json metacognition_baselines.json" in script
    assert "anima_data/anima.db" not in script
    assert "sqlite3 \"$DB_PATH\"" not in script
    assert "ConditionPathExists=!/run/anima-restore-attempted" in unit
    assert "ExecStartPost=/bin/touch /run/anima-restore-attempted" in unit
    assert "EnvironmentFile=-/home/unitares-anima/.anima/anima.env" in unit


def test_bootstrap_restarts_both_services_after_unit_install():
    script = (ROOT / "scripts" / "bootstrap_deploy.py").read_text()

    install_at = script.index("Synchronizing core systemd units")
    restart_at = script.index("Restarting broker and anima services")
    assert install_at < restart_at
    assert "deploy_elixir_broker.sh" in script
    assert '["sudo", "systemctl", "restart", "anima-broker", "anima"]' in script
    assert "anima-storage-maintenance.timer" in script


def test_elixir_deploy_is_change_aware_and_verifies_shadow_state():
    script = (ROOT / "scripts" / "deploy_elixir_broker.sh").read_text()

    assert ".source-sha256" in script
    assert ".restart-required" in script
    assert "mix compile --warnings-as-errors" in script
    assert "mix release --overwrite" in script
    assert "sudo systemctl restart anima-broker-ex" in script
    assert "/dev/shm/anima_state.shadow.json" in script
    assert "Elixir broker is active but did not publish fresh shadow state" in script


def test_server_never_initializes_a_direct_sensor_backend():
    server = (ROOT / "src" / "anima_mcp" / "server.py").read_text()
    accessors = (ROOT / "src" / "anima_mcp" / "accessors.py").read_text()

    assert "from .sensors import get_sensors" not in server
    assert 'get_sensors(backend="shm")' in accessors
    assert "no direct sensor takeover" in server


# --- deployed-ref marker: git on the Pi must not be the only witness --------
# Deploys reset/clean/overlay the checkout, so `git log` there cannot be
# trusted to name the running code. The marker written at process start to
# ~/.anima/deployed_ref.json (outside the repo tree — deploys have wiped
# repo-resident files before) is the record an outside observer reads.


def test_server_startup_records_deployed_ref_marker():
    server = (ROOT / "src" / "anima_mcp" / "server.py").read_text()

    # After the pidfile is claimed, before the runtime wakes: every restart
    # path (deploy.sh, _delayed_restart, manual, reboot) passes through here.
    pid_at = server.index("pidfile.write_text(str(os.getpid()))")
    marker_at = server.index("write_startup_marker")
    # Leading space: "wake(db_path, anima_id)" alone also matches the
    # _lifecycle_wake delegation inside wake()'s own definition.
    wake_at = server.index(" wake(db_path, anima_id)")
    assert pid_at < marker_at < wake_at


def test_startup_marker_names_the_running_head(tmp_path):
    from anima_mcp import deploy_marker

    marker = tmp_path / "deployed_ref.json"
    assert deploy_marker.write_startup_marker(path=marker) is True

    data = json.loads(marker.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    assert data["head"] == head
    assert data["source"] == "startup"
    assert data["pid"] == os.getpid()
    assert data["started_at"].endswith("+00:00")
    # Atomic write: no half-written tmp residue left beside the marker.
    assert not marker.with_name(marker.name + ".tmp").exists()


def test_startup_marker_tolerates_absent_git(tmp_path, monkeypatch):
    from anima_mcp import deploy_marker

    gitless = tmp_path / "gitless"
    gitless.mkdir()
    monkeypatch.setattr(deploy_marker, "_repo_root", lambda: gitless)

    marker = tmp_path / "deployed_ref.json"
    # Without .git, a prior zip-overlay record is the ONLY statement of the
    # deployed ref — startup must carry it forward, not erase it.
    marker.write_text(json.dumps({"deployed_zip_sha": "ab" * 20}), encoding="utf-8")

    assert deploy_marker.write_startup_marker(path=marker) is True
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["head"] is None
    assert data["branch"] is None
    assert data["source"] == "startup"
    assert data["deployed_zip_sha"] == "ab" * 20


def test_startup_marker_failure_returns_false_never_raises(tmp_path):
    from anima_mcp import deploy_marker

    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    # Parent is a regular file: mkdir/replace must fail — quietly.
    marker = blocker / "deployed_ref.json"
    assert deploy_marker.write_startup_marker(path=marker) is False


def test_zip_overlay_record_merges_into_existing_marker(tmp_path):
    from anima_mcp import deploy_marker

    marker = tmp_path / "deployed_ref.json"
    assert deploy_marker.write_startup_marker(path=marker) is True
    assert deploy_marker.record_zip_overlay("cd" * 20, path=marker) is True

    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["deployed_zip_sha"] == "cd" * 20
    assert data["source"] == "zip-overlay"
    assert data["head"]  # startup fields survive the merge


def test_ship_script_routes_auto_merge_and_detects_agents(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts" / "dev").mkdir(parents=True)
    (repo / "src" / "anima_mcp").mkdir(parents=True)
    ship_src = ROOT / "scripts" / "dev" / "ship.sh"
    ship_dst = repo / "scripts" / "dev" / "ship.sh"
    ship_dst.write_text(ship_src.read_text(encoding="utf-8"), encoding="utf-8")
    ship_dst.chmod(0o755)

    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    # Non-runtime change on main -> direct push
    (repo / "README.md").write_text("updated\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    env = {**os.environ, "ANTIGRAVITY_AGENT": "1"}
    env.pop("CLAUDECODE", None)
    env.pop("UNITARES_SHIP_AGENT", None)
    plan_other = subprocess.run(
        [str(ship_dst), "--plan", "--deploy", "docs: update readme"],
        cwd=repo, env=env, check=True, capture_output=True, text=True,
    ).stdout
    assert "kind=other" in plan_other
    assert "delivery=direct" in plan_other
    assert "deploy_pi=1" in plan_other
    assert "agent_prefix=gemini" in plan_other

    # Runtime change on main -> auto_merge on fresh branch
    (repo / "src" / "anima_mcp" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/anima_mcp/mod.py"], cwd=repo, check=True)
    plan_rt = subprocess.run(
        [str(ship_dst), "--plan", "fix: runtime change"],
        cwd=repo, env=env, check=True, capture_output=True, text=True,
    ).stdout
    assert "kind=runtime" in plan_rt
    assert "delivery=auto_merge" in plan_rt
    assert "force_auto_branch=1" in plan_rt

