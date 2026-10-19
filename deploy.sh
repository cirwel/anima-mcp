#!/bin/bash
# Quick deploy script - Run from Mac to deploy changes to Pi

set -e

# Configuration
PI_HOST="${PI_HOST:-lumen.local}"
PI_USER="${PI_USER:-unitares-anima}"
PI_PORT="${PI_PORT:-22}"
PI_PATH="${PI_PATH:-~/anima-mcp}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519_pi}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Anima MCP - Deploy to Pi${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || ! grep -q "anima-mcp" pyproject.toml; then
    echo -e "${RED}Error: Must run from anima-mcp directory${NC}"
    exit 1
fi

# Parse arguments
RESTART=true
SHOW_LOGS=false
BACKUP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-restart)
            RESTART=false
            shift
            ;;
        --logs)
            SHOW_LOGS=true
            shift
            ;;
        --skip-backup)
            BACKUP=false
            shift
            ;;
        --host)
            PI_HOST="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./deploy.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-restart    Don't restart anima service after deploy"
            echo "  --skip-backup   Explicitly deploy without a pre-deploy state snapshot"
            echo "  --logs          Show logs after deploy"
            echo "  --host HOST     Override Pi hostname/IP (default: lumen.local)"
            echo "  --help          Show this help"
            echo ""
            echo "Environment variables:"
            echo "  PI_HOST         Pi hostname/IP (default: lumen.local)"
            echo "  PI_USER         Pi username (default: unitares-anima)"
            echo "  PI_PORT         SSH port (default: 22)"
            echo "  PI_PATH         Path on Pi (default: ~/anima-mcp)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# File-sync deploys must still name the exact commit they are deploying.  A
# dirty source tree has no commit whose contents can truthfully describe the
# files rsync will copy, so fail before taking a backup or touching the Pi.
if ! DEPLOYED_REF="$(git rev-parse --verify HEAD^{commit} 2>/dev/null)"; then
    echo -e "${RED}Error: deploy source is not a Git commit${NC}" >&2
    exit 1
fi
SOURCE_STATUS="$(git status --porcelain --untracked-files=all)"
if [ -n "$SOURCE_STATUS" ]; then
    echo -e "${RED}Error: refusing deploy from a dirty source checkout${NC}" >&2
    echo "$SOURCE_STATUS" >&2
    exit 1
fi

echo -e "${BLUE}Target:${NC} $PI_USER@$PI_HOST:$PI_PATH"
echo -e "${BLUE}Commit:${NC} $DEPLOYED_REF"
echo ""

SSH_EXTRA=""
[ -f "$SSH_KEY" ] && SSH_EXTRA="-i $SSH_KEY"

# Step 0a: move adaptive calibration out of the code checkout before rsync can
# ever replace or orphan it. This is an idempotent format migration; the
# resulting JSON joins every subsequent state snapshot.
echo -e "${BLUE}[0a/4] Migrating persistent calibration...${NC}"
if ssh -p $PI_PORT $SSH_EXTRA -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
    "$PI_USER@$PI_HOST" \
    "set -e; mkdir -p ~/.anima; py=$PI_PATH/.venv/bin/python; if [ ! -x \"\$py\" ]; then py=python3; fi; \"\$py\" - '$PI_PATH/anima_config.yaml' '$PI_PATH/anima_config.yaml.example'" <<'PY'
import json
import os
import sys
import uuid
from pathlib import Path

destination = Path.home() / ".anima" / "anima_config.json"
if destination.exists():
    data = json.loads(destination.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("persistent calibration is not a JSON object")
    raise SystemExit(0)

data = {}
for raw in sys.argv[1:]:
    source = Path(raw).expanduser()
    if source.is_file():
        try:
            import yaml
        except ImportError:
            if source.name.endswith(".example"):
                data = {}
                break
            raise SystemExit(f"PyYAML is required to migrate {source}")
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        break
if not isinstance(data, dict):
    raise SystemExit("checkout calibration is not a mapping")
temporary = destination.with_name(
    f"{destination.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
)
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, destination)
descriptor = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
then
    echo -e "${GREEN}  Persistent calibration ready${NC}"
else
    echo -e "${RED}✗ Calibration migration failed; deployment aborted${NC}"
    exit 1
fi
echo ""

# Step 0b: capture a verified DB + learned-state snapshot before mutation.
# This fails closed unless the operator explicitly chooses --skip-backup.
echo -e "${BLUE}[0b/4] Backing up Pi state...${NC}"
if [ "$BACKUP" = true ]; then
    SYNC_ARGS=(pull --host "$PI_HOST" --user "$PI_USER" --port "$PI_PORT")
    if [ -f "${SSH_KEY:-}" ]; then
        SYNC_ARGS+=(--identity-file "$SSH_KEY")
    fi
    if python3 scripts/sync_state.py "${SYNC_ARGS[@]}"; then
        echo -e "${GREEN}  State backed up${NC}"
    else
        echo -e "${RED}✗ Verified state backup failed; deployment aborted${NC}"
        echo -e "${YELLOW}  Use --skip-backup only after explicitly accepting that risk${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}  Backup explicitly skipped (--skip-backup)${NC}"
fi
echo ""

# Step 0c: refuse the legacy REST compatibility mode before changing the
# deployed checkout.  On an internet-routed host this flag exposes both state
# and /v1/tools/call without authentication.  Also repair historical modes on
# the state directory and SQLite sidecars; every runtime service uses the same
# owner, so group/world access is unnecessary.
echo -e "${BLUE}[0c/4] Checking runtime security...${NC}"
if ssh -p $PI_PORT $SSH_EXTRA -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
    "$PI_USER@$PI_HOST" \
    'set -e
if [ -f "$HOME/.anima/anima.env" ] && grep -Eiq "^[[:space:]]*ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN[[:space:]]*=[[:space:]]*(1|true|yes|on)[[:space:]]*(#.*)?$" "$HOME/.anima/anima.env"; then
    echo "Refusing deploy: ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN enables unauthenticated REST access" >&2
    echo "Set it to false before deploying" >&2
    exit 1
fi
if [ -f "$HOME/.anima/anima.env" ] && grep -Eiq "^[[:space:]]*ANIMA_OAUTH_DYNAMIC_REGISTRATION[[:space:]]*=[[:space:]]*(1|true|yes|on)[[:space:]]*(#.*)?$" "$HOME/.anima/anima.env"; then
    echo "Refusing deploy: ANIMA_OAUTH_DYNAMIC_REGISTRATION leaves public client enrollment open" >&2
    echo "Set it to false after connector onboarding and before deploying" >&2
    exit 1
fi
chmod 700 "$HOME/.anima"
for sensitive in "$HOME/.anima/anima.env" "$HOME/.anima/anima.env".* "$HOME/.anima/anima.db"* "$HOME/.anima/oauth.db"*; do
    [ ! -f "$sensitive" ] || chmod 600 "$sensitive"
done'; then
    echo -e "${GREEN}  Runtime authentication and secret modes verified${NC}"
else
    echo -e "${RED}✗ Runtime security check failed; deployment aborted${NC}"
    exit 1
fi
echo ""

# Step 1: Sync code
echo -e "${BLUE}[1/4] Syncing code...${NC}"
rsync -avz \
    --exclude='.venv' \
    --exclude='*.db' \
    --exclude='*.log' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    --exclude='htmlcov' \
    --exclude='anima_broker/_build' \
    --exclude='anima_broker/deps' \
    -e "ssh -p $PI_PORT $SSH_EXTRA -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new" \
    ./ "$PI_USER@$PI_HOST:$PI_PATH/"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Code synced${NC}"
else
    echo -e "${RED}✗ Sync failed (connection timeout?)${NC}"
    echo -e "${BLUE}  Lumen continues operating autonomously - deploy when WiFi returns${NC}"
    exit 1
fi

# rsync deliberately excludes .git and never deletes, so the just-synced
# aligner updates the checkout's ref and index, removes leftovers of files the
# deployed history deleted, and proves the copied files are exactly this
# commit. Do this before any service restart so a successful deploy cannot
# leave code and repository bookkeeping on different revisions.
echo -e "${BLUE}[1b/4] Aligning deployed Git revision...${NC}"
if ssh -p $PI_PORT $SSH_EXTRA -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
    "$PI_USER@$PI_HOST" "bash $PI_PATH/scripts/align_deployed_checkout.sh '$DEPLOYED_REF'"; then
    echo -e "${GREEN}✓ Git HEAD, index, and deployed files agree${NC}"
else
    echo -e "${RED}✗ Deployed files do not match commit $DEPLOYED_REF${NC}"
    echo -e "${YELLOW}  Services were not restarted; inspect the Pi checkout before retrying${NC}"
    exit 1
fi

# Step 2: Keep installed systemd units in lockstep with the deployed source.
# A code-only rsync left the live anima.service on its April 7 definition while
# the checkout contained the April 28 I2C single-owner safeguard.
echo ""
echo -e "${BLUE}[2/4] Syncing core systemd units and runtime releases...${NC}"
if ssh -p $PI_PORT $SSH_EXTRA -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$PI_USER@$PI_HOST" \
    "set -e; changed=0; for unit in anima-restore.service anima-broker.service anima.service anima-storage-maintenance.service anima-storage-maintenance.timer; do src=$PI_PATH/systemd/\$unit; dst=/etc/systemd/system/\$unit; if ! cmp -s \"\$src\" \"\$dst\"; then sudo install -m 0644 \"\$src\" \"\$dst\"; changed=1; fi; done; if [ \"\$changed\" -eq 1 ]; then sudo systemctl daemon-reload; fi; sudo systemctl enable anima-restore.service >/dev/null; sudo systemctl enable --now anima-storage-maintenance.timer >/dev/null"; then
    echo -e "${GREEN}✓ Core service definitions synchronized${NC}"
else
    echo -e "${RED}✗ Could not synchronize core service definitions${NC}"
    exit 1
fi

# The Elixir process is the deployed environmental-sensor owner. Its source is
# rsynced with the Python tree, but a running OTP release does not consume that
# source directly. Build and restart it only when its hashed release inputs or
# unit changed; unconfigured development hosts are skipped.
ELIXIR_DEPLOY_ARGS=""
if [ "$RESTART" != true ]; then
    ELIXIR_DEPLOY_ARGS="--no-restart"
fi
if ssh -p $PI_PORT $SSH_EXTRA -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$PI_USER@$PI_HOST" \
    "bash $PI_PATH/scripts/deploy_elixir_broker.sh $ELIXIR_DEPLOY_ARGS"; then
    echo -e "${GREEN}✓ Elixir sensor broker release synchronized${NC}"
else
    echo -e "${RED}✗ Elixir sensor broker synchronization failed${NC}"
    exit 1
fi

# Step 3: Restart service (if requested)
if [ "$RESTART" = true ]; then
    echo ""
    echo -e "${BLUE}[3/4] Restarting anima service...${NC}"
    
    # A deploy is not successful until both processes are active, the mind is
    # pinned to SHM, and the broker has published fresh, structurally valid state.
    VERIFY_CODE='import json,os,sys,time;p="/dev/shm/anima_state.json";started=time.time();end=started+45
while time.time()<end:
 try:
  envelope=json.load(open(p));data=envelope.get("data",{});fresh=os.path.getmtime(p)>=started
  if fresh and isinstance(data.get("readings"),dict) and isinstance(data.get("anima"),dict):sys.exit(0)
 except Exception:pass
 time.sleep(1)
sys.exit(1)'
    VERIFY_SHADOW_CODE='import json,os,sys,time;p="/dev/shm/anima_state.shadow.json";end=time.time()+15
while time.time()<end:
 try:
  envelope=json.load(open(p));data=envelope.get("data",{});fresh=time.time()-os.path.getmtime(p)<15
  if fresh and isinstance(data.get("readings"),dict):sys.exit(0)
 except Exception:pass
 time.sleep(1)
sys.exit(1)'
    VERIFY_SECURITY_CODE='import json,sys,time,urllib.error,urllib.request;end=time.time()+45
while time.time()<end:
 try:
  with urllib.request.urlopen("http://127.0.0.1:8766/state",timeout=2) as response:state=json.load(response)
  mode=(state.get("api_security") or {}).get("mode")
  oauth_closed=True
  try:
   with urllib.request.urlopen("http://127.0.0.1:8766/.well-known/oauth-authorization-server",timeout=2) as response:metadata=json.load(response)
   oauth_closed="registration_endpoint" not in metadata
  except urllib.error.HTTPError as error:
   oauth_closed=error.code==404
  if mode and mode!="permissive-no-token" and oauth_closed:sys.exit(0)
 except Exception:pass
 time.sleep(1)
sys.exit(1)'
    # Keep the short restart separate from the longer readiness checks.  A
    # transient SSH disconnect during verification must be retryable without
    # restarting Lumen again.
    # Capture the old PIDs first. If WiFi drops while systemctl is restarting
    # the services, a later readiness check may still prove that the requested
    # restart completed. Requiring both PIDs to change prevents an untouched,
    # already-healthy runtime from being mistaken for a successful deploy.
    if ! OLD_PIDS=$(ssh -p $PI_PORT $SSH_EXTRA -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$PI_USER@$PI_HOST" \
        "printf '%s %s\n' \"\$(systemctl show anima-broker -p MainPID --value)\" \"\$(systemctl show anima -p MainPID --value)\""); then
        echo -e "${RED}✗ Could not capture pre-restart service PIDs${NC}"
        exit 1
    fi
    read -r OLD_BROKER_PID OLD_ANIMA_PID <<< "$OLD_PIDS"
    if [[ ! "$OLD_BROKER_PID" =~ ^[1-9][0-9]*$ ]] || [[ ! "$OLD_ANIMA_PID" =~ ^[1-9][0-9]*$ ]]; then
        echo -e "${RED}✗ Invalid pre-restart service PIDs: broker=$OLD_BROKER_PID anima=$OLD_ANIMA_PID${NC}"
        exit 1
    fi

    if ! ssh -p $PI_PORT $SSH_EXTRA -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$PI_USER@$PI_HOST" \
        "sudo systemctl restart anima-broker anima"; then
        echo -e "${YELLOW}  Restart connection interrupted; verifying changed service PIDs${NC}"
    fi

    VERIFIED=false
    for _attempt in 1 2 3; do
        if ssh -p $PI_PORT $SSH_EXTRA -o ConnectTimeout=5 -o ServerAliveInterval=10 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new "$PI_USER@$PI_HOST" \
            "set -e; new_broker_pid=\$(systemctl show anima-broker -p MainPID --value); new_anima_pid=\$(systemctl show anima -p MainPID --value); test \"\$new_broker_pid\" -gt 0; test \"\$new_anima_pid\" -gt 0; test \"\$new_broker_pid\" != '$OLD_BROKER_PID'; test \"\$new_anima_pid\" != '$OLD_ANIMA_PID'; systemctl is-active --quiet anima-broker; systemctl is-active --quiet anima; systemctl show anima -p Environment --value | tr ' ' '\n' | grep -qx 'ANIMA_SENSORS_BACKEND=shm'; if grep -Eq '^[[:space:]]*ANIMA_ENV_SENSORS_FROM_SHM=[^[:space:]#]+' ~/.anima/anima.env 2>/dev/null; then systemctl is-active --quiet anima-broker-ex; python3 -c '$VERIFY_SHADOW_CODE'; fi; python3 -c '$VERIFY_CODE'; python3 -c '$VERIFY_SECURITY_CODE'"; then
            VERIFIED=true
            break
        fi
        sleep 2
    done
    if [ "$VERIFIED" = true ]; then
        echo -e "${GREEN}✓ Services active; fresh state, strict REST, and closed OAuth registration verified${NC}"
    else
        echo -e "${RED}✗ Restart or post-deploy verification failed${NC}"
        echo "  Inspect: systemctl status anima-broker-ex anima-broker anima"
        exit 1
    fi
else
    echo ""
    echo -e "${YELLOW}[3/4] Restart and runtime verification deferred (--no-restart)${NC}"
fi

# Step 4: Show logs (if requested)
if [ "$SHOW_LOGS" = true ]; then
    echo ""
    echo -e "${BLUE}[4/4] Showing logs...${NC}"
    echo -e "${BLUE}=========================================${NC}"
    ssh -p $PI_PORT $SSH_EXTRA -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
        "$PI_USER@$PI_HOST" "journalctl -u anima -u anima-broker -u anima-broker-ex -n 50 --no-pager" || \
        echo -e "${BLUE}ℹ Could not read logs${NC}"
else
    echo ""
    echo -e "${BLUE}[4/4] Done${NC}"
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Deploy complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

if [ "$SHOW_LOGS" = false ]; then
    echo "To view logs, run:"
    echo "  ssh $PI_USER@$PI_HOST 'journalctl -u anima -u anima-broker -u anima-broker-ex -f'"
    echo ""
fi

echo "To check status:"
echo "  ssh $PI_USER@$PI_HOST 'systemctl status anima-broker-ex anima-broker anima'"
echo ""
