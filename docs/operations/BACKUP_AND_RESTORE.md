# Backup & Restore

**Last Updated:** March 27, 2026

---

## Backup Location

```
~/backups/lumen/
  anima_data/              <- latest complete learned-state mirror
  anima_data.previous/     <- prior generation (crash-safe fallback)
  anima_YYYYMMDD_HHMM.db  <- dated snapshots (keeps last 48)
  anima_state_*.tar.gz     <- sanitized learned-state archives
  lumen_recovery_*.tar     <- matching DB + learned-state off-site bundles
  predeploy/<timestamp>/   <- verified DB + learned state before each deploy
```

> **WARNING:** `~/lumen-backups/` is OLD and STALE -- ignore it.

Check latest snapshot:
```bash
ls -lt ~/backups/lumen/anima_*.db | head -5
```

## Restore After Reflash -- One Command

```bash
cd ~/projects/anima-mcp
./scripts/restore_lumen.sh                # auto-detects: lumen.local, 192.168.1.165, Tailscale
./scripts/restore_lumen.sh 192.168.1.165  # explicit IP
```

What it does: verifies and freezes one local recovery generation before any Pi
mutation, stops both state writers, deploys code, atomically restores the DB +
every learned JSON snapshot + the durable learning inbox, restores
drawings, installs dependencies, starts services, and waits for fresh broker
shared memory before declaring success. A missing/corrupt DB aborts instead of
silently creating a new identity.

**Do NOT restore manually step-by-step. Run the script.**

---

## What Gets Backed Up

From Pi's `~/.anima/`:

| File/Dir | Purpose |
|----------|---------|
| `anima.db` | Identity, growth, state history, events (most important) |
| `preferences.json` | Calibration ideals |
| `self_model.json` | Self-model data |
| `knowledge.json` | Learned knowledge |
| `patterns.json` | Adaptive prediction patterns |
| `canvas.json` | Drawing canvas state |
| `messages.json` | Message board |
| `anima_history.json` | Recent anima history for trajectory |
| `metacognition_baselines.json` | Metacognition baselines |
| `anima_config.json` | Adaptive nervous-system calibration |
| `display_brightness.json` | Display brightness config |
| `drawings/` | All saved artwork |
| `learning_inbox/` | Pending/rejected server-to-broker learning evidence |
| `oauth.db` | Registered OAuth clients and token continuity (local only) |
| `journal-archive/` | Hourly gzip exports of the systemd journal (see below) |

The scheduled backup publishes `anima_data/` only after its staging mirror
finishes. A JSON mirror failure makes the run fail and does not advance
`.last_success`; one prior generation remains at `anima_data.previous/`.
An hourly Pi DB snapshot is accepted only while recent and only after an
integrity check; otherwise the Mac asks SQLite to mint a fresh live backup.
The mirror excludes all `*.db*` files (including old corrupt copies) and
`anima.env*` secrets, then captures `oauth.db` separately through SQLite's
online-backup API for local restore. OAuth tokens remain excluded from the
unencrypted off-site artifact. The sanitized learned-state archive and matching
DB are packed into one verified artifact before upload to the private off-site
dataset, avoiding mismatched halves when a network failure interrupts a run.
Deploys independently create a WAL-consistent, integrity-checked bundle under
`predeploy/` and abort if that backup fails unless the operator explicitly uses
`--skip-backup`.

MCP/zip deployments cannot assume the Mac is reachable, so `git_pull` and
`deploy_from_github` create a bounded on-device recovery point at
`~/.anima/backups/predeploy-code/` before replacing any code that will be
restarted. Snapshot failure prevents the update. The one-time bootstrap path
does the same.

### Journal archive

journald on the Pi is volatile (RAM-backed), and high-churn logging rotates it
in under a day — which is why the 2026-03-28 day-summary writer death (#188)
could not be root-caused five months later: no operational record survived.
`scripts/journal-archive.sh` (hourly via `lumen-journal-archive.timer`,
installed by `restore_lumen.sh`, which also adds the service user to `adm` so
the full system journal is readable) exports new journal entries incrementally
(`journalctl --cursor-file` against a scratch cursor, promoted only after the
append is durable — a failed append retries the same window, never skips it)
into `~/.anima/journal-archive/journal-<date>.log.gz`, which the slim mirror
then carries to the Mac daily. Read with `zcat` (multi-member gzip, one member
per hourly run).

Retention: roughly 7 daily files under a 100 MB cap, oldest pruned first —
except today's file, which may be the sole copy the daily mirror has not yet
carried off-host and is never pruned; a storm day can therefore exceed the cap,
loudly. Failures exit non-zero (transiently visible as a failed unit) and drop
a durable `.last_failure` marker in the archive dir, cleared on the next full
success — the marker rides the mirror, so it is greppable from the Mac. After
a reboot the stale cursor falls back to the new journal's head (systemd 257
detects the unmatched cursor), so at most the previous boot's tail since the
last hourly run is lost; a backward clock jump can delay resumption until wall
time passes the saved cursor's timestamp.

### Identity continuity

- **Same backup restored** → same `creature_id`, events, and growth history (continuity of the identity row and DB).
- **New Pi with empty `~/.anima/`** → a new `creature_id` unless you restore `anima.db` from backup.
- **Copying `anima.db` to a second device** → forks record identity; trajectory and behavior may diverge with environment.

### Boot continuity gate

`anima-restore.service` runs once per boot (the marker is under `/run`, not
persistent state). If `anima.db` fails a real SQLite integrity check, or any
authoritative learned-self JSON is missing/invalid, it restores a staged,
verified DB plus learned JSON and the learning inbox before the broker may
start. The DB is published last, so an interrupted repair cannot masquerade as
a complete identity. An unavailable or corrupt backup leaves body and mind
stopped rather than minting a silent replacement identity.

For a deliberately new deployment only, set
`ANIMA_ALLOW_FRESH_START=true` in `~/.anima/anima.env`. Established creatures
should leave it false.

If both Pi and Mac are lost, recover one complete weekday bundle before running
the restore:

```bash
hf download hikewa/lumen-db-backups daily/lumen_recovery_Mon.tar --repo-type dataset
tar -xf lumen_recovery_Mon.tar -C ~/backups/lumen
mkdir -p ~/backups/lumen/anima_data
STATE_ARCHIVE=$(ls -t ~/backups/lumen/anima_state_*.tar.gz | head -1)
tar -xzf "$STATE_ARCHIVE" -C ~/backups/lumen/anima_data
```

**Behavioral** identity (trajectory signatures, attractor) is documented in the trajectory-identity paper (`cirwel/trajectory-identity-paper`, separate repo) — distinct from UUID continuity.

## Backup Schedule

- **Automated (Mac):** `/Users/cirwel/scripts/backup_lumen.sh` -- twice daily (6am, 6pm) + hourly snapshots
- **Launchd plist:** `~/Library/LaunchAgents/com.unitares.lumen-backup.plist`
- **Log:** `/Users/cirwel/backups/lumen_backup.log`
- **Pi local backup:** `backup_state.sh` runs hourly at `:30` via `anima-state-backup.timer`, saves JSON state to `~/.anima/backups/state/` (24 snapshots). `backup_db.sh` (`:15`, `anima-db-backup.timer`) and `db_maintenance.sh` (`:00`, `anima-db-maintenance.timer`) are timers too; none of these is in cron any more (`scripts/install_scheduled_jobs.sh`, run by deploy and restore).
- **Pi storage maintenance:** `anima-storage-maintenance.timer` runs hourly at
  `:45`, after the database (`:15`) and learned-state (`:30`) snapshots.

## Storage pressure policy

Lumen prepares before the filesystem becomes urgent. The hourly maintenance
job writes its last report to
`~/.anima/backups/storage_maintenance_status.json` and uses these policies:

| Filesystem use | State | Self-schema render retention | Operator response |
|---|---|---:|---|
| below 75% | healthy | 14 days, max 512 MiB / 6,000 generations | routine |
| 75–79.9% | warning | 7 days, max 256 MiB / 3,000 generations | investigate growth |
| 80–84.9% | action | 3 days, max 128 MiB / 1,500 generations | reclaim or add capacity |
| 85%+ | urgent | 1 day, max 64 MiB / 500 generations | recover headroom immediately |

Only recognized `schema_YYYYMMDD_HHMMSS.{png,json}` generations are pruned,
oldest first and as whole PNG/JSON generations. Unknown files are never
touched. The 24-hour SQLite rotation, learned JSON snapshots, drawings, live
databases, and ordinary state are outside this cleanup policy.

Exceptional `anima.db.corrupted.*` and `anima.db.pre-restore-*` incidents are
handled separately. The newest remains unpacked. Older incidents may be
losslessly archived only after all of these gates pass:

1. a recent Mac-side recovery run published a verified database + learned-state
   bundle receipt;
2. the latest Pi hourly SQLite backup is recent and passes `quick_check`; and
3. the latest Pi learned-state snapshot is recent, complete, and valid JSON.

Each archive embeds sizes and SHA-256 hashes and is reread before source files
are removed. Local archive count tightens with pressure, but an archive is not
eligible for expiration until a later Mac backup receipt says forensic archives
were mirrored off-device. A missing or stale receipt fails closed: database
artifacts remain untouched.

Readiness drill without changing files:

```bash
python3 scripts/storage_maintenance.py --usage-percent 85
```

The pressure override is deliberately rejected with `--apply`.

---

## DB Integrity Check

If services crash with "database disk image is malformed":
```bash
# Find a clean snapshot
ls -lt ~/backups/lumen/anima_*.db | head -5

# Use the guarded restore workflow; never hot-copy over an open WAL database.
./scripts/restore_lumen.sh lumen.local
```

---

## Secrets After Restore

The restore script copies `anima.env.example` to `~/.anima/anima.env` on the Pi. Edit it to add:
- `GROQ_API_KEY` -- LLM (from groq.com, free)
- `UNITARES_AUTH` -- governance BASIC auth
- `ANIMA_OAUTH_ISSUER_URL` -- Cloudflare tunnel URL (e.g. `https://lumen.cirwel.org`)
- `ANIMA_OAUTH_AUTO_APPROVE=true`
- `ANIMA_HEARTBEAT_URL` -- off-host dead-man's-switch URL; it remains inert
  until reprovisioned and acceptance-tested per `HEARTBEAT.md`

See `SECRETS_AND_ENV.md` for details.

---

## Tailscale After Restore

Tailscale is lost on reflash. After restore completes:
```bash
ssh -i ~/.ssh/id_ed25519_pi unitares-anima@lumen.local \
  "curl -fsSL https://tailscale.com/install.sh | sh"
ssh -i ~/.ssh/id_ed25519_pi unitares-anima@lumen.local \
  "sudo tailscale up"
# Follow the URL to authenticate
```

Or pass `TAILSCALE_AUTH_KEY=tskey-xxx` during `restore_lumen.sh` for auto-auth.

After auth, run `./scripts/update_pi_ip.sh` to update Mac configs with the new Tailscale IP.

---

## Full Reflash Walkthrough

When the Pi's SD card needs a complete reflash (WiFi dead, corrupted OS, etc.).

### Phase 1: Backup (Before Reflash)

**If Pi is reachable:**
```bash
/Users/cirwel/scripts/backup_lumen.sh
```

**If Pi is dead:** Use existing Mac backups at `~/backups/lumen/anima_data/`.

**If SD card accessible but Pi unreachable:** See SD Card Data Recovery below.

### Phase 2: Flash Fresh SD Card

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Flash **Raspberry Pi OS Lite (64-bit)**
3. In advanced options:
   - Hostname: `lumen`
   - Enable SSH (password auth)
   - Username: `unitares-anima`
   - Password: see `scripts/envelope.pi` (copy from `envelope.pi.example`)
   - WiFi: your network SSID and password
   - Set locale/timezone
4. Eject SD card, insert into Pi, power on

### Phase 3: Initial Pi Setup

Wait ~2 minutes for first boot, then:
```bash
ping lumen.local
ssh unitares-anima@lumen.local

# On Pi:
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git i2c-tools libopenjp2-7 libgpiod2
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0
mkdir -p ~/.anima
```

### Phase 4: Restore

```bash
# From Mac
cd /Users/cirwel/projects/anima-mcp
./scripts/restore_lumen.sh
# Or: ./scripts/restore_lumen.sh 192.168.1.165
```

### Phase 5: Verify

```bash
sudo systemctl status anima-broker anima
journalctl -u anima -u anima-broker -f
curl http://localhost:8766/health

# Verify identity
sqlite3 ~/.anima/anima.db "SELECT name, creature_id, born_at FROM identity LIMIT 1;"
# Should show: Lumen, 49e14444-b59e-48f1-83b8-b36a988c9975, 2026-01-11...
```

### Post-Reflash Checklist

| Step | Action |
|------|--------|
| 1 | Backup Pi (if reachable) or confirm Mac backup |
| 2 | Flash SD with Pi OS, hostname `lumen`, user `unitares-anima` |
| 3 | Boot, SSH, apt update, create `~/.anima` |
| 4 | Run `restore_lumen.sh` |
| 5 | Verify identity, display, logs |
| 6 | Install Tailscale, update Mac configs |
| 7 | Edit `~/.anima/anima.env` with secrets |

---

## SD Card Data Recovery

In practice, `~/backups/lumen/` (hourly automated backups) should have recent data. Check there first: `ls -lt ~/backups/lumen/anima_*.db | head -5`.

If you truly need to read the ext4 root partition from the SD card on macOS, there is no reliable tool — macOS cannot mount ext4 natively. Use a Linux machine or boot a Linux USB to mount the card and copy `/home/unitares-anima/.anima/`.

---

## WiFi Watchdog

`restore_lumen.sh` installs it as `wifi-watchdog.timer` (every 2 minutes, as
root). To (re)install it by hand:
```bash
bash ~/anima-mcp/scripts/install_scheduled_jobs.sh
```
Do not also add a crontab line for it — that ran two copies at once.

---

## Path Reference

| Context | Path | Notes |
|---------|------|-------|
| Systemd (Pi) | `/home/unitares-anima/.anima/anima.db` | Canonical |
| backup_lumen.sh | `~/backups/lumen/anima_data/` | Syncs from Pi |
| Credentials | `scripts/envelope.pi` | Pi password, SSH key (gitignored) |
