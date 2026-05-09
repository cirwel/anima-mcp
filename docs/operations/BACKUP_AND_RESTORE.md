# Backup & Restore

**Last Updated:** March 27, 2026

---

## Backup Location

```
~/backups/lumen/
  anima_data/              <- latest rsync mirror of Pi's ~/.anima/
  anima_YYYYMMDD_HHMM.db  <- dated snapshots (keeps last 48)
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

What it does: deploys code, restores DB + JSON + drawings, installs deps, enables I2C/SPI, starts services, installs watchdog + cron, installs Tailscale, updates Mac configs.

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
| `display_brightness.json` | Display brightness config |
| `drawings/` | All saved artwork |

### Identity continuity

- **Same backup restored** → same `creature_id`, events, and growth history (continuity of the identity row and DB).
- **New Pi with empty `~/.anima/`** → a new `creature_id` unless you restore `anima.db` from backup.
- **Copying `anima.db` to a second device** → forks record identity; trajectory and behavior may diverge with environment.

**Behavioral** identity (trajectory signatures, attractor) is documented in the trajectory-identity paper (`cirwel/trajectory-identity-paper`, separate repo) — distinct from UUID continuity.

## Backup Schedule

- **Automated (Mac):** `/Users/cirwel/scripts/backup_lumen.sh` -- twice daily (6am, 6pm) + hourly snapshots
- **Launchd plist:** `~/Library/LaunchAgents/com.unitares.lumen-backup.plist`
- **Log:** `/Users/cirwel/backups/lumen_backup.log`
- **Pi local backup:** `backup_state.sh` runs hourly via crontab, saves JSON state to `~/.anima/backups/state/` (24 snapshots)

---

## DB Integrity Check

If services crash with "database disk image is malformed":
```bash
# Find a clean snapshot
ls -lt ~/backups/lumen/anima_*.db | head -5

# Copy to Pi
scp -i ~/.ssh/id_ed25519_pi \
  ~/backups/lumen/anima_YYYYMMDD_HHMM.db \
  unitares-anima@lumen.local:~/.anima/anima.db

# Clear WAL files and restart
ssh -i ~/.ssh/id_ed25519_pi unitares-anima@lumen.local \
  "rm -f ~/.anima/anima.db-wal ~/.anima/anima.db-shm && \
   sudo systemctl restart anima-broker anima"
```

---

## Secrets After Restore

The restore script copies `anima.env.example` to `~/.anima/anima.env` on the Pi. Edit it to add:
- `GROQ_API_KEY` -- LLM (from groq.com, free)
- `UNITARES_AUTH` -- governance BASIC auth
- `ANIMA_OAUTH_ISSUER_URL` -- Cloudflare tunnel URL (e.g. `https://lumen.cirwel.org`)
- `ANIMA_OAUTH_AUTO_APPROVE=true`

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

To reduce future WiFi drops after reflash:
```bash
chmod +x ~/anima-mcp/scripts/wifi_watchdog.sh
(crontab -l 2>/dev/null; echo "*/5 * * * * $HOME/anima-mcp/scripts/wifi_watchdog.sh") | crontab -
```

---

## Path Reference

| Context | Path | Notes |
|---------|------|-------|
| Systemd (Pi) | `/home/unitares-anima/.anima/anima.db` | Canonical |
| backup_lumen.sh | `~/backups/lumen/anima_data/` | Syncs from Pi |
| Credentials | `scripts/envelope.pi` | Pi password, SSH key (gitignored) |
