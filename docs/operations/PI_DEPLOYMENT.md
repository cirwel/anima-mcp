# Raspberry Pi Deployment Guide

**Last Updated:** March 14, 2026

---

## Quick Setup (5 Minutes)

### 1. Copy code to Pi

```bash
# From Mac (via git on Pi)
ssh -i ~/.ssh/id_ed25519_pi unitares-anima@lumen.local \
  "cd ~ && git clone <repo-url> anima-mcp && cd anima-mcp"

# Or via rsync
cd ~/projects/anima-mcp
rsync -avz --exclude='.venv' --exclude='*.db' --exclude='__pycache__' --exclude='.git' \
  -e "ssh -i ~/.ssh/id_ed25519_pi" \
  ./ unitares-anima@lumen.local:/home/unitares-anima/anima-mcp/
```

### 2. Setup Python environment

```bash
# On Pi
cd ~/anima-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[pi]"
```

### 3. Install services

```bash
sudo cp systemd/anima-broker.service /etc/systemd/system/
sudo cp systemd/anima.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable anima-broker anima
```

### 4. Start services

```bash
# Broker must start first
sudo systemctl start anima-broker
sudo systemctl start anima
sudo systemctl status anima-broker anima
```

### 5. Verify

```bash
curl http://localhost:8766/health
```

---

## Prerequisites

### Hardware
- Raspberry Pi 4 (recommended) or Pi 3B+
- BrainCraft HAT (optional - for display/LEDs)
- AHT20 sensor (temp/humidity)
- BMP280 sensor (pressure)
- VEML7700 light sensor

### Software
- Raspberry Pi OS (Debian-based)
- Python 3.11+
- SSH access configured
- User account: `unitares-anima`

---

## Full Setup Guide

### Initial Pi Setup

```bash
# On Pi — create user and add to hardware groups
sudo adduser unitares-anima
sudo usermod -aG gpio,i2c,spi unitares-anima

# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv git i2c-tools libopenjp2-7 libgpiod2

# Enable I2C and SPI
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

# Install hardware dependencies (if using BrainCraft HAT)
sudo apt install -y python3-rpi.gpio python3-pil python3-numpy
```

### Python Environment

```bash
cd ~/anima-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[pi]"

# Verify
.venv/bin/anima --help
```

### Systemd Services

```bash
# Copy service files
sudo cp ~/anima-mcp/systemd/anima.service /etc/systemd/system/
sudo cp ~/anima-mcp/systemd/anima-broker.service /etc/systemd/system/

# Verify service file settings:
#   User=unitares-anima
#   WorkingDirectory — correct path
#   ExecStart — correct venv path
#   ANIMA_ID — creature UUID
#   UNITARES_URL — governance URL (if using)
#   ANIMA_GOVERNANCE_INTERVAL_SECONDS — broker heartbeat cadence (default 180)

sudo systemctl daemon-reload
sudo systemctl enable anima-broker anima
sudo systemctl start anima-broker anima
sudo systemctl status anima-broker anima
```

### Health Monitoring (Optional)

```bash
# Copy and test health monitor
cp ~/anima-mcp/scripts/monitor_health_pi.sh ~/monitor_health.sh
chmod +x ~/monitor_health.sh
~/monitor_health.sh --once

# Add to cron (every 5 min)
(crontab -l 2>/dev/null; echo "*/5 * * * * $HOME/monitor_health.sh --once >> /tmp/anima_health.log 2>&1") | crontab -
```

### Network Access (Optional)

**Cloudflare tunnel:** Managed by `cloudflared-lumen.service` (systemd). Routes `lumen.cirwel.org` → `localhost:8766`.

See `SECRETS_AND_ENV.md` and `DEFINITIVE_PORTS.md` for OAuth and port details.

---

## MCP Client Configuration

### Cursor / Claude Code

Edit `~/.cursor/mcp.json` or `~/.claude.json`:

```json
{
  "mcpServers": {
    "anima": {
      "type": "http",
      "url": "http://<tailscale-ip>:8766/mcp/"
    }
  }
}
```

Tailscale (no auth required, verify IP with `tailscale status`). LAN IP (`http://192.168.1.165:8766/mcp/`) also works.

### Claude.ai Web (via Cloudflare tunnel + OAuth 2.1)

- URL: `https://lumen.cirwel.org/mcp/`
- Auth: OAuth 2.1 (PKCE, auto-approve)

Required env vars in `~/.anima/anima.env`:
```bash
ANIMA_OAUTH_ISSUER_URL=https://lumen.cirwel.org
ANIMA_OAUTH_AUTO_APPROVE=true
```

---

## Common Commands

```bash
# Service management
sudo systemctl start anima-broker anima
sudo systemctl stop anima anima-broker
sudo systemctl restart anima-broker anima
sudo systemctl status anima-broker anima

# Logs
sudo journalctl -u anima -f          # Follow MCP server
sudo journalctl -u anima-broker -f   # Follow broker
sudo journalctl -u anima -n 50       # Last 50 lines

# Health
curl http://localhost:8766/health

# Update code (preferred)
git pull && sudo systemctl restart anima-broker anima

# Or via deploy script from Mac
./deploy.sh
./deploy.sh --no-restart    # Deploy without restart
./deploy.sh --host IP       # Override Pi IP
```

---

## Troubleshooting

### Service Won't Start

```bash
sudo systemctl status anima
sudo journalctl -u anima -n 100
# Common: port 8766 in use, missing deps, wrong paths
```

### Display/LEDs Not Working

```bash
ls -la /dev/spi*
groups unitares-anima  # Should include gpio, spi, i2c
cd ~/anima-mcp && source .venv/bin/activate && python scripts/test_display_visual.py
```

### Database Issues

```bash
ls -la ~/.anima/anima.db
chmod 644 ~/.anima/anima.db
```

---

## Security

```bash
# Firewall
sudo ufw allow 22/tcp
sudo ufw allow from 192.168.1.0/24 to any port 8766
sudo ufw enable
```

- Service runs as `unitares-anima` (non-root, no sudo)
- OAuth 2.1 protects `/mcp/` via Cloudflare tunnel; LAN/Tailscale are open
- Tokens are in-memory, reset on restart

---

## Post-Deployment Checklist

- [ ] Service starts automatically on boot
- [ ] Service restarts on failure
- [ ] Health endpoint responds
- [ ] Logs show no errors
- [ ] Display/LEDs updating (if hardware present)
- [ ] Sensors reading correctly (if hardware present)
- [ ] MCP clients can connect
- [ ] Cloudflare tunnel working (if configured)

---

## Related

- `PI_ACCESS.md` — SSH and service access
- `SECRETS_AND_ENV.md` — OAuth, secrets, env vars
- `DEFINITIVE_PORTS.md` — Port conventions
- `BROKER_ARCHITECTURE.md` — Body/mind service architecture
- `BACKUP_AND_RESTORE.md` — Backup, restore, and reflash recovery
