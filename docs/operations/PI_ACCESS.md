# Anima Pi Access

## Quick SSH
```bash
ssh lumen.local
# or via Tailscale (port 22)
ssh -i ~/.ssh/id_ed25519_pi unitares-anima@<tailscale-ip>
```

## If SSH doesn't work
If `ssh lumen.local` or port 22 times out or is refused, the Pi may be using **port 2222** (e.g. after `fix_ssh_port` was used when 22 was blocked). Try:

```bash
# Tailscale IP, port 2222 (works when 22 is blocked)
ssh -p 2222 -i ~/.ssh/id_ed25519_pi unitares-anima@<tailscale-ip>
```

- **Timeout on 22?** Use the Tailscale IP and/or try `-p 2222`.
- **Once in via 2222:** You can call the MCP tool `fix_ssh_port` with `port: 22` (when connected to Lumen over HTTP) to switch SSH back to port 22.
- More: See troubleshooting sections in `docs/operations/PI_DEPLOYMENT.md` (service, SSH, and network checks).

## Details
- **Hostname**: lumen.local (mDNS) or via Tailscale
- **IP**: 192.168.1.165 (LAN) or <tailscale-ip> (Tailscale — verify with `tailscale status`)
- **Port**: 22 (standard); if blocked, Pi may be on 2222
- **User**: unitares-anima
- **Key**: ~/.ssh/id_ed25519_pi

## SSH Config (~/.ssh/config)
```
Host pi-anima lumen lumen.local
    HostName lumen.local
    User unitares-anima
    Port 22
    IdentityFile ~/.ssh/id_ed25519_pi
    StrictHostKeyChecking no

# Use when port 22 is blocked (Pi on 2222)
Host lumen-2222
    HostName <tailscale-ip>
    User unitares-anima
    Port 2222
    IdentityFile ~/.ssh/id_ed25519_pi
    StrictHostKeyChecking no
```

## Deploy Code
```bash
# Preferred: guarded deploy from clean checkout
cd ~/projects/anima-mcp && ./deploy.sh --host lumen

# Direct rsync (initial setup only; prefer ./deploy.sh so Git revision stays aligned)
rsync -avz -e "ssh -i ~/.ssh/id_ed25519_pi" \
  --exclude='.venv' --exclude='*.db' --exclude='__pycache__' --exclude='.git' \
  ~/projects/anima-mcp/ \
  unitares-anima@lumen.local:/home/unitares-anima/anima-mcp/
```

## Service Management

**Restart MCP server (mind + display/LEDs):**
```bash
ssh lumen.local 'sudo systemctl restart anima.service'
```

**Restart Python hardware/learning broker:**
```bash
ssh lumen.local 'sudo systemctl restart anima-broker.service'
```

**Restart Elixir sensor + governance broker:**
```bash
ssh lumen.local 'sudo systemctl restart anima-broker-ex.service'
```

**Check status (all three services must run):**
```bash
ssh lumen.local 'sudo systemctl status anima-broker-ex.service anima-broker.service anima.service --no-pager'
```

## Check Logs
```bash
# MCP server logs
ssh lumen.local 'tail -50 ~/.anima/mcp.log'

# Creature/broker logs
ssh lumen.local 'tail -50 ~/.anima/creature.log'

# Systemd logs
ssh lumen.local 'journalctl -u anima.service -n 50 --no-pager'
```

## Secrets (API Keys)

Edit `~/.anima/anima.env` on Pi for GROQ_API_KEY, UNITARES_AUTH, and ANIMA_OAUTH_* vars. See `docs/operations/SECRETS_AND_ENV.md`.

## Check Processes
```bash
ssh lumen.local 'ps aux | grep -E "anima|creature" | grep -v grep'
```

## Shared Memory State
```bash
# Current anima state
ssh lumen.local 'cat /dev/shm/anima_state.json | python3 -m json.tool | head -30'
```

---

*Last updated: February 11, 2026*
