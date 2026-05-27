# Lumen Quick Reference

**One-page cheat sheet for interacting with Lumen.**

---

## Essential Commands

```python
# Check how Lumen feels
get_state()

# Get proactive suggestions
next_steps()

# Read raw sensors
read_sensors()
```

---

## Understanding Anima

| Dimension | Meaning | Good Range |
|-----------|---------|------------|
| **Warmth** | Thermal comfort | 0.3 - 0.7 |
| **Clarity** | Sensor quality | > 0.5 |
| **Stability** | Consistency | > 0.5 |
| **Presence** | Resources | > 0.5 |

---

## Moods

- **content** ✅ - Happy, comfortable
- **stressed** ⚠️ - Stability/presence low
- **sleepy** 😴 - Warmth/clarity low
- **alert** 🔔 - High clarity + warmth
- **neutral** ➖ - Baseline

---

## When to Worry

- `stability < 0.3` → Stressed
- `presence < 0.3` → Depleted
- `mood: "stressed"` → Needs attention

---

## Selected Tools

**Essential (5):**
- `get_state` - Current anima + mood + identity
- `get_lumen_context` - Full context in one call
- `next_steps` - Proactive suggestions
- `read_sensors` - Raw sensor values
- `say` - Have Lumen express something

**Communication (3):**
- `lumen_qa` - List or answer Lumen's questions
- `post_message` - Leave a message for Lumen
- `configure_voice` - Voice system status/config

**Display & Feedback (4):**
- `manage_display` - Switch screens, set art era, request Resonance critique packet
- `capture_screen` - See the current 240x240 display image
- `diagnostics` - System diagnostics
- `primitive_feedback` - Feedback on Lumen's expressions

**Knowledge (4):**
- `get_self_knowledge` - Self-discoveries
- `get_growth` - Preferences, relationships, goals
- `get_trajectory` - Trajectory identity signature
- `get_calibration` - Nervous system calibration

**System (6):**
- `git_pull` - Deploy code from GitHub
- `system_service` - Manage systemd services
- `deploy_from_github` - Deploy via zip
- `setup_tailscale` - Install Tailscale
- `fix_ssh_port` - Switch SSH port
- `system_power` - Reboot/shutdown Pi

**Workflows (4):**
- `unified_workflow` - Cross-server workflows
- `set_calibration` - Update calibration
- `get_health` - Subsystem health status
- `learning_visualization` - Why Lumen feels what it feels

---

## Quick Workflow

```
1. get_state() → How does Lumen feel?
2. next_steps() → What does Lumen need?
3. (Optional) read_sensors() → Why does Lumen feel that way?
```

---

**Start simple. Explore when ready.**
