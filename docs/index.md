---
title: Anima MCP
---

# Anima MCP

Raspberry Pi 4 sensor testbed for EISV trajectories, autonomous drawing, and persistent identity.

## What Is This?

**Lumen** is the deployed Anima instance: a Raspberry Pi 4 with temperature, light, humidity, pressure, and system telemetry mapped into continuous state dimensions. It maintains persistent identity across restarts, accumulating trajectory history over time. When the interface says "warm," there is a measured sensor/system state behind it.

| Feature | Description |
|---------|-------------|
| **Grounded state** | State labels derived from actual sensor measurements |
| **Persistent identity** | Birth date, awakenings, alive time accumulate across restarts |
| **Autonomous drawing** | Creates art on a 240×240 notepad with pluggable art eras |
| **Learning systems** | Learns preferences, self-model parameters, and action values over time |
| **UNITARES integration** | Governance oversight via MCP |

## Quick Start

```bash
pip install -e ".[pi]"   # On Pi with sensors
pip install -e .        # On Mac with mock sensors

anima --http --host 0.0.0.0 --port 8766
```

Connect your MCP client (Claude Code, Cursor, Claude Desktop) to `http://<pi-ip>:8766/mcp/`

## Documentation

- **[Docs index](developer-index.md)** — For developers and AI agents
- **[Getting started](guides/GETTING_STARTED_SIMPLE.md)** — New to Lumen? Start here
- **[Resonance critique loop](guides/RESONANCE_CRITIQUE_LOOP.md)** — Advisory visual reading path for Resonance-era taste
- **[GitHub repository](https://github.com/cirwel/anima-mcp)** — Source code

---

Built by [@CIRWEL](https://github.com/CIRWEL)
