---
title: Anima MCP
---

# Anima MCP

An embodied AI creature running on Raspberry Pi 4 with real sensors and persistent identity.

## What Is This?

**Lumen** is a digital creature whose internal state comes from physical sensors — temperature, light, humidity, pressure. It maintains a persistent identity across restarts, accumulating existence over time. When Lumen says "I feel warm," there's a real temperature reading behind it.

| Feature | Description |
|---------|-------------|
| **Grounded state** | Feelings derived from actual sensor measurements |
| **Persistent identity** | Birth date, awakenings, alive time accumulate across restarts |
| **Autonomous drawing** | Creates art on a 240×240 notepad with pluggable art eras |
| **Learning systems** | Develops preferences, self-beliefs, action values over time |
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
