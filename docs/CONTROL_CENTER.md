# Lumen Control Center

**Web dashboard for monitoring and interacting with Lumen.**

## Quick Start

1. **Start the message server** (on Mac):
   ```bash
   cd ~/projects/anima-mcp/scripts
   python3 message_server.py
   ```

2. **Open the dashboard**:
   Open `docs/control_center.html` in your browser (or serve it via HTTP for remote access).

## Features

### Live State
- Current anima values (warmth, clarity, stability, presence)
- Mood indicator
- Real-time sensor readings
- CPU-derived computational neural bands (explicitly not physical EEG)
- Voice status (speaking/listening)

### Learning Progress
- Total awakenings
- Time alive (hours)
- State samples (24h)
- Average values (W/C/S/P)
- Stability trend

### Gallery
- Browse Lumen's drawings (stored in `~/.anima/drawings/`)
- Click to enlarge (lightbox)
- Shows 30 most recent, sorted by timestamp
- Auto-refreshes every 2 minutes

### Send Message
- Post messages to Lumen's message board
- Messages appear on Pi display's Visitors screen

### Q&A (Questions & Answers)
- See Lumen's unanswered questions
- Answer questions directly from the dashboard
- **Author field**: Enter your name (blank resolves to the caretaker)
- Answered questions show the answer and author

## Architecture

```
┌─────────────────┐      ┌──────────────────┐ HTTP ┌─────────────┐
│ control_center  │ HTTP │  message_server  │ ───► │ Lumen REST  │
│     .html       │ ───► │      .py         │      │    API      │
└─────────────────┘      └──────────────────┘      └─────────────┘
     Browser              localhost:8771           Pi port 8766
                                  │
                                  └── SSH fallback when HTTP is unavailable
```

> **Port 8771, not 8768.** 8768 is allocated to the UNITARES gateway
> (`com.unitares.gateway-mcp`). See `docs/operations/DEFINITIVE_PORTS.md`.

- **control_center.html**: Static HTML/JS dashboard
- **message_server.py**: Local relay that prefers Lumen's canonical REST API and
  retains SSH as a reduced-capability fallback
- **Pi**: Runs anima services, stores data in `~/.anima/`

The neural bands are computational proprioception: mappings from CPU load, idle
fraction, I/O activity, context switches/interrupts, load variance, and thermal
stability. They are useful signals about Lumen's computing substrate, but they
are not measurements from EEG hardware.

## Endpoints (message_server.py)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/state` | GET | Current `body_anima`, `body_eisv_projection`, sensors, and separately sourced governance state (legacy `anima`/`eisv` aliases retained) |
| `/qa` | GET | List questions (answered and unanswered) |
| `/messages?limit=N` | GET | Recent message-board entries |
| `/answer` | POST | Submit answer to a question |
| `/message` | POST | Post message to Lumen |
| `/learning` | GET | Learning stats from identity database |
| `/growth` | GET | Growth, relationship, and interaction history |
| `/self-knowledge?limit=N` | GET | Lumen's recent self-knowledge entries |
| `/voice` | GET | Voice status (speaking/listening) |
| `/health/detailed` | GET | Detailed subsystem health from Lumen |
| `/gallery` | GET | List of drawings |
| `/gallery/<filename>` | GET | Stream an individual drawing |
| `/health` | GET | Local relay mode and connection configuration |

## Q&A and Learning

When you answer a question via the Control Center:

1. Answer is saved to Pi's question store
2. Knowledge extraction runs (if LLM available)
3. Insight is saved to `~/.anima/knowledge.json` with:
   - Your answer text
   - Author attribution
   - Source question
   - Extracted insight
   - Category (self, world, relationships, etc.)

Lumen uses these insights in future reflections and can reference who taught it what.

## Configuration

Configure the canonical REST bridge with `LUMEN_HTTP_URL`. A local resident proxy
is convenient because the dashboard relay never needs to own the Pi tunnel:

```bash
LUMEN_HTTP_URL=http://127.0.0.1:8769 python3 scripts/message_server.py
```

Set `LUMEN_HTTP_AUTH=user:pass` when the upstream REST API requires Basic auth.
If the HTTP bridge cannot be reached, legacy state, Q&A, voice, learning, and
gallery reads fall back to `unitares-anima@${LUMEN_HOST:-lumen-local}` over SSH.
Growth, self-knowledge, and detailed health require the HTTP bridge. Message and
answer writes use SSH only when no HTTP bridge is configured; an ambiguous HTTP
failure is never retried automatically because that could duplicate a write.

## Troubleshooting

**"Could not load..." errors:**
- Check message_server.py is running
- Open `http://localhost:8771/health` and confirm `mode` is `http`
- Check the configured `LUMEN_HTTP_URL` directly; then check SSH only if using fallback
- Check port (default 8771 — was 8768 until 2026-08-10; a bookmark still pointing
  at 8768 now reaches the UNITARES gateway, so every fetch 404s)

**Gallery not loading:**
- Drawings are in `~/.anima/drawings/` on Pi
- Check the HTTP bridge or, in fallback mode, SSH connectivity

**Q&A not updating:**
- Questions auto-refresh every 30 seconds
- Click refresh button for immediate update

---

*Last updated: August 23, 2026*
