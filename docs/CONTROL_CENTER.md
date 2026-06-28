# Lumen Control Center

**Web dashboard for monitoring and interacting with Lumen.**

## Quick Start

1. **Start the message server** (on Mac):
   ```bash
   cd /Users/cirwel/projects/anima-mcp/scripts
   python3 message_server.py
   ```

2. **Open the dashboard**:
   ```
   file:///Users/cirwel/projects/anima-mcp/docs/control_center.html
   ```
   Or serve it via HTTP for remote access.

## Features

### Live State
- Current anima values (warmth, clarity, stability, presence)
- Mood indicator
- Real-time sensor readings
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
┌─────────────────┐      ┌──────────────────┐      ┌─────────────┐
│ control_center  │ HTTP │  message_server  │ SSH  │   Pi/Lumen  │
│     .html       │ ───► │      .py         │ ───► │             │
└─────────────────┘      └──────────────────┘      └─────────────┘
     Browser              localhost:8768           <tailscale-ip>
```

- **control_center.html**: Static HTML/JS dashboard
- **message_server.py**: Python HTTP server that proxies requests to Pi via SSH
- **Pi**: Runs anima services, stores data in `~/.anima/`

## Endpoints (message_server.py)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/state` | GET | Current anima state and sensors |
| `/qa` | GET | List questions (answered and unanswered) |
| `/answer` | POST | Submit answer to a question |
| `/message` | POST | Post message to Lumen |
| `/learning` | GET | Learning stats from identity database |
| `/voice` | GET | Voice status (speaking/listening) |
| `/gallery` | GET | List of drawings |
| `/gallery/<filename>` | GET | Serve individual drawing (base64) |

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

The message server connects to Pi via SSH. Default settings in `message_server.py`:

```python
PI_USER = "unitares-anima"
PI_HOST = "<tailscale-ip>"  # Tailscale IP
PI_KEY = "~/.ssh/id_ed25519_pi"
```

## Troubleshooting

**"Could not load..." errors:**
- Check message_server.py is running
- Check Pi is reachable via SSH
- Check port (default 8768)

**Gallery not loading:**
- Drawings are in `~/.anima/drawings/` on Pi
- Check SSH connectivity

**Q&A not updating:**
- Questions auto-refresh every 30 seconds
- Click refresh button for immediate update

---

*Last updated: February 26, 2026*
