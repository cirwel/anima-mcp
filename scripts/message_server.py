#!/usr/bin/env python3
"""
Lumen Control Server - Bridges the Control Center to Lumen on the Pi.

Endpoints:
  POST /message - Send a message to Lumen
  GET /state    - Get Lumen's current state (anima, identity, sensors)
  GET /qa       - Get questions and answers
  POST /answer  - Answer a question from Lumen

Connection methods (in order of preference):
  1. HTTP to Pi's anima-mcp server (via LUMEN_HTTP_URL env var or Cloudflare tunnel)
  2. SSH fallback (via LUMEN_HOST env var)
"""
import http.server
import socketserver
import json
import subprocess
import os
import base64
import urllib.request
import urllib.error

PORT = 8768
PI_USER = "unitares-anima"
PI_HOST = os.environ.get("LUMEN_HOST", "lumen-local")  # SSH config alias (local network)

# HTTP URL for Pi's anima-mcp server (preferred over SSH)
# Default to Tailscale IP for reliable access from Mac
# DEFINITIVE: anima-mcp runs on port 8766 - see docs/operations/DEFINITIVE_PORTS.md
LUMEN_HTTP_URL = os.environ.get("LUMEN_HTTP_URL", "http://100.79.215.83:8766")
LUMEN_HTTP_AUTH = os.environ.get("LUMEN_HTTP_AUTH", "")  # "user:pass" for basic auth


def http_call_tool(tool_name: str, arguments: dict = None, timeout: int = 10) -> tuple[bool, str]:
    """Call an MCP tool on Pi's anima-mcp server via HTTP."""
    if not LUMEN_HTTP_URL:
        return False, "LUMEN_HTTP_URL not configured"

    url = f"{LUMEN_HTTP_URL.rstrip('/')}/v1/tools/call"
    data = json.dumps({"name": tool_name, "arguments": arguments or {}}).encode()

    headers = {"Content-Type": "application/json"}
    if LUMEN_HTTP_AUTH:
        import base64 as b64
        auth = b64.b64encode(LUMEN_HTTP_AUTH.encode()).decode()
        headers["Authorization"] = f"Basic {auth}"

    req = urllib.request.Request(url, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
            if not result.get("success"):
                return False, result.get("error", "Tool call failed")
            return True, json.dumps(result.get("result", result))
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, f"Connection failed: {e.reason}"
    except Exception as e:
        return False, str(e)


def ssh_command(python_code: str, timeout: int = 10) -> tuple[bool, str]:
    """Run Python code on the Pi via SSH using base64 to avoid escaping issues."""
    encoded = base64.b64encode(python_code.encode()).decode()
    cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", f"{PI_USER}@{PI_HOST}",
        f"cd anima-mcp && echo {encoded} | base64 -d | .venv/bin/python3"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "SSH timeout"
    except Exception as e:
        return False, str(e)


class LumenControlHandler(http.server.SimpleHTTPRequestHandler):

    def send_json(self, data: dict, status: int = 200):
        """Send JSON response with CORS headers."""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == '/state':
            self.handle_get_state()
        elif self.path == '/qa':
            self.handle_get_qa()
        elif self.path == '/messages':
            self.handle_get_messages()
        elif self.path == '/learning':
            self.handle_get_learning()
        elif self.path == '/voice':
            self.handle_get_voice()
        elif self.path == '/gallery':
            self.handle_get_gallery()
        elif self.path.startswith('/gallery/'):
            self.handle_get_gallery_image()
        elif self.path == '/health':
            self.send_json({
                "status": "ok",
                "http_url": LUMEN_HTTP_URL or None,
                "ssh_host": PI_HOST,
                "mode": "http" if LUMEN_HTTP_URL else "ssh"
            })
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/message':
            self.handle_post_message()
        elif self.path == '/answer':
            self.handle_post_answer()
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def handle_get_state(self):
        """Get Lumen's current state via REST endpoint."""
        if LUMEN_HTTP_URL:
            try:
                url = f"{LUMEN_HTTP_URL.rstrip('/')}/state"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    self.send_json(data)
                    return
            except Exception:
                pass  # Fall through to SSH

        # SSH fallback - use the SAME get_state logic as the MCP server
        # This reads from shared memory and uses anima.feeling() for mood
        code = '''
import json
import sys, io

# Suppress init messages
old_stdout, old_stderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = io.StringIO()

try:
    from src.anima_mcp.shared_memory import SharedMemoryClient
    from src.anima_mcp.anima import Anima, SensorReadings
    from src.anima_mcp.identity import IdentityStore

    # Read from shared memory (same source as Pi display)
    shm = SharedMemoryClient()
    shm_data = shm.read()

    sys.stdout, sys.stderr = old_stdout, old_stderr

    if not shm_data or "anima" not in shm_data:
        print(json.dumps({"error": "No shared memory data - is broker running?"}))
    else:
        # Reconstruct anima from shared memory
        a = shm_data["anima"]
        r = shm_data.get("readings", {})

        readings = SensorReadings(
            timestamp=r.get("timestamp", ""),
            cpu_temp_c=r.get("cpu_temp_c"),
            ambient_temp_c=r.get("ambient_temp_c"),
            humidity_pct=r.get("humidity_pct"),
            light_lux=r.get("light_lux"),
            pressure_hpa=r.get("pressure_hpa"),
            cpu_percent=r.get("cpu_percent"),
            memory_percent=r.get("memory_percent"),
            disk_percent=r.get("disk_percent"),
        )

        anima = Anima(
            warmth=a.get("warmth", 0.5),
            clarity=a.get("clarity", 0.5),
            stability=a.get("stability", 0.5),
            presence=a.get("presence", 0.5),
            readings=readings,
        )

        # Use anima.feeling() for consistent mood calculation
        feeling = anima.feeling()

        # Get identity
        store = IdentityStore()
        creature = store.get_identity()

        print(json.dumps({
            "name": (creature.name or "Lumen") if creature else "Lumen",
            "mood": feeling["mood"],
            "warmth": anima.warmth,
            "clarity": anima.clarity,
            "stability": anima.stability,
            "presence": anima.presence,
            "feeling": feeling,
            "cpu_temp": readings.cpu_temp_c or 0,
            "ambient_temp": readings.ambient_temp_c or 0,
            "light": readings.light_lux or 0,
            "humidity": readings.humidity_pct or 0,
            "awakenings": creature.total_awakenings if creature else 0,
            "timestamp": readings.timestamp,
            "source": "shared_memory"
        }))
except Exception as e:
    sys.stdout, sys.stderr = old_stdout, old_stderr
    print(json.dumps({"error": str(e)}))
'''
        success, output = ssh_command(code)
        if success:
            try:
                self.send_json(json.loads(output))
            except json.JSONDecodeError:
                self.send_json({"error": "Invalid JSON from Pi", "raw": output}, 500)
        else:
            self.send_json({"error": output, "offline": True}, 503)

    def handle_get_qa(self):
        """Get questions and answers from Lumen via REST endpoint."""
        if LUMEN_HTTP_URL:
            try:
                url = f"{LUMEN_HTTP_URL.rstrip('/')}/qa"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    self.send_json(data)
                    return
            except Exception:
                pass  # Fall through to SSH

        # SSH fallback
        code = '''
import json
from src.anima_mcp.messages import get_board, MESSAGE_TYPE_QUESTION, MESSAGE_TYPE_AGENT
board = get_board()
board._load()
questions = [m for m in board._messages if m.msg_type == MESSAGE_TYPE_QUESTION]
qa_pairs = []
for q in questions:
    answer = None
    for m in board._messages:
        if getattr(m, "responds_to", None) == q.message_id:
            answer = {"text": m.text, "author": m.author, "timestamp": m.timestamp}
            break
    qa_pairs.append({
        "id": q.message_id,
        "question": q.text,
        "answered": q.answered,
        "timestamp": q.timestamp,
        "answer": answer
    })
qa_pairs.reverse()
print(json.dumps({"questions": qa_pairs[:10], "total": len(qa_pairs), "unanswered": sum(1 for q in qa_pairs if q["answer"] is None)}))
'''
        success, output = ssh_command(code)
        if success:
            try:
                self.send_json(json.loads(output))
            except json.JSONDecodeError:
                self.send_json({"error": "Invalid JSON", "raw": output}, 500)
        else:
            self.send_json({"error": output}, 503)

    def handle_get_messages(self):
        """Get recent messages from Lumen's message board via REST endpoint."""
        if LUMEN_HTTP_URL:
            try:
                url = f"{LUMEN_HTTP_URL.rstrip('/')}/messages"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    self.send_json(data)
                    return
            except Exception:
                pass  # Fall through to SSH

        # SSH fallback
        code = '''
import json
from src.anima_mcp.messages import get_recent_messages
messages = get_recent_messages(20)
result = [{"id": m.message_id, "text": m.text, "type": m.msg_type, "author": m.author, "timestamp": m.timestamp, "responds_to": m.responds_to} for m in messages]
print(json.dumps({"messages": result, "total": len(result)}))
'''
        success, output = ssh_command(code)
        if success:
            try:
                self.send_json(json.loads(output))
            except json.JSONDecodeError:
                self.send_json({"error": "Invalid JSON", "raw": output}, 500)
        else:
            self.send_json({"error": output}, 503)

    def handle_get_learning(self):
        """Get Lumen's learning stats via REST endpoint."""
        # Try REST endpoint first
        if LUMEN_HTTP_URL:
            try:
                url = f"{LUMEN_HTTP_URL.rstrip('/')}/learning"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
                    self.send_json(data)
                    return
            except Exception:
                pass  # Fall through to SSH

        # SSH fallback
        code = '''
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

# Check multiple possible database locations
db_path = None
for p in [Path.home() / "anima-mcp" / "anima.db", Path.home() / ".anima" / "anima.db"]:
    if p.exists():
        db_path = p
        break

if not db_path:
    print(json.dumps({"error": "No identity database"}))
else:
    conn = sqlite3.connect(str(db_path))

    # Get identity stats
    identity = conn.execute("SELECT name, total_awakenings, total_alive_seconds FROM identity LIMIT 1").fetchone()

    # Get recent state history for learning trends
    one_day_ago = (datetime.now() - timedelta(hours=24)).isoformat()
    recent_states = conn.execute(
        "SELECT warmth, clarity, stability, presence, timestamp FROM state_history WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 100",
        (one_day_ago,)
    ).fetchall()

    # Calculate averages and trends
    if recent_states:
        avg_warmth = sum(s[0] for s in recent_states) / len(recent_states)
        avg_clarity = sum(s[1] for s in recent_states) / len(recent_states)
        avg_stability = sum(s[2] for s in recent_states) / len(recent_states)
        avg_presence = sum(s[3] for s in recent_states) / len(recent_states)

        # Trend: compare first half to second half
        mid = len(recent_states) // 2
        if mid > 0:
            first_half = recent_states[mid:]
            second_half = recent_states[:mid]
            stability_trend = sum(s[2] for s in second_half) / len(second_half) - sum(s[2] for s in first_half) / len(first_half)
        else:
            stability_trend = 0
    else:
        avg_warmth = avg_clarity = avg_stability = avg_presence = 0
        stability_trend = 0

    # Get recent events
    events = conn.execute(
        "SELECT event_type, timestamp FROM events ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()

    alive_hours = identity[2] / 3600 if identity else 0

    print(json.dumps({
        "name": identity[0] if identity else "Unknown",
        "awakenings": identity[1] if identity else 0,
        "alive_hours": round(alive_hours, 1),
        "samples_24h": len(recent_states),
        "avg_warmth": round(avg_warmth, 3),
        "avg_clarity": round(avg_clarity, 3),
        "avg_stability": round(avg_stability, 3),
        "avg_presence": round(avg_presence, 3),
        "stability_trend": round(stability_trend, 3),
        "recent_events": [{"type": e[0], "time": e[1]} for e in events[:5]]
    }))
    conn.close()
'''
        success, output = ssh_command(code, timeout=15)
        if success:
            try:
                self.send_json(json.loads(output))
            except json.JSONDecodeError:
                self.send_json({"error": "Invalid JSON", "raw": output}, 500)
        else:
            self.send_json({"error": output}, 503)

    def handle_get_voice(self):
        """Get Lumen's voice/audio status via REST endpoint."""
        # Try REST endpoint first
        if LUMEN_HTTP_URL:
            try:
                url = f"{LUMEN_HTTP_URL.rstrip('/')}/voice"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    self.send_json(data)
                    return
            except Exception:
                pass  # Fall through to SSH

        # SSH fallback
        code = '''
import json
try:
    with open("/dev/shm/anima_voice.json") as f:
        data = json.load(f)
    print(json.dumps(data))
except FileNotFoundError:
    print(json.dumps({"active": False, "status": "no voice data"}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
'''
        success, output = ssh_command(code)
        if success:
            try:
                self.send_json(json.loads(output))
            except json.JSONDecodeError:
                self.send_json({"error": "Invalid JSON", "raw": output}, 500)
        else:
            self.send_json({"error": output}, 503)

    def handle_get_gallery(self):
        """Get list of Lumen's drawings via REST endpoint."""
        # Try REST endpoint first
        if LUMEN_HTTP_URL:
            try:
                url = f"{LUMEN_HTTP_URL.rstrip('/')}/gallery"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    self.send_json(data)
                    return
            except Exception:
                pass  # Fall through to SSH

        # SSH fallback
        code = '''
import json
import re
from pathlib import Path
from datetime import datetime

drawings_dir = Path.home() / ".anima" / "drawings"

if not drawings_dir.exists():
    print(json.dumps({"drawings": [], "total": 0}))
else:
    files = list(drawings_dir.glob("lumen_drawing*.png"))

    def parse_ts(f):
        m = re.search(r"(\d{8})_(\d{6})", f.name)
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
            except Exception:
                pass
        return f.stat().st_mtime

    files = sorted(files, key=parse_ts, reverse=True)

    drawings = []
    for f in files[:30]:
        drawings.append({
            "filename": f.name,
            "timestamp": parse_ts(f),
            "size": f.stat().st_size
        })
    print(json.dumps({"drawings": drawings, "total": len(files)}))
'''
        success, output = ssh_command(code)
        if success:
            try:
                self.send_json(json.loads(output))
            except json.JSONDecodeError:
                self.send_json({"error": "Invalid JSON", "raw": output}, 500)
        else:
            self.send_json({"error": output}, 503)

    def handle_get_gallery_image(self):
        """Serve a drawing image from the Pi via REST endpoint."""
        filename = self.path.split('/gallery/')[-1]
        # Sanitize filename
        if '/' in filename or '..' in filename:
            self.send_response(400)
            self.end_headers()
            return

        # Try REST endpoint first (proxy the image)
        if LUMEN_HTTP_URL:
            try:
                url = f"{LUMEN_HTTP_URL.rstrip('/')}/gallery/{filename}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    img_data = resp.read()
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Cache-Control', 'max-age=3600')
                    self.end_headers()
                    self.wfile.write(img_data)
                    return
            except Exception:
                pass  # Fall through to SSH

        # SSH fallback
        code = f'''
import base64
from pathlib import Path

img_path = Path.home() / ".anima" / "drawings" / "{filename}"
if img_path.exists():
    with open(img_path, "rb") as f:
        print(base64.b64encode(f.read()).decode())
else:
    print("NOT_FOUND")
'''
        success, output = ssh_command(code, timeout=15)
        if success and output != "NOT_FOUND":
            try:
                img_data = base64.b64decode(output)
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'max-age=3600')
                self.end_headers()
                self.wfile.write(img_data)
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()

    def handle_post_message(self):
        """Send a message to Lumen, optionally responding to a question."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data)
            text = data.get('text', '').replace("'", "\\'").replace('"', '\\"')
            author = data.get('author', 'user')
            # Normalize identity: known person aliases → canonical name
            if author.lower() in ('kenny', 'caretaker'):
                author = 'Kenny'  # Canonical person name (normalization happens server-side)
            author = author.replace("'", "\\'").replace('"', '\\"')
            responds_to = data.get('responds_to', '').replace("'", "\\'").replace('"', '\\"')

            if not text:
                self.send_json({"error": "No text provided"}, 400)
                return

            if responds_to:
                # Answering a question - use agent message with responds_to
                print(f"[{author}] Answering question {responds_to}: {text[:50]}...")
                code = f'''
from src.anima_mcp.messages import MessageBoard
board = MessageBoard()
board._load()
# Mark question as answered
for m in board._messages:
    if m.message_id == "{responds_to}":
        m.answered = True
        break
# Add the answer
board.add_agent_message("{text}", agent_name="{author}", responds_to="{responds_to}")
print("ok")
'''
            else:
                # Regular message
                print(f"[{author}] Sending message to Lumen: {text[:50]}...")
                code = f"from src.anima_mcp.messages import MessageBoard; b = MessageBoard(); b.add_user_message('{text}'); print('ok')"

            success, output = ssh_command(code)
            if success:
                self.send_json({"status": "answered" if responds_to else "sent", "responds_to": responds_to or None})
            else:
                self.send_json({"error": output}, 500)

        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_post_answer(self):
        """Answer a question from Lumen."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data)
            question_id = data.get('question_id') or data.get('id', '')  # Accept both
            answer_text = data.get('answer', '').replace("'", "\\'").replace('"', '\\"')
            author = data.get('author', 'Kenny')
            # Normalize identity: known person aliases → canonical name
            if author.lower() in ('kenny', 'caretaker'):
                author = 'Kenny'  # Canonical person name (normalization happens server-side)
            author = author.replace("'", "\\'").replace('"', '\\"')

            if not answer_text:
                self.send_json({"error": "No answer provided"}, 400)
                return

            print(f"[{author}] Answering question {question_id}: {answer_text[:50]}...")
            code = f'''
from src.anima_mcp.messages import MessageBoard
board = MessageBoard()
board._load()
# Find the question and mark as answered
for m in board._messages:
    if m.message_id == "{question_id}":
        m.answered = True
        m.answered_by = "{author}"
        break
# Add the answer
board.add_agent_message("{answer_text}", agent_name="{author}", responds_to="{question_id}")
print("ok")
'''
            success, output = ssh_command(code)
            if success:
                self.send_json({"status": "answered"})
            else:
                self.send_json({"error": output}, 500)

        except Exception as e:
            self.send_json({"error": str(e)}, 500)


def main():
    print("╭──────────────────────────────────────────╮")
    print("│  Lumen Control Server                    │")
    print(f"│  http://localhost:{PORT}                    │")
    print("╰──────────────────────────────────────────╯")
    print()
    if LUMEN_HTTP_URL:
        print("  Mode: HTTP (preferred)")
        print(f"  URL:  {LUMEN_HTTP_URL}")
    else:
        print("  Mode: SSH (fallback)")
    print(f"  SSH:  {PI_USER}@{PI_HOST}")
    print()
    print("Endpoints:")
    print("  GET  /state       - Lumen's current state")
    print("  GET  /qa          - Questions & answers")
    print("  GET  /gallery     - List Lumen's drawings")
    print("  GET  /gallery/<f> - Get drawing image")
    print("  GET  /health      - Connection status")
    print("  POST /message     - Send message to Lumen")
    print("  POST /answer      - Answer Lumen's question")
    print()

    try:
        # Use ThreadingTCPServer to handle concurrent requests
        # (prevents blocking when SSH commands are slow)
        class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads = True

        with ThreadedServer(("", PORT), LumenControlHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
