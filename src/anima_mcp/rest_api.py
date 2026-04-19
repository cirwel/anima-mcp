"""REST API endpoint functions for the Anima HTTP server.

Extracted from server.py's _run_http_server_async() closures.
Each function takes a Starlette Request and returns a Starlette Response.

Server globals are accessed via late imports from .server (same pattern as handlers/).
"""

import ipaddress
import json
import os
import sys
from pathlib import Path

from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)

from .eisv_mapper import anima_to_eisv
from .server_state import extract_neural_bands
from .tool_registry import HANDLERS

# --- Project paths (for serving HTML pages) ---
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# --- Auth helpers ---

# Bearer token for REST endpoints (optional)
_ANIMA_HTTP_API_TOKEN = os.environ.get("ANIMA_HTTP_API_TOKEN")
# Legacy compatibility switch: when true, untrusted requests are allowed if no
# API token is configured. Default false for secure-by-default behavior.
_ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN = os.environ.get(
    "ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN", "false"
).lower() in ("1", "true", "yes", "on")

# Trusted networks: localhost, Tailscale CGNAT, private RFC1918 ranges
_TRUSTED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("100.64.0.0/10"),   # Tailscale CGNAT
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
]


def _parse_networks(raw: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Parse comma-separated CIDR blocks into IP network objects."""
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for item in (raw or "").split(","):
        cidr = item.strip()
        if not cidr:
            continue
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            continue
    return networks


# Optional: only trust X-Forwarded-For when the immediate peer is in this set.
# Example: ANIMA_TRUSTED_PROXY_NETWORKS="127.0.0.1/32,::1/128"
_TRUSTED_PROXY_NETWORKS = _parse_networks(os.environ.get("ANIMA_TRUSTED_PROXY_NETWORKS", ""))


def _is_trusted_network(request) -> bool:
    """Check if request originates from a trusted network."""
    peer_ip = request.client.host if request.client else None
    if not peer_ip:
        return False
    try:
        client_addr = ipaddress.ip_address(peer_ip)
    except ValueError:
        return False

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and _TRUSTED_PROXY_NETWORKS and any(client_addr in net for net in _TRUSTED_PROXY_NETWORKS):
        candidate = forwarded.split(",")[0].strip()
        try:
            client_addr = ipaddress.ip_address(candidate)
        except ValueError:
            return False

    try:
        return any(client_addr in net for net in _TRUSTED_NETWORKS)
    except ValueError:
        return False


def _check_rest_auth(request) -> bool:
    """Bearer token auth for REST endpoints. Trusted networks bypass auth."""
    if _is_trusted_network(request):
        return True
    if not _ANIMA_HTTP_API_TOKEN:
        return _ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN
    # Allow requests with valid bearer token
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not isinstance(auth, str):
        return False
    if not auth.lower().startswith("bearer "):
        return False
    token = auth.split(" ", 1)[1].strip()
    return token == _ANIMA_HTTP_API_TOKEN


def _require_rest_auth(request, *, success_shape: bool = False):
    """Return unauthorized response when auth fails, otherwise None."""
    if _check_rest_auth(request):
        return None
    if success_shape:
        return JSONResponse({"success": False, "error": "Unauthorized"}, status_code=401)
    return JSONResponse({"error": "Unauthorized"}, status_code=401)


# --- HTML page helpers ---

def _serve_html_page(filename: str, label: str):
    """Serve an HTML file from docs/ or return 404."""
    page_path = _PROJECT_ROOT / "docs" / filename
    if page_path.exists():
        return FileResponse(page_path, media_type="text/html")
    return HTMLResponse(
        f"<html><body><h1>{label} Not Found</h1>"
        f"<p>Expected at: {page_path}</p></body></html>",
        status_code=404,
    )


# --- Endpoint functions ---

async def health_check(request):
    """Health check -- always public (monitoring, load balancers)."""
    from .server import SERVER_READY

    status = "ok" if SERVER_READY else "starting"
    return PlainTextResponse(f"{status}\n")


async def rest_tool_call(request):
    """REST API for calling MCP tools directly.

    POST /v1/tools/call
    Body: {"name": "tool_name", "arguments": {...}}
    Returns: {"success": true, "result": ...} or {"success": false, "error": "..."}
    """
    auth_error = _require_rest_auth(request, success_shape=True)
    if auth_error:
        return auth_error
    try:
        body = await request.json()
        tool_name = body.get("name")
        arguments = body.get("arguments", {})

        if not tool_name:
            return JSONResponse({"success": False, "error": "Missing 'name' field"}, status_code=400)

        if tool_name not in HANDLERS:
            return JSONResponse({"success": False, "error": f"Unknown tool: {tool_name}"}, status_code=404)

        # Call the tool handler
        handler = HANDLERS[tool_name]
        result = await handler(arguments)

        # Extract text from TextContent
        if result and len(result) > 0:
            text_result = result[0].text
            try:
                # Try to parse as JSON for cleaner response
                parsed = json.loads(text_result)
                return JSONResponse({"success": True, "result": parsed})
            except json.JSONDecodeError:
                return JSONResponse({"success": True, "result": text_result})

        return JSONResponse({"success": True, "result": None})

    except Exception as e:
        print(f"[REST API] Error: {e}", file=sys.stderr, flush=True)
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


async def dashboard(request):
    """Serve the Lumen Control Center dashboard."""
    return _serve_html_page("control_center.html", "Dashboard")


async def rest_state(request):
    """GET /state - Format matching message_server.py."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        from datetime import datetime as _dt
        from .server import SHM_GOVERNANCE_STALE_SECONDS
        from .accessors import (
            _get_readings_and_anima,
            _get_store,
            _get_last_governance_decision,
            _get_activity,
        )

        # Use internal functions (same as MCP get_state)
        readings, anima = _get_readings_and_anima()
        if readings is None or anima is None:
            return JSONResponse({"error": "Unable to read sensor data"}, status_code=500)

        feeling = anima.feeling()
        store = _get_store()
        identity = store.get_identity() if store else None

        # Build neural bands from raw sensor data
        neural = extract_neural_bands(readings)

        # EISV
        eisv = anima_to_eisv(anima, readings)

        # Governance
        gov = _get_last_governance_decision() or {}
        gov_timestamp = gov.get("governance_at")
        gov_age_seconds = None
        gov_fresh = None
        if gov_timestamp:
            try:
                gov_age_seconds = max(0, int((_dt.now().timestamp()) - _dt.fromisoformat(gov_timestamp).timestamp()))
                gov_fresh = gov_age_seconds <= SHM_GOVERNANCE_STALE_SECONDS
            except (ValueError, TypeError):
                gov_timestamp = None
                gov_age_seconds = None
                gov_fresh = None

        token_configured = bool(_ANIMA_HTTP_API_TOKEN)
        allow_no_token = bool(_ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN)
        if token_configured:
            auth_mode = "token"
        elif allow_no_token:
            auth_mode = "permissive-no-token"
        else:
            auth_mode = "strict-no-token"

        return JSONResponse({
            "name": identity.name if identity else "Lumen",
            "mood": feeling["mood"],
            "warmth": anima.warmth,
            "clarity": anima.clarity,
            "stability": anima.stability,
            "presence": anima.presence,
            "feeling": feeling,
            "surprise": 0,
            "cpu_temp": readings.cpu_temp_c or 0,
            "ambient_temp": readings.ambient_temp_c or 0,
            "light": readings.light_lux or 0,
            "humidity": readings.humidity_pct or 0,
            "pressure": readings.pressure_hpa,
            "cpu_percent": readings.cpu_percent or 0,
            "memory_percent": readings.memory_percent or 0,
            "disk_percent": readings.disk_percent or 0,
            "neural": neural,
            "eisv": eisv.to_dict(),
            "governance": {
                "decision": gov.get("action", "unknown").upper() if gov else "OFFLINE",
                "margin": gov.get("margin", "") if gov else "",
                "source": gov.get("source", "") if gov else "",
                "connected": bool(gov),
                "timestamp": gov_timestamp,
                "age_seconds": gov_age_seconds,
                "fresh": gov_fresh,
                "path": "broker_shm" if gov_timestamp else ("server_fallback" if gov else ""),
            },
            "api_security": {
                "mode": auth_mode,
                "token_configured": token_configured,
                "allow_unauth_if_no_token": allow_no_token,
                "trusted_proxy_networks_configured": bool(_TRUSTED_PROXY_NETWORKS),
            },
            "awakenings": identity.total_awakenings if identity else 0,
            "alive_hours": round((identity.total_alive_seconds + store.get_session_alive_seconds()) / 3600, 1) if identity and store else 0,
            "alive_ratio": round(identity.alive_ratio(), 2) if identity else 0,
            "activity": {
                **(_get_activity().get_status() if _get_activity() else {"level": "active"}),
                "sleep": _get_activity().get_sleep_summary() if _get_activity() else {"sessions": 0},
            },
            "timestamp": str(readings.timestamp) if readings.timestamp else "",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_qa(request):
    """GET /qa - Get questions and answers (matching message_server.py format)."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        from .messages import get_board, MESSAGE_TYPE_QUESTION

        board = get_board()
        board._load(force=True)

        # Get all questions
        questions = [m for m in board._messages if m.msg_type == MESSAGE_TYPE_QUESTION]

        # Build Q&A pairs with answers
        qa_pairs = []
        for q in questions:
            # Find answer for this question
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

        # Count truly unanswered (no actual answer message) from ALL questions
        truly_unanswered = sum(1 for q in qa_pairs if q["answer"] is None)

        # Reverse to show newest first
        limit = int(request.query_params.get("limit", "20"))
        limit = min(limit, 50)
        qa_pairs.reverse()
        qa_pairs = qa_pairs[:limit]

        return JSONResponse({"questions": qa_pairs, "total": len(questions), "unanswered": truly_unanswered})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_messages(request):
    """GET /messages - Get recent message board entries."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        from .messages import get_board, get_recent_messages
        limit = int(request.query_params.get("limit", "20"))
        limit = min(limit, 100)
        messages = get_recent_messages(limit)
        board = get_board()
        return JSONResponse({
            "messages": [
                {
                    "id": m.message_id,
                    "text": m.text,
                    "type": m.msg_type,
                    "author": m.author,
                    "timestamp": m.timestamp,
                    "responds_to": m.responds_to,
                }
                for m in messages
            ],
            "total": len(board._messages),
            "returned": len(messages),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_answer(request):
    """POST /answer - Answer a question from Lumen."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        from .handlers.communication import handle_lumen_qa
        from .growth import normalize_visitor_identity

        body = await request.json()
        question_id = body.get("question_id") or body.get("id")
        answer = body.get("answer")
        author = body.get("author", "Kenny")
        # Normalize identity: dashboard interactions resolve to person
        _, display_name, _ = normalize_visitor_identity(author, source="dashboard")
        result = await handle_lumen_qa({
            "question_id": question_id,
            "answer": answer,
            "agent_name": display_name
        })
        if result and len(result) > 0:
            data = json.loads(result[0].text)
            return JSONResponse(data)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_message(request):
    """POST /message - Send a message to Lumen."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        from .handlers.communication import handle_post_message
        from .growth import normalize_visitor_identity

        body = await request.json()
        message = body.get("message", body.get("text", ""))
        author = body.get("author", "dashboard")
        # Normalize identity: dashboard interactions resolve to person
        _, display_name, _ = normalize_visitor_identity(author, source="dashboard")
        responds_to = body.get("responds_to")
        payload = {"message": message, "source": "dashboard", "agent_name": display_name}
        if responds_to:
            payload["responds_to"] = responds_to
        result = await handle_post_message(payload)
        if result and len(result) > 0:
            data = json.loads(result[0].text)
            return JSONResponse(data)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_learning(request):
    """GET /learning - Exact copy of message_server.py format."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        import sqlite3
        from datetime import datetime, timedelta

        # Find database - prefer ANIMA_DB env var, then ~/.anima/
        db_path = None
        env_db = os.environ.get("ANIMA_DB")
        candidates = [Path(env_db)] if env_db else []
        candidates.extend([Path.home() / ".anima" / "anima.db", Path.home() / "anima-mcp" / "anima.db"])
        for p in candidates:
            if p.exists():
                db_path = p
                break

        if not db_path:
            return JSONResponse({"error": "No identity database"}, status_code=500)

        conn = sqlite3.connect(str(db_path))

        # Get identity stats
        identity = conn.execute("SELECT name, total_awakenings, total_alive_seconds, born_at FROM identity LIMIT 1").fetchone()

        # Get recent state history for learning trends
        one_day_ago = (datetime.now() - timedelta(hours=24)).isoformat()

        # Real count (no limit)
        sample_count_24h = conn.execute(
            "SELECT COUNT(*) FROM state_history WHERE timestamp > ?",
            (one_day_ago,)
        ).fetchone()[0]

        # Averages via SQL (all samples, not capped)
        avgs = conn.execute(
            "SELECT AVG(warmth), AVG(clarity), AVG(stability), AVG(presence) FROM state_history WHERE timestamp > ?",
            (one_day_ago,)
        ).fetchone()
        avg_warmth = avgs[0] or 0
        avg_clarity = avgs[1] or 0
        avg_stability = avgs[2] or 0
        avg_presence = avgs[3] or 0

        # Stability trend: compare first half vs second half of 24h window
        twelve_hours_ago = (datetime.now() - timedelta(hours=12)).isoformat()
        older_avg = conn.execute(
            "SELECT AVG(stability) FROM state_history WHERE timestamp > ? AND timestamp <= ?",
            (one_day_ago, twelve_hours_ago)
        ).fetchone()[0]
        newer_avg = conn.execute(
            "SELECT AVG(stability) FROM state_history WHERE timestamp > ?",
            (twelve_hours_ago,)
        ).fetchone()[0]
        stability_trend = (newer_avg or 0) - (older_avg or 0) if older_avg else 0

        alive_hours = identity[2] / 3600 if identity else 0
        age_days = 0
        if identity and identity[3]:
            try:
                born = datetime.fromisoformat(identity[3])
                age_days = (datetime.now() - born).days
            except Exception:
                pass
        conn.close()

        return JSONResponse({
            "name": identity[0] if identity else "Unknown",
            "awakenings": identity[1] if identity else 0,
            "age_days": age_days,
            "alive_hours": round(alive_hours, 1),
            "samples_24h": sample_count_24h,
            "avg_warmth": round(avg_warmth, 3),
            "avg_clarity": round(avg_clarity, 3),
            "avg_stability": round(avg_stability, 3),
            "avg_presence": round(avg_presence, 3),
            "stability_trend": round(stability_trend, 3),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_voice(request):
    """GET /voice - Get voice system status."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        from .handlers.communication import handle_configure_voice

        result = await handle_configure_voice({"action": "status"})
        if result and len(result) > 0:
            data = json.loads(result[0].text)
            return JSONResponse(data)
        return JSONResponse({"mode": "text"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_gallery(request):
    """GET /gallery - Get Lumen's drawings."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        import re
        from datetime import datetime as dt
        drawings_dir = Path.home() / ".anima" / "drawings"

        if not drawings_dir.exists():
            return JSONResponse({"drawings": [], "total": 0})

        files = [f for f in drawings_dir.glob("lumen_drawing*.png") if f.stat().st_size > 0]

        def get_era(filename):
            """Determine which era a drawing belongs to."""
            # New format: lumen_drawing_YYYYMMDD_HHMMSS_eraname[_manual].png
            m = re.match(r"lumen_drawing_\d{8}_\d{6}_([a-z]+)(?:_manual)?\.png", filename)
            if m:
                return m.group(1)
            # Legacy: timestamp-based for pre-era-tag drawings
            ts_m = re.search(r"(\d{8}_\d{6})", filename)
            if ts_m and ts_m.group(1) < "20260207_190000":
                return "geometric"
            return "gestural"  # legacy default for untagged gestural-era drawings

        def parse_ts(f):
            m = re.search(r"(\d{8})_(\d{6})", f.name)
            if m:
                try:
                    return dt.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
                except ValueError:
                    pass
            return f.stat().st_mtime

        files = sorted(files, key=parse_ts, reverse=True)

        # Pagination support
        offset = int(request.query_params.get("offset", 0))
        limit = int(request.query_params.get("limit", 50))
        limit = min(limit, 100)  # cap at 100 per request

        page_files = files[offset:offset + limit]

        drawings = []
        for f in page_files:
            drawings.append({
                "filename": f.name,
                "timestamp": parse_ts(f),
                "size": f.stat().st_size,
                "manual": "_manual" in f.name,
                "era": get_era(f.name),
            })

        return JSONResponse({
            "drawings": drawings,
            "total": len(files),
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < len(files),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_gallery_image(request):
    """GET /gallery/{filename} - Serve a drawing image."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    filename = request.path_params.get("filename", "")
    # Sanitize filename
    if "/" in filename or ".." in filename or not filename.endswith(".png"):
        return Response(content="Bad request", status_code=400)
    img_path = Path.home() / ".anima" / "drawings" / filename
    if not img_path.exists():
        return Response(content="Not found", status_code=404)
    try:
        with open(img_path, "rb") as f:
            img_data = f.read()
        return Response(
            content=img_data,
            media_type="image/png",
            headers={"Cache-Control": "max-age=3600"}
        )
    except Exception as e:
        return Response(content=str(e), status_code=500)


async def rest_health_detailed(request):
    """GET /health/detailed - Get subsystem health status."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        from .handlers.state_queries import handle_get_health

        result = await handle_get_health({})
        if result and len(result) > 0:
            data = json.loads(result[0].text)
            return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"error": "no data"}, status_code=500)


async def rest_self_knowledge(request):
    """GET /self-knowledge - Get Lumen's accumulated self-knowledge insights."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        from .handlers.knowledge import handle_get_self_knowledge

        category = request.query_params.get("category")
        limit = int(request.query_params.get("limit", "50"))
        result = await handle_get_self_knowledge({"category": category, "limit": limit})
        if result and len(result) > 0:
            data = json.loads(result[0].text)
            return JSONResponse(data)
        return JSONResponse({"error": "no data"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_growth(request):
    """GET /growth - Get Lumen's growth data (autobiography, goals, memories, preferences)."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        from .handlers.knowledge import handle_get_growth

        result = await handle_get_growth({"include": ["all"]})
        if result and len(result) > 0:
            data = json.loads(result[0].text)
            return JSONResponse(data)
        return JSONResponse({"error": "no data"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_gallery_page(request):
    """Serve the Lumen Drawing Gallery page."""
    return _serve_html_page("gallery.html", "Gallery")


async def rest_layers(request):
    """GET /layers - Full proprioception stack for architecture page."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        from .accessors import (
            _get_readings_and_anima, _get_store, _get_last_governance_decision,
            _get_schema_hub,
        )

        readings, anima = _get_readings_and_anima()
        if readings is None or anima is None:
            return JSONResponse({"error": "Unable to read sensor data"}, status_code=500)

        feeling = anima.feeling()
        store = _get_store()
        identity = store.get_identity() if store else None

        # Physical sensors
        physical = {
            "ambient_temp_c": readings.ambient_temp_c or 0,
            "humidity_pct": readings.humidity_pct or 0,
            "light_lux": readings.light_lux or 0,
            "pressure_hpa": readings.pressure_hpa,
        }

        # Neural bands
        neural = extract_neural_bands(readings)

        # Anima
        anima_data = {
            "warmth": round(anima.warmth, 3),
            "clarity": round(anima.clarity, 3),
            "stability": round(anima.stability, 3),
            "presence": round(anima.presence, 3),
        }

        # EISV
        eisv = anima_to_eisv(anima, readings)
        eisv_data = eisv.to_dict()

        # Governance
        gov = _get_last_governance_decision() or {}
        governance_data = {
            "decision": gov.get("action", "unknown").upper() if gov else "OFFLINE",
            "margin": gov.get("margin", "unknown") if gov else "n/a",
            "source": gov.get("source", "") if gov else "",
            "connected": bool(gov),
        }
        if gov and gov.get("eisv"):
            governance_data["eisv"] = gov["eisv"]

        # System
        system = {
            "cpu_temp_c": readings.cpu_temp_c or 0,
            "cpu_percent": readings.cpu_percent or 0,
            "memory_percent": readings.memory_percent or 0,
            "disk_percent": readings.disk_percent or 0,
        }

        # Identity
        identity_data = {}
        if identity:
            alive_seconds = identity.total_alive_seconds + (store.get_session_alive_seconds() if store else 0)
            identity_data = {
                "name": identity.name,
                "awakenings": identity.total_awakenings,
                "alive_hours": round(alive_seconds / 3600, 1),
                "alive_ratio": round(identity.alive_ratio(), 3),
                "age_days": round(identity.age_seconds() / 86400, 1),
            }

        # Schema Hub - trajectory and circulation data
        schema_hub_data = {}
        try:
            hub = _get_schema_hub()
            schema_hub_data = {
                "history_size": len(hub.schema_history),
                "history_max": hub.history_size,
                "has_trajectory": hub.last_trajectory is not None,
            }
            if hub.last_trajectory:
                traj = hub.last_trajectory
                schema_hub_data["trajectory"] = {
                    "observation_count": traj.observation_count,
                    "identity_maturity": round(min(1.0, traj.observation_count / 50), 3),
                }
                if traj.attractor and traj.attractor.get("center"):
                    center = traj.attractor["center"]
                    schema_hub_data["trajectory"]["attractor_magnitude"] = round(sum(center) / 4, 3)
                if traj.attractor and traj.attractor.get("variance"):
                    variance = traj.attractor["variance"]
                    schema_hub_data["trajectory"]["stability"] = round(max(0, 1 - sum(variance) * 10), 3)
            if hub.last_gap_delta:
                schema_hub_data["gap"] = {
                    "duration_hours": round(hub.last_gap_delta.duration_seconds / 3600, 2),
                    "was_gap": hub.last_gap_delta.was_gap,
                }
        except Exception:
            pass

        return JSONResponse({
            "physical": physical,
            "neural": neural,
            "anima": anima_data,
            "feeling": feeling,
            "eisv": eisv_data,
            "governance": governance_data,
            "system": system,
            "identity": identity_data,
            "schema_hub": schema_hub_data,
            "mood": feeling.get("mood", "unknown"),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_architecture_page(request):
    """Serve the Lumen Architecture page."""
    return _serve_html_page("architecture.html", "Architecture Page")


async def rest_schema_data(request):
    """Return full self-schema graph, trajectory, and history."""
    auth_error = _require_rest_auth(request)
    if auth_error:
        return auth_error
    try:
        from .accessors import _get_schema_hub, _get_store, _get_readings_and_anima, _get_growth

        hub = _get_schema_hub()

        # Single source of truth: hub.schema_history (seeded on wake)
        schema = hub.schema_history[-1].to_dict() if hub.schema_history else None

        # Fallback when hub has no history yet (same as LCD screen fallback)
        if schema is None:
            try:
                from .self_schema import get_current_schema
                from .self_model import get_self_model
                store = _get_store()
                identity = store.get_identity() if store else None
                readings, anima = _get_readings_and_anima()
                schema = get_current_schema(
                    identity=identity,
                    anima=anima,
                    readings=readings,
                    growth_system=_get_growth(),
                    include_preferences=True,
                    self_model=get_self_model(),
                ).to_dict()
            except Exception:
                pass

        # Trajectory with component detail
        trajectory = None
        if hub.last_trajectory:
            traj = hub.last_trajectory
            trajectory = traj.summary()
            trajectory["preferences_detail"] = traj.preferences
            trajectory["beliefs_detail"] = traj.beliefs
            trajectory["attractor_detail"] = traj.attractor
            trajectory["recovery_detail"] = traj.recovery
            trajectory["relational_detail"] = traj.relational

        # Condensed history
        history = [{
            "timestamp": s.timestamp.isoformat(),
            "node_count": len(s.nodes),
            "edge_count": len(s.edges),
        } for s in hub.schema_history]

        # Gap info
        gap = None
        if hub.last_gap_delta:
            g = hub.last_gap_delta
            gap = {
                "duration_hours": round(g.duration_seconds / 3600, 2),
                "was_gap": g.was_gap,
                "was_restore": g.was_restore,
                "anima_delta": g.anima_delta,
                "beliefs_decayed": g.beliefs_decayed,
            }

        return JSONResponse({
            "schema": schema,
            "trajectory": trajectory,
            "history": history,
            "gap": gap,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def rest_schema_page(request):
    """Serve the Self-Schema visualization page."""
    return _serve_html_page("schema.html", "Schema Page")
