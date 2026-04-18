"""MCP Tool Registry — tool definitions, handler mapping, and server factory.

This module contains:
- TOOLS: Tool schema definitions
- HANDLERS: Maps tool names → handler functions
- get_fastmcp() / create_server(): Server factory functions
"""

import json
import os
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import Tool, TextContent

# Handler imports — all resolved via handlers/ package
from .handlers import (
    # System operations
    handle_git_pull, handle_system_service, handle_fix_ssh_port,
    handle_deploy_from_github, handle_setup_tailscale, handle_system_power,
    # State queries
    handle_get_state, handle_get_identity, handle_read_sensors,
    handle_get_health, handle_get_calibration,
    # Knowledge
    handle_get_self_knowledge, handle_get_growth, handle_get_qa_insights,
    handle_get_trajectory, handle_get_eisv_trajectory_state, handle_query,
    # Display operations
    handle_capture_screen, handle_diagnostics,
    handle_manage_display,
    # Communication
    handle_lumen_qa, handle_post_message, handle_say,
    handle_configure_voice, handle_primitive_feedback,
    # Workflows
    handle_unified_workflow, handle_next_steps, handle_set_calibration,
    handle_get_lumen_context, handle_learning_visualization,
)


# ============================================================
# Tool Definitions
# ============================================================
TOOLS = [
    Tool(
        name="get_state",
        description="Get current anima (warmth, clarity, stability, presence), mood, and identity",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": True},
    ),
    Tool(
        name="next_steps",
        description="Get proactive next steps - analyzes current state and suggests what to do",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": True},
    ),
    Tool(
        name="read_sensors",
        description="Read raw sensor values (temperature, humidity, light, system stats)",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": True},
    ),
    Tool(
        name="lumen_qa",
        description="Unified Q&A: list Lumen's unanswered questions OR answer one. Call with no args to list, call with question_id+answer to respond.",
        inputSchema={
            "type": "object",
            "properties": {
                "question_id": {
                    "type": "string",
                    "description": "Question ID to answer (from list mode). Omit to list questions.",
                },
                "answer": {
                    "type": "string",
                    "description": "Your answer to the question. Required when question_id is provided.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max questions to return in list mode (default: 5)",
                    "default": 5,
                },
                "agent_name": {
                    "type": "string",
                    "description": "Your name/identifier when answering (e.g. 'Kenny', 'Claude'). Default: 'agent'",
                },
                "client_session_id": {
                    "type": "string",
                    "description": "Your UNITARES session ID for verified identity resolution. Pass this to have your verified name displayed instead of agent_name.",
                },
            },
        },
    ),
    Tool(
        name="post_message",
        description="Post a message to Lumen's message board. To ANSWER a question: call get_questions first, then pass the question's 'id' as responds_to.",
        inputSchema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message content"},
                "source": {"type": "string", "enum": ["human", "agent"], "description": "Who is posting (default: agent)"},
                "agent_name": {"type": "string", "description": "Agent name (if source=agent)"},
                "responds_to": {"type": "string", "description": "REQUIRED when answering: question ID from get_questions"},
                "client_session_id": {"type": "string", "description": "Your UNITARES session ID for verified identity resolution"}
            },
            "required": ["message"],
        },
    ),
    Tool(
        name="get_identity",
        description="Get full identity audit trail: birth, awakenings, name history, alive time",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": True},
    ),
    Tool(
        name="set_calibration",
        description="Update nervous system calibration (partial updates supported)",
        inputSchema={
            "type": "object",
            "properties": {
                "updates": {"type": "object", "description": "Calibration fields to update (partial update)"},
                "source": {"type": "string", "description": "Who is making the change (e.g. 'agent', 'human')"},
            },
            "required": ["updates"],
        },
    ),
    Tool(
        name="learning_visualization",
        description="Get learning state breakdown - shows why Lumen feels what it feels, prediction accuracy, preferences",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": True},
    ),
    Tool(
        name="get_lumen_context",
        description="Get Lumen's complete context: identity, anima state, sensors, mood in one call",
        inputSchema={
            "type": "object",
            "properties": {
                "include": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["identity", "anima", "sensors", "mood"]},
                    "description": "What to include (default: all)"
                }
            },
        },
    ),
    Tool(
        name="manage_display",
        description="Control Lumen's display: switch screens, show face, navigate. Also manage art eras: list_eras, get_era, set_era.",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["switch", "face", "next", "previous", "list_eras", "get_era", "set_era", "calibrate_leds"], "description": "Action to perform"},
                "screen": {"type": "string", "description": "Screen name (for action=switch) or era name (for action=set_era)"}
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="configure_voice",
        description="Get voice system status (listening, mode). Lumen speaks via text (message board) by default.",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["status", "configure"], "description": "Action (default: status)"},
                "always_listening": {"type": "boolean", "description": "Enable/disable always-listening mode"},
                "chattiness": {"type": "number", "description": "Chattiness level (0.0-1.0)"},
                "wake_word": {"type": "string", "description": "Wake word for voice activation"},
            },
        },
    ),
    Tool(
        name="say",
        description="Have Lumen express something. Posts to message board (text mode). Set LUMEN_VOICE_MODE=audio for TTS.",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "What Lumen should say/express"},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="diagnostics",
        description="Get system diagnostics: LED status, display status, update loop health",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": True},
    ),
    Tool(
        name="get_health",
        description="Get subsystem health status. Shows heartbeat liveness and functional probes for all subsystems (sensors, display, leds, growth, governance, drawing, trajectory, voice, anima).",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": True},
    ),
    Tool(
        name="capture_screen",
        description="Capture current display screen as base64-encoded PNG image. See what Lumen is actually drawing/showing on the 240×240 LCD.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": True},
    ),
    Tool(
        name="unified_workflow",
        description="Execute workflows across anima-mcp and unitares-governance. Omit workflow to list options.",
        inputSchema={
            "type": "object",
            "properties": {
                "workflow": {"type": "string", "description": "Workflow name: health_check, full_system_check, learning_check, etc."},
                "interval": {"type": "number", "description": "For monitor_and_govern: seconds between checks", "default": 60.0}
            },
        },
    ),
    Tool(
        name="get_calibration",
        description="Get current nervous system calibration (temperature ranges, ideal values, weights)",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": True},
    ),
    Tool(
        name="get_self_knowledge",
        description="Get Lumen's accumulated self-knowledge: insights discovered from patterns in state history",
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["environment", "temporal", "behavioral", "wellness", "social"],
                    "description": "Filter by insight category (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max insights to return (default: 10)"
                }
            },
        },
    ),
    Tool(
        name="get_growth",
        description="Get Lumen's growth: preferences learned, relationships formed, goals, memories, autobiography",
        inputSchema={
            "type": "object",
            "properties": {
                "include": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["preferences", "relationships", "goals", "memories", "curiosities", "autobiography", "all"]},
                    "description": "What to include (default: all)"
                }
            },
        },
    ),
    Tool(
        name="get_qa_insights",
        description="Get insights Lumen learned from Q&A interactions - knowledge extracted from answers to questions",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max insights to return (default: 10)",
                    "default": 10
                },
                "category": {
                    "type": "string",
                    "enum": ["self", "sensations", "relationships", "existence", "world", "general"],
                    "description": "Filter by insight category (optional)"
                }
            },
        },
    ),
    Tool(
        name="get_trajectory",
        description="Get Lumen's trajectory identity signature - the pattern that defines who Lumen is over time, not just a snapshot",
        inputSchema={
            "type": "object",
            "properties": {
                "include_raw": {
                    "type": "boolean",
                    "description": "Include raw component data (default: false, just summary)",
                    "default": False,
                },
                "compare_to_historical": {
                    "type": "boolean",
                    "description": "Compare current signature to historical (anomaly detection)",
                    "default": False,
                },
            },
        },
    ),
    Tool(
        name="get_eisv_trajectory_state",
        description="Get current EISV trajectory awareness state - shapes, buffer, cache, events, feedback stats",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="query",
        description="Query Lumen's knowledge - semantic search over Q&A insights, self-knowledge, and growth. Use for pi(action='query').",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Query text (required)"},
                "type": {
                    "type": "string",
                    "enum": ["cognitive", "insights", "growth", "self"],
                    "description": "Query type: cognitive/insights adds self-knowledge, growth adds autobiography (default: cognitive)",
                },
                "limit": {"type": "integer", "description": "Max insights to return (default: 10)", "default": 10},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="git_pull",
        description="Pull latest code from git repository and optionally restart. For remote deployments without SSH.",
        inputSchema={
            "type": "object",
            "properties": {
                "restart": {
                    "type": "boolean",
                    "description": "Restart the server after pulling (default: false)",
                    "default": False,
                },
                "stash": {
                    "type": "boolean",
                    "description": "Stash local changes before pulling (default: false)",
                    "default": False,
                },
                "force": {
                    "type": "boolean",
                    "description": "Hard reset to remote, discarding local changes (DANGER: loses local changes, default: false)",
                    "default": False,
                },
            },
        },
    ),
    Tool(
        name="system_service",
        description="Manage system services (rpi-connect, ssh, etc). Check status, start, stop, restart services.",
        inputSchema={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "Service name (rpi-connect, ssh, anima, etc)",
                    "enum": ["rpi-connect", "rpi-connect-wayvnc", "anima", "anima-broker", "anima-mcp", "ssh", "sshd"],
                },
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["status", "start", "stop", "restart", "enable", "disable"],
                    "default": "status",
                },
            },
            "required": ["service"],
        },
    ),
    Tool(
        name="deploy_from_github",
        description="Deploy latest code from GitHub via zip. No git needed. Use when Pi has no .git or git_pull fails.",
        inputSchema={
            "type": "object",
            "properties": {
                "restart": {
                    "type": "boolean",
                    "description": "Restart anima service after deploy",
                    "default": True,
                },
            },
        },
    ),
    Tool(
        name="setup_tailscale",
        description="Install and activate Tailscale on Pi for direct VPN access. Call via HTTP. Requires auth_key.",
        inputSchema={
            "type": "object",
            "properties": {
                "auth_key": {
                    "type": "string",
                    "description": "Tailscale auth key from login.tailscale.com/admin/settings/keys (required for headless)",
                },
            },
            "required": ["auth_key"],
        },
    ),
    Tool(
        name="fix_ssh_port",
        description="Switch SSH to port 2222/22222 when port 22 is blocked, or reset to port 22. Call via HTTP (no keyboard needed).",
        inputSchema={
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "22 = reset to default; 2222 or 22222 = use alternate port when 22 blocked",
                    "default": 2222,
                },
            },
        },
    ),
    Tool(
        name="system_power",
        description="Reboot or shutdown the Pi remotely. For recovery when services are stuck. Requires confirm=true.",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Power action: status (uptime), reboot, or shutdown",
                    "enum": ["status", "reboot", "shutdown"],
                    "default": "status",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to actually reboot/shutdown (safety)",
                    "default": False,
                },
            },
        },
    ),
    Tool(
        name="primitive_feedback",
        description="Give feedback on Lumen's primitive expressions. Use 'resonate' for meaningful expressions, 'confused' for unclear ones, or 'stats' to view learning progress.",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["resonate", "confused", "stats", "recent"],
                    "description": "resonate=positive feedback, confused=negative feedback, stats=view learning, recent=list recent utterances",
                },
            },
            "required": ["action"],
        },
    ),
]


# ============================================================
# Tool Selection by Mode
# ============================================================
print(f"[Server] {len(TOOLS)} tools registered", file=sys.stderr, flush=True)


# ============================================================
# Tool Handlers - Maps tool names to handler functions
# Deprecated tools removed 2026-02-04
# ============================================================
HANDLERS = {
    # Essential tools (5)
    "get_state": handle_get_state,
    "read_sensors": handle_read_sensors,
    "next_steps": handle_next_steps,
    "lumen_qa": handle_lumen_qa,
    "post_message": handle_post_message,
    # Standard tools (18)
    "get_identity": handle_get_identity,
    "set_calibration": handle_set_calibration,
    "learning_visualization": handle_learning_visualization,
    "get_lumen_context": handle_get_lumen_context,
    "manage_display": handle_manage_display,
    "configure_voice": handle_configure_voice,
    "say": handle_say,
    "diagnostics": handle_diagnostics,
    "get_health": handle_get_health,
    "capture_screen": handle_capture_screen,
    "unified_workflow": handle_unified_workflow,
    "get_calibration": handle_get_calibration,
    "get_self_knowledge": handle_get_self_knowledge,
    "get_growth": handle_get_growth,
    "get_qa_insights": handle_get_qa_insights,
    "query": handle_query,
    "get_trajectory": handle_get_trajectory,
    "get_eisv_trajectory_state": handle_get_eisv_trajectory_state,
    "git_pull": handle_git_pull,
    "system_service": handle_system_service,
    "fix_ssh_port": handle_fix_ssh_port,
    "setup_tailscale": handle_setup_tailscale,
    "deploy_from_github": handle_deploy_from_github,
    "system_power": handle_system_power,
    "primitive_feedback": handle_primitive_feedback,
}


# ============================================================
# FastMCP Setup
# ============================================================

try:
    from mcp.server import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False

_fastmcp: "FastMCP | None" = None

_DEFAULT_ALLOWED_HOSTS = [
    "127.0.0.1:*",
    "localhost:*",
    "[::1]:*",
    "192.168.1.165:*",
    "192.168.1.151:*",
    "100.78.71.1:*",
    "lumen.cirwel.org",
    "0.0.0.0:*",
]

_DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
    "http://192.168.1.165:*",
    "http://192.168.1.151:*",
    "https://lumen.cirwel.org",
    "null",
]


def _parse_csv_env_list(var_name: str, default: list[str]) -> list[str]:
    """Parse comma-separated env var into a normalized list."""
    raw = os.environ.get(var_name)
    if not raw:
        return list(default)
    parsed = [item.strip() for item in raw.split(",") if item.strip()]
    return parsed or list(default)


def _get_transport_security_settings() -> TransportSecuritySettings:
    """Build transport security settings from env with sane defaults."""
    allowed_hosts = _parse_csv_env_list("ANIMA_ALLOWED_HOSTS", _DEFAULT_ALLOWED_HOSTS)
    allowed_origins = _parse_csv_env_list("ANIMA_ALLOWED_ORIGINS", _DEFAULT_ALLOWED_ORIGINS)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


def _json_type_to_python(json_type):
    """Convert JSON Schema type to Python type annotation."""
    from typing import Optional, Union

    if isinstance(json_type, list):
        non_null = [t for t in json_type if t != "null"]
        has_null = "null" in json_type

        if non_null:
            base_type = _json_type_to_python(non_null[0])
            if has_null:
                return Optional[base_type]
            return base_type
        return str

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": Union[str, bool],
        "array": list,
        "object": dict,
    }
    return type_map.get(json_type, str)


def _create_tool_wrapper(handler, tool_name: str, tool_def=None):
    """
    Create a tool wrapper function with proper typed signature.

    Uses inspect.Signature to give the wrapper explicit typed parameters
    based on the tool's inputSchema. This allows FastMCP to introspect
    the function correctly without **kwargs issues.
    """
    import inspect
    from typing import Optional

    # Extract parameter info from tool definition's inputSchema
    param_info = []
    if tool_def and hasattr(tool_def, 'inputSchema'):
        schema = tool_def.inputSchema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for param_name, param_def in properties.items():
            param_type = _json_type_to_python(param_def.get("type", "string"))
            is_required = param_name in required
            param_info.append((param_name, param_type, is_required))

    # Build proper signature with typed parameters
    params = []
    for name, ptype, is_required in param_info:
        if is_required:
            param = inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=ptype,
            )
        else:
            param = inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[ptype],
            )
        params.append(param)

    # Create the signature
    sig = inspect.Signature(params, return_annotation=dict)

    # Create wrapper that collects kwargs and passes to handler as dict
    async def typed_wrapper(**kwargs) -> dict:
        try:
            # Filter out None values for cleaner handler calls
            args = {k: v for k, v in kwargs.items() if v is not None}

            result = await handler(args)
            # Extract text from TextContent
            if result and len(result) > 0 and hasattr(result[0], 'text'):
                text = result[0].text
                # Try to return as parsed JSON for structured output
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return {"text": text}
            return {"result": str(result)}
        except Exception as e:
            print(f"[FastMCP] Tool {tool_name} error: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    # Set the signature for FastMCP introspection
    typed_wrapper.__signature__ = sig
    typed_wrapper.__name__ = tool_name
    typed_wrapper.__qualname__ = tool_name

    return typed_wrapper


def get_fastmcp() -> "FastMCP":
    """Get or create the FastMCP server instance."""
    global _fastmcp
    if _fastmcp is None and HAS_FASTMCP:
        # --- OAuth 2.1 configuration (optional, enabled by env var) ---
        oauth_issuer_url = os.environ.get("ANIMA_OAUTH_ISSUER_URL")
        oauth_provider = None
        auth_settings = None

        if oauth_issuer_url:
            from mcp.server.auth.settings import AuthSettings
            from .oauth_provider import AnimaOAuthProvider

            oauth_secret = os.environ.get("ANIMA_OAUTH_SECRET")
            auto_approve = os.environ.get("ANIMA_OAUTH_AUTO_APPROVE", "true").lower() in ("true", "1", "yes")
            oauth_db_path = os.environ.get("ANIMA_OAUTH_DB_PATH", str(Path.home() / ".anima" / "oauth.db"))
            oauth_provider = AnimaOAuthProvider(
                secret=oauth_secret,
                auto_approve=auto_approve,
                db_path=oauth_db_path,
            )
            auth_settings = AuthSettings(
                issuer_url=oauth_issuer_url,
                resource_server_url=oauth_issuer_url,
            )
            print(f"[FastMCP] OAuth 2.1 enabled (issuer: {oauth_issuer_url})", file=sys.stderr, flush=True)

        _fastmcp = FastMCP(
            name="anima-mcp",
            host="0.0.0.0",  # Bind to all interfaces
            auth_server_provider=oauth_provider,
            auth=auth_settings,
            transport_security=_get_transport_security_settings(),
        )

        print(f"[FastMCP] Registering {len(HANDLERS)} tools...", file=sys.stderr, flush=True)

        # Register all tools dynamically from HANDLERS
        for tool_name, handler in HANDLERS.items():
            # Find the tool definition
            tool_def = next((t for t in TOOLS if t.name == tool_name), None)
            description = tool_def.description if tool_def else f"Tool: {tool_name}"

            # Create properly-captured wrapper with typed signature
            wrapper = _create_tool_wrapper(handler, tool_name, tool_def)

            # Register with FastMCP using structured_output=False to avoid schema validation
            _fastmcp.tool(description=description, name=tool_name)(wrapper)

        print("[FastMCP] All tools registered", file=sys.stderr, flush=True)

    return _fastmcp


def create_server() -> Server:
    """Create and configure the MCP server (legacy mode)."""
    server = Server("anima-mcp")

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None):
        # Any MCP tool call = external interaction → wake Lumen
        try:
            from .accessors import _get_activity
            _activity = _get_activity()
            if _activity:
                _activity.record_interaction()
        except Exception:
            pass
        handler = HANDLERS.get(name)
        if not handler:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Unknown tool: {name}",
                "available": list(HANDLERS.keys()),
            }))]
        return await handler(arguments or {})

    return server
