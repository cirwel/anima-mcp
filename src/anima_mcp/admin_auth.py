"""Admin-secret gate for destructive MCP handlers.

Destructive handlers (git_pull, system_power, system_service, fix_ssh_port,
deploy_from_github, setup_tailscale) can reboot the Pi or modify system
state. A typo'd URL or misconfigured agent could trigger them by accident.

Gating: when `ANIMA_ADMIN_SECRET` is set on the server, requests to
destructive handlers must include an `X-Anima-Admin` header matching the
secret. When unset (default), the gate is a no-op.

The ASGI layer reads the header and stashes it in a ContextVar so handlers
can consult it without knowing about the request pipeline.
"""
from __future__ import annotations

import os
from contextvars import ContextVar

from mcp.types import TextContent

_admin_header_value: ContextVar[str | None] = ContextVar(
    "anima_admin_header", default=None
)


def set_admin_header(value: str | None) -> None:
    """Called by the ASGI layer with the raw X-Anima-Admin header value."""
    _admin_header_value.set(value)


def require_admin() -> list[TextContent] | None:
    """Return an error response if the admin secret is required and missing.

    Returns None when the check passes (secret not configured, or header
    matches). Returns a TextContent list when the handler should abort.
    """
    secret = os.environ.get("ANIMA_ADMIN_SECRET")
    if not secret:
        return None
    got = _admin_header_value.get()
    if got and got == secret:
        return None
    return [
        TextContent(
            type="text",
            text=(
                "error: this operation requires the X-Anima-Admin header. "
                "Set it to the value of ANIMA_ADMIN_SECRET on the server."
            ),
        )
    ]
