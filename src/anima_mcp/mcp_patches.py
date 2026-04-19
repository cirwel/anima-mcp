"""Runtime patches for upstream MCP SDK bugs that surface under Lumen's load.

Keep this module very small and very justified. Each patch documents:
  - Which upstream version exhibited the bug
  - The exact traceback we saw
  - Why the workaround is safe
  - When to delete the patch (upstream fix lands)

Applied once at server startup via ``apply_all_patches()``.
"""
from __future__ import annotations

import sys
from typing import Any


_patches_applied = False


def _wrap_swallow_closed(original: Any) -> Any:
    """Wrap *original* so that ``anyio.ClosedResourceError`` is swallowed.

    Broken out from the patch application for direct unit testing.
    """
    import anyio

    async def safe(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await original(self, *args, **kwargs)
        except anyio.ClosedResourceError:
            # Client went away mid-response. Log at stderr so we can still
            # see how often this fires, but don't propagate — the session
            # would crash and upstream MCP has no mechanism to recover.
            print(
                "[mcp_patches] Dropped notification to closed stream "
                "(client disconnected)",
                file=sys.stderr,
                flush=True,
            )

    safe._anima_closed_stream_patch = True  # type: ignore[attr-defined]
    return safe


def _patch_send_notification_swallows_closed_stream() -> None:
    """MCP 1.26.0 crashes a whole session if a client disconnects while the
    server is trying to send a log-notification.

    Observed on Lumen 2026-04-19 (``Stateless session crashed``):

        File "mcp/server/lowlevel/server.py", line 701, in _handle_message
            await session.send_log_message(...)
        File "mcp/server/session.py", line 213, in send_log_message
            await self.send_notification(...)
        File "mcp/shared/session.py", line 335, in send_notification
            await self._write_stream.send(session_message)
        File "anyio/streams/memory.py", line 218, in send_nowait
            raise ClosedResourceError
        anyio.ClosedResourceError

    The notification was purely informational (server → client log). Dropping
    it when the client is already gone is harmless and prevents the session
    crash + noisy traceback. Rate observed: ~3/24h under normal load.

    Delete this patch once MCP SDK upstream catches ``ClosedResourceError``
    in ``send_notification`` — track https://github.com/modelcontextprotocol/python-sdk
    """
    try:
        from mcp.shared import session as mcp_session
    except ImportError:
        return

    BaseSession = getattr(mcp_session, "BaseSession", None)
    if BaseSession is None:
        return

    original = BaseSession.send_notification
    if getattr(original, "_anima_closed_stream_patch", False):
        return  # idempotent — already wrapped

    BaseSession.send_notification = _wrap_swallow_closed(original)


def apply_all_patches() -> None:
    """Apply every runtime patch. Safe to call multiple times."""
    global _patches_applied
    if _patches_applied:
        return
    _patch_send_notification_swallows_closed_stream()
    _patches_applied = True
