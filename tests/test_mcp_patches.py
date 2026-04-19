"""Tests for runtime patches to the MCP SDK.

The ClosedResourceError swallow patch is a workaround for an upstream bug
where MCP 1.26.0 crashes a whole stateless session if the server tries to
send a log-notification to a client that has already disconnected.
"""
from __future__ import annotations

import asyncio

import anyio
import pytest

from anima_mcp import mcp_patches


@pytest.fixture
def fresh_patches():
    """Reset the patch module's applied-flag between tests so apply_all_patches
    can be exercised multiple times."""
    mcp_patches._patches_applied = False
    yield
    # Don't un-wrap — the wrapper is idempotent and the flag guards re-entry


class TestClosedResourceSwallow:
    def test_apply_all_patches_is_idempotent(self, fresh_patches):
        """Must be safe to call twice (server restart, test reload, etc.)"""
        mcp_patches.apply_all_patches()
        mcp_patches.apply_all_patches()
        # No exception = pass

    def test_send_notification_is_flagged_after_apply(self, fresh_patches):
        """After apply, BaseSession.send_notification carries the flag."""
        try:
            from mcp.shared.session import BaseSession
        except ImportError:
            pytest.skip("mcp package not installed in test env")

        mcp_patches.apply_all_patches()
        assert getattr(
            BaseSession.send_notification, "_anima_closed_stream_patch", False
        )

    def test_wrapper_swallows_closed_stream(self):
        """Core behaviour: ClosedResourceError is caught and not re-raised."""
        async def raiser(self):
            raise anyio.ClosedResourceError

        wrapped = mcp_patches._wrap_swallow_closed(raiser)
        asyncio.run(wrapped(object()))  # no raise = pass

    def test_wrapper_propagates_other_errors(self):
        """Patch must only swallow ClosedResourceError — other exceptions
        (a real bug in the server) must still propagate so we see them."""
        async def raiser(self):
            raise RuntimeError("real bug")

        wrapped = mcp_patches._wrap_swallow_closed(raiser)
        with pytest.raises(RuntimeError, match="real bug"):
            asyncio.run(wrapped(object()))

    def test_wrapper_returns_original_value_on_success(self):
        """Happy path must not be disturbed — ordinary notifications
        pass through unchanged."""
        async def ok(self):
            return "sent"

        wrapped = mcp_patches._wrap_swallow_closed(ok)
        assert asyncio.run(wrapped(object())) == "sent"
