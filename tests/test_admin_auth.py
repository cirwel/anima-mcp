"""Tests for destructive-handler admin gate.

The gate is transparent when ANIMA_ADMIN_SECRET is unset (backward compat).
When set, handlers like handle_git_pull and handle_system_power must see the
matching secret in the ContextVar populated from X-Anima-Admin.
"""
from __future__ import annotations

import pytest

from anima_mcp.admin_auth import require_admin, set_admin_header


@pytest.fixture(autouse=True)
def _reset_header():
    set_admin_header(None)
    yield
    set_admin_header(None)


class TestNoSecretConfigured:
    def test_no_env_no_header_passes(self, monkeypatch):
        monkeypatch.delenv("ANIMA_ADMIN_SECRET", raising=False)
        assert require_admin() is None

    def test_no_env_with_header_still_passes(self, monkeypatch):
        monkeypatch.delenv("ANIMA_ADMIN_SECRET", raising=False)
        set_admin_header("anything")
        assert require_admin() is None


class TestSecretConfigured:
    def test_missing_header_returns_error(self, monkeypatch):
        monkeypatch.setenv("ANIMA_ADMIN_SECRET", "shh")
        result = require_admin()
        assert result is not None
        assert "X-Anima-Admin" in result[0].text

    def test_wrong_header_returns_error(self, monkeypatch):
        monkeypatch.setenv("ANIMA_ADMIN_SECRET", "shh")
        set_admin_header("wrong")
        result = require_admin()
        assert result is not None

    def test_matching_header_passes(self, monkeypatch):
        monkeypatch.setenv("ANIMA_ADMIN_SECRET", "shh")
        set_admin_header("shh")
        assert require_admin() is None

    def test_empty_header_does_not_satisfy_secret(self, monkeypatch):
        monkeypatch.setenv("ANIMA_ADMIN_SECRET", "shh")
        set_admin_header("")
        result = require_admin()
        assert result is not None


class TestHandlerIntegration:
    async def test_system_power_blocked_without_header(self, monkeypatch):
        monkeypatch.setenv("ANIMA_ADMIN_SECRET", "shh")
        from anima_mcp.handlers.system_ops import handle_system_power
        result = await handle_system_power({"action": "status"})
        assert "X-Anima-Admin" in result[0].text

    async def test_system_power_allowed_with_matching_header(self, monkeypatch):
        monkeypatch.setenv("ANIMA_ADMIN_SECRET", "shh")
        set_admin_header("shh")
        from anima_mcp.handlers.system_ops import handle_system_power
        result = await handle_system_power({"action": "status"})
        assert "X-Anima-Admin" not in result[0].text

    async def test_system_power_open_when_secret_unset(self, monkeypatch):
        monkeypatch.delenv("ANIMA_ADMIN_SECRET", raising=False)
        from anima_mcp.handlers.system_ops import handle_system_power
        result = await handle_system_power({"action": "status"})
        assert "X-Anima-Admin" not in result[0].text
