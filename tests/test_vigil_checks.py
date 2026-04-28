"""Tests for the anima-mcp Vigil plugin.

This plugin is loaded into unitares's Vigil via VIGIL_CHECK_PLUGINS. The
tests require unitares to be importable (Vigil lives there). Skip gracefully
if someone runs anima-mcp's test suite in isolation.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

UNITARES_DIR = Path(__file__).resolve().parent.parent.parent / "unitares"
if UNITARES_DIR.exists() and str(UNITARES_DIR) not in sys.path:
    sys.path.insert(0, str(UNITARES_DIR))

pytest.importorskip(
    "agents.vigil.checks.base",
    reason="unitares not on sys.path — plugin can't run in isolation",
)


def _reset_registry():
    from agents.vigil.checks import registry
    registry._CHECKS.clear()
    registry._LOADED = False


@pytest.fixture(autouse=True)
def clean_registry():
    _reset_registry()
    yield
    _reset_registry()


def test_lumen_health_identity():
    from anima_mcp.vigil_checks import LumenHealth

    check = LumenHealth()
    assert check.name == "lumen_health"
    assert check.service_key == "lumen"


def test_lumen_health_ok_when_first_url_reachable(monkeypatch):
    """If the first URL responds, no fallback needed and ok_url is recorded."""
    from anima_mcp import vigil_checks

    calls = []

    def fake(url, timeout=10.0):
        calls.append(url)
        return True, "ok (15ms)"

    monkeypatch.setattr(vigil_checks, "check_http_health", fake)

    result = asyncio.run(vigil_checks.LumenHealth().run(prev_state={}))
    assert result.ok is True
    assert len(calls) == 1
    assert result.detail is not None
    assert result.detail["lumen_last_ok_url"] == calls[0]


def test_lumen_health_falls_back_to_second_url(monkeypatch):
    """First URL unreachable → second URL tried → success recorded."""
    from anima_mcp import vigil_checks

    attempts = []
    monkeypatch.setattr(
        vigil_checks.LumenHealth,
        "URLS",
        ["http://primary.local:8766/health", "http://backup.local:8766/health"],
    )

    def fake(url, timeout=10.0):
        attempts.append(url)
        # First URL fails, second succeeds
        if len(attempts) == 1:
            return False, "unreachable"
        return True, "ok"

    monkeypatch.setattr(vigil_checks, "check_http_health", fake)

    result = asyncio.run(vigil_checks.LumenHealth().run(prev_state={}))
    assert result.ok is True
    assert len(attempts) == 2
    assert result.detail["lumen_last_ok_url"] == attempts[1]


def test_lumen_health_reorders_urls_from_prev_state(monkeypatch):
    """The last-known-good URL is tried first next cycle."""
    from anima_mcp import vigil_checks

    attempts = []
    monkeypatch.setattr(vigil_checks, "check_http_health",
                        lambda url, timeout=10.0: (attempts.append(url) or (True, "ok")))

    # Pretend previous cycle's successful URL was the Tailscale one
    tailscale_url = vigil_checks.LumenHealth.URLS[-1]
    asyncio.run(vigil_checks.LumenHealth().run(prev_state={"lumen_last_ok_url": tailscale_url}))
    # Tailscale URL must be the one actually tried first
    assert attempts[0] == tailscale_url


def test_lumen_health_all_urls_unreachable(monkeypatch):
    """All URLs fail → critical severity + stable fingerprint_key."""
    from anima_mcp import vigil_checks

    monkeypatch.setattr(vigil_checks, "check_http_health",
                        lambda url, timeout=10.0: (False, "connection refused"))

    result = asyncio.run(vigil_checks.LumenHealth().run(prev_state={}))
    assert result.ok is False
    assert result.severity == "critical"
    assert result.fingerprint_key == "lumen_unreachable"
    assert "UNREACHABLE" in result.summary or "unreachable" in result.summary.lower()


def test_lumen_health_urls_overridable_via_env(monkeypatch):
    """LUMEN_HEALTH_URLS env var (comma-separated) wins over hardcoded defaults."""
    from anima_mcp import vigil_checks

    monkeypatch.setenv(
        "LUMEN_HEALTH_URLS",
        "http://override.local:9999/health,http://backup.local:9999/health",
    )
    assert vigil_checks._parse_urls() == [
        "http://override.local:9999/health",
        "http://backup.local:9999/health",
    ]


def test_lumen_health_default_urls_when_env_unset(monkeypatch):
    from anima_mcp import vigil_checks

    monkeypatch.delenv("LUMEN_HEALTH_URLS", raising=False)
    urls = vigil_checks._parse_urls()
    assert any("8766/health" in u for u in urls)


def test_plugin_registers_check_on_import(monkeypatch):
    """Importing anima_mcp.vigil_checks causes LumenHealth to appear in the registry."""
    from agents.vigil.checks import registry

    # Fresh import
    sys.modules.pop("anima_mcp.vigil_checks", None)
    import anima_mcp.vigil_checks  # noqa: F401
    names = [c.name for c in registry.all_checks()]
    assert "lumen_health" in names
