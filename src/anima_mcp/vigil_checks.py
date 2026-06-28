"""Lumen health check — plugs into unitares's Vigil via VIGIL_CHECK_PLUGINS.

Loads lazily: unitares is only imported when this module is imported, not at
anima-mcp startup. The operator's Vigil launchd sets
VIGIL_CHECK_PLUGINS=anima_mcp.vigil_checks which triggers registration.

An agnostic unitares user without anima-mcp simply doesn't set the env var;
Vigil then runs with only its built-in governance_health check.
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import httpx

# These imports only succeed when unitares is on sys.path. The plugin has a
# hard dependency on unitares by design — plugins point one way (anima → unitares),
# never the reverse.
from agents.vigil.checks.base import CheckResult
from agents.vigil.checks import registry


def _parse_urls() -> List[str]:
    raw = os.environ.get("LUMEN_HEALTH_URLS")
    if raw:
        return [u.strip() for u in raw.split(",") if u.strip()]
    # Default uses Tailscale hostname (resolves via magic-DNS); operator-
    # specific LAN IPs belong in LUMEN_HEALTH_URLS, not in a public default.
    return [
        "http://lumen:8766/health",            # Tailscale hostname
    ]


def check_http_health(url: str, timeout: float = 10.0) -> Tuple[bool, str]:
    """Sync HTTP probe — mirrors the helper in unitares agent.py.

    Defined locally (not imported) so tests can monkeypatch this symbol in
    isolation without reaching into unitares internals.
    """
    start = time.monotonic()
    try:
        resp = httpx.get(url, timeout=timeout)
        latency_ms = int((time.monotonic() - start) * 1000)
        if resp.status_code == 200:
            try:
                data = resp.json()
                status = data.get("status", "ok")
                return True, f"{status} ({latency_ms}ms)"
            except Exception:
                return True, f"ok ({latency_ms}ms)"
        return False, f"HTTP {resp.status_code} ({latency_ms}ms)"
    except httpx.ConnectError:
        return False, "unreachable"
    except httpx.TimeoutException:
        return False, f"timeout (>{int(timeout*1000)}ms)"
    except Exception as e:
        return False, str(e)


class LumenHealth:
    name = "lumen_health"
    service_key = "lumen"
    URLS: List[str] = _parse_urls()

    async def run(self, prev_state: dict | None = None) -> CheckResult:
        prev_state = prev_state or {}
        # Prefer whichever URL worked last cycle — avoids useless LAN attempts
        # when the Pi is on Tailscale only (or vice versa).
        last_ok = prev_state.get("lumen_last_ok_url")
        urls = list(self.URLS)
        if last_ok and last_ok in urls:
            urls.remove(last_ok)
            urls.insert(0, last_ok)

        # Reach through the module so monkeypatches in tests apply.
        from . import vigil_checks as _this

        last_detail = "unreachable"
        for url in urls:
            ok, detail = _this.check_http_health(url)
            last_detail = detail
            if ok:
                return CheckResult(
                    ok=True,
                    summary=f"Lumen: {detail}",
                    detail={"lumen_last_ok_url": url},
                )
        return CheckResult(
            ok=False,
            summary=f"Lumen: UNREACHABLE ({last_detail})",
            severity="critical",
            fingerprint_key="lumen_unreachable",
        )


registry.register(LumenHealth())
