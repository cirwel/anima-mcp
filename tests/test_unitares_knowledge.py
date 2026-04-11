"""Tests for unitares_knowledge session lifecycle helpers."""

from __future__ import annotations

import asyncio

from anima_mcp import unitares_knowledge as uk


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True


def setup_function() -> None:
    uk._http_session = None
    uk._session_loop = None


def teardown_function() -> None:
    uk._http_session = None
    uk._session_loop = None


def test_share_insight_sync_closes_loop_owned_shared_session(monkeypatch):
    session = _FakeSession()

    async def fake_share(*args, **kwargs):
        uk._http_session = session
        uk._session_loop = asyncio.get_running_loop()
        return {"status": "ok"}

    monkeypatch.setattr(uk, "share_insight_to_unitares", fake_share)

    result = uk.share_insight_sync("significant insight")

    assert result == {"status": "ok"}
    assert session.closed is True
    assert session.close_calls == 1
    assert uk._http_session is None
    assert uk._session_loop is None
