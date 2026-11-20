"""Tests for unitares_knowledge: session lifecycle and the knowledge store call."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from anima_mcp import unitares_knowledge as uk


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _FakeResponse:
    def __init__(self, body: str, status: int = 200) -> None:
        self.status = status
        self.headers = {"Content-Type": "application/json"}
        self._body = body

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _PostingSession:
    """Records the one HTTP request share_insight_to_unitares sends."""

    def __init__(self, body: str, status: int = 200) -> None:
        self.closed = False
        self.requests: list = []
        self._response = _FakeResponse(body, status)

    def post(self, url, json=None, headers=None, **kwargs):
        self.requests.append({"url": url, "json": json, "headers": headers, **kwargs})
        return self._response


def _tool_result(payload=None, *, is_error=False, text=None) -> str:
    result = {"content": [{"type": "text", "text": text if text is not None else json.dumps(payload)}]}
    if is_error:
        result["isError"] = True
    return json.dumps({"jsonrpc": "2.0", "id": 1, "result": result})


IDENTITY = SimpleNamespace(creature_id="49e14444-b59e-48f1-83b8-b36a988c9975")
INSIGHT = "I noticed a pattern: I feel calmer when the room dims"


def _share(session: _PostingSession, **kwargs):
    async def run():
        uk._http_session = session
        uk._session_loop = asyncio.get_running_loop()
        return await uk.share_insight_to_unitares(
            INSIGHT, discovery_type="insight", tags=["unified-reflection"], identity=IDENTITY, **kwargs
        )

    return asyncio.run(run())


def setup_function() -> None:
    uk._http_session = None
    uk._session_loop = None
    uk._shared_insights.clear()
    uk._last_share_time = 0.0
    uk._auth_hold_until = 0.0


def teardown_function() -> None:
    setup_function()


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


def test_share_insight_sync_threads_client_session_id(monkeypatch):
    seen = {}

    async def fake_share(*args, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(uk, "share_insight_to_unitares", fake_share)

    uk.share_insight_sync("significant insight", client_session_id="agent-69a1a4f7-a30")

    assert seen["client_session_id"] == "agent-69a1a4f7-a30"


def test_store_uses_canonical_knowledge_tool_with_binding(monkeypatch):
    monkeypatch.setenv("UNITARES_URL", "http://unitares.example:8767/mcp/")
    session = _PostingSession(_tool_result({"success": True, "discovery_id": "d-1"}))

    result = _share(session, client_session_id="agent-69a1a4f7-a30")

    assert result is not None
    (request,) = session.requests
    params = request["json"]["params"]
    assert params["name"] == "knowledge"
    args = params["arguments"]
    assert args["action"] == "store"
    assert args["client_session_id"] == "agent-69a1a4f7-a30"
    assert args["summary"] == INSIGHT
    assert json.loads(args["details"])["source"] == "lumen_autonomous"
    assert "content" not in args
    assert "unified-reflection" in args["tags"]
    assert hash(INSIGHT) in uk._shared_insights


def test_unknown_tool_refusal_is_not_recorded_as_shared(monkeypatch, capsys):
    """The exact /mcp/ refusal every share received before this fix."""
    monkeypatch.setenv("UNITARES_URL", "http://unitares.example:8767/mcp/")
    session = _PostingSession(_tool_result(text="Unknown tool: store_knowledge_graph", is_error=True))

    result = _share(session, client_session_id="agent-69a1a4f7-a30")

    assert result is None
    assert hash(INSIGHT) not in uk._shared_insights
    assert uk._last_share_time == 0.0
    assert "Share refused: Unknown tool: store_knowledge_graph" in capsys.readouterr().err


def test_handler_success_false_is_not_recorded_as_shared(monkeypatch, capsys):
    monkeypatch.setenv("UNITARES_URL", "http://unitares.example:8767/mcp/")
    session = _PostingSession(
        _tool_result({"success": False, "error": "Write operations require session binding"})
    )

    result = _share(session, client_session_id="agent-69a1a4f7-a30")

    assert result is None
    assert hash(INSIGHT) not in uk._shared_insights
    assert "Write operations require session binding" in capsys.readouterr().err


def test_refused_insight_is_offered_again_on_next_share(monkeypatch):
    monkeypatch.setenv("UNITARES_URL", "http://unitares.example:8767/mcp/")
    _share(_PostingSession(_tool_result(text="Unknown tool: x", is_error=True)),
           client_session_id="agent-69a1a4f7-a30")

    retry = _PostingSession(_tool_result({"success": True, "discovery_id": "d-2"}))
    assert _share(retry, client_session_id="agent-69a1a4f7-a30") is not None
    assert len(retry.requests) == 1


def test_unattributed_share_is_skipped_without_a_request(monkeypatch, capsys):
    monkeypatch.setenv("UNITARES_URL", "http://unitares.example:8767/mcp/")
    session = _PostingSession(_tool_result({"success": True}))

    assert _share(session) is None
    assert session.requests == []
    assert "no client_session_id" in capsys.readouterr().err


def test_rest_mode_posts_canonical_store_with_bearer_and_no_forbidden_keys(monkeypatch):
    monkeypatch.setenv("UNITARES_URL", "https://gov.cirwel.org/v1/tools/call")
    monkeypatch.setenv("UNITARES_HTTP_API_TOKEN", "secret-agent-token")
    body = json.dumps({"name": "knowledge", "result": {"success": True, "discovery_id": "d-rest"}, "success": True})
    session = _PostingSession(body)

    res = _share(session, client_session_id="agent-69a1a4f7-a30")

    assert res == {"success": True, "discovery_id": "d-rest"}
    assert len(session.requests) == 1
    req = session.requests[0]
    assert req["url"] == "https://gov.cirwel.org/v1/tools/call"
    assert req["headers"]["Authorization"] == "Bearer secret-agent-token"
    assert "X-Agent-Id" not in req["headers"]
    assert "X-Session-ID" not in req["headers"]
    assert req["allow_redirects"] is False
    assert req["json"]["name"] == "knowledge"
    args = req["json"]["arguments"]
    assert args["action"] == "store"
    assert args["severity"] == "low"
    assert args["client_session_id"] == "agent-69a1a4f7-a30"
    for forbidden in ("agent_id", "agent_uuid", "continuity_token", "session_id", "kwargs"):
        assert forbidden not in args
    assert hash(INSIGHT) in uk._shared_insights


def test_rest_mode_scope_denied_and_refused_are_not_recorded_as_shared(monkeypatch, capsys):
    monkeypatch.setenv("UNITARES_URL", "https://gov.cirwel.org/v1/tools/call")
    monkeypatch.setenv("UNITARES_HTTP_API_TOKEN", "secret-agent-token")
    denied_body = json.dumps({"success": False, "error": "lacks knowledge_store scope", "error_type": "AgentTokenScope"})
    session = _PostingSession(denied_body, status=403)

    res = _share(session, client_session_id="agent-69a1a4f7-a30")

    assert res is None
    assert hash(INSIGHT) not in uk._shared_insights
    assert "scope_denied" in capsys.readouterr().err


def test_rest_mode_refuses_bearer_over_non_loopback_http(monkeypatch, capsys):
    monkeypatch.setenv("UNITARES_URL", "http://unitares.tail.ts.net:8767/v1/tools/call")
    monkeypatch.setenv("UNITARES_HTTP_API_TOKEN", "secret-agent-token")
    session = _PostingSession("{}")

    res = _share(session, client_session_id="agent-69a1a4f7-a30")

    assert res is None
    assert session.requests == []
    assert "non-loopback HTTP" in capsys.readouterr().err


def test_rest_mode_unauthorized_holds_cooldown_and_redacts_token(monkeypatch, capsys):
    monkeypatch.setenv("UNITARES_URL", "https://gov.cirwel.org/v1/tools/call")
    monkeypatch.setenv("UNITARES_HTTP_API_TOKEN", "raw-secret-tok")
    session = _PostingSession("{}", status=401)
    assert _share(session, client_session_id="agent-69a1a4f7-a30") is None
    assert len(session.requests) == 1
    assert uk._auth_hold_until > 0.0

    # While cooldown is open, second call skips HTTP POST
    session2 = _PostingSession("{}", status=200)
    assert _share(session2, client_session_id="agent-69a1a4f7-a30") is None
    assert session2.requests == []
    assert "raw-secret-tok" not in capsys.readouterr().err
