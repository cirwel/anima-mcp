"""
Tests for UNITARES bridge module.

Validates governance integration and fallback behavior.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from anima_mcp.unitares_bridge import (
    UnitaresBridge,
    _extract_governance_eisv,
    check_governance,
    mcp_tool_refusal,
)
from anima_mcp.anima import Anima
from anima_mcp.sensors.base import SensorReadings
from anima_mcp.eisv_mapper import EISVMetrics


def _mock_http_response(status=200, body='{"result": {}}', content_type="application/json"):
    """Create async-context-manager response object for aiohttp mocks."""
    resp = AsyncMock()
    resp.status = status
    resp.headers = {"Content-Type": content_type}
    resp.text = AsyncMock(return_value=body)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def create_test_readings() -> SensorReadings:
    """Create test sensor readings."""
    return SensorReadings(
        timestamp=datetime.now(),
        cpu_temp_c=45.0,
        ambient_temp_c=22.0,
        humidity_pct=50.0,
        light_lux=300.0,
        cpu_percent=50.0,
        memory_percent=50.0,
        disk_percent=50.0,
        eeg_alpha_power=0.5,
        eeg_beta_power=0.6,
        eeg_gamma_power=0.4,
    )


def create_test_anima() -> Anima:
    """Create test anima state."""
    readings = create_test_readings()
    return Anima(
        warmth=0.7,
        clarity=0.6,
        stability=0.8,
        presence=0.7,
        readings=readings
    )


def test_extracts_flat_minimal_governance_eisv_without_inventing_source():
    vector, source = _extract_governance_eisv(
        {"_mode": "minimal", "E": 0.2, "I": 0.8, "S": 0.1, "V": -0.6}
    )

    assert vector == {"E": 0.2, "I": 0.8, "S": 0.1, "V": -0.6}
    assert source == "unitares_primary_response_source_omitted"


def test_missing_governance_eisv_stays_missing():
    vector, source = _extract_governance_eisv({"action": "proceed"})

    assert vector is None
    assert source == "not_returned_by_unitares"


def test_out_of_range_governance_eisv_stays_missing():
    vector, source = _extract_governance_eisv(
        {"E": 0.2, "I": 0.8, "S": 1.1, "V": -0.6}
    )

    assert vector is None
    assert source == "not_returned_by_unitares"


@pytest.mark.asyncio
async def test_local_governance_fallback():
    """Test local governance when UNITARES unavailable."""
    bridge = UnitaresBridge(unitares_url=None)  # No URL = local only
    
    anima = create_test_anima()
    readings = create_test_readings()
    
    decision = await bridge.check_in(anima, readings)
    
    assert "action" in decision
    assert "margin" in decision
    assert "reason" in decision
    assert "eisv" in decision
    assert decision["source"] == "local"
    assert decision["action"] in ["proceed", "pause", "halt"]


@pytest.mark.asyncio
async def test_local_governance_high_entropy():
    """Test local governance pauses on high entropy."""
    bridge = UnitaresBridge(unitares_url=None)
    
    # Low stability = high entropy
    anima = Anima(
        warmth=0.5,
        clarity=0.5,
        stability=0.2,  # Low stability = high entropy
        presence=0.5,
        readings=create_test_readings()
    )
    readings = create_test_readings()
    
    decision = await bridge.check_in(anima, readings)
    
    # High entropy should trigger pause (local governance calls this "risk")
    assert decision["action"] == "pause"
    assert "risk" in decision["reason"].lower()


@pytest.mark.asyncio
async def test_local_governance_low_integrity():
    """Test local governance pauses on low integrity."""
    bridge = UnitaresBridge(unitares_url=None)
    
    # Low clarity = low integrity
    anima = Anima(
        warmth=0.5,
        clarity=0.3,  # Low clarity = low integrity
        stability=0.5,
        presence=0.5,
        readings=create_test_readings()
    )
    readings = create_test_readings()

    decision = await bridge.check_in(anima, readings)

    # Low integrity should trigger pause
    assert decision["action"] == "pause"
    assert "coherence" in decision["reason"].lower()


@pytest.mark.asyncio
async def test_local_governance_comfortable():
    """Test local governance proceeds when comfortable."""
    bridge = UnitaresBridge(unitares_url=None)
    
    # Healthy state
    anima = Anima(
        warmth=0.6,
        clarity=0.7,
        stability=0.8,
        presence=0.7,
        readings=create_test_readings()
    )
    readings = create_test_readings()
    
    decision = await bridge.check_in(anima, readings)
    
    # Healthy state should proceed
    assert decision["action"] == "proceed"
    assert decision["margin"] in ["comfortable", "tight"]


@pytest.mark.asyncio
async def test_check_availability_no_url():
    """Test availability check with no URL."""
    bridge = UnitaresBridge(unitares_url=None)
    available = await bridge.check_availability()
    assert available is False


@pytest.mark.asyncio
async def test_check_availability_unreachable():
    """Test availability check with unreachable URL."""
    bridge = UnitaresBridge(unitares_url="http://127.0.0.1:99999/sse")
    try:
        available = await bridge.check_availability()
        assert available is False
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_check_governance_convenience():
    """Test convenience function."""
    anima = create_test_anima()
    readings = create_test_readings()
    
    decision = await check_governance(anima, readings, unitares_url=None)
    
    assert "action" in decision
    assert "margin" in decision
    assert "source" in decision


@pytest.mark.asyncio
async def test_bridge_with_agent_id():
    """Test bridge with agent ID."""
    bridge = UnitaresBridge(unitares_url=None, agent_id="test-creature")
    assert bridge._agent_id == "test-creature"
    
    bridge.set_agent_id("new-id")
    assert bridge._agent_id == "new-id"


@pytest.mark.asyncio
async def test_bridge_with_session_id():
    """Test bridge with session ID."""
    bridge = UnitaresBridge(unitares_url=None)
    bridge.set_session_id("test-session")
    assert bridge._session_id == "test-session"


@pytest.mark.asyncio
async def test_decision_includes_eisv():
    """Legacy EISV remains an explicitly tagged alias of body telemetry."""
    bridge = UnitaresBridge(unitares_url=None)
    anima = create_test_anima()
    readings = create_test_readings()
    
    decision = await bridge.check_in(anima, readings)
    
    assert "eisv" in decision
    assert "E" in decision["eisv"]
    assert "I" in decision["eisv"]
    assert "S" in decision["eisv"]
    assert "V" in decision["eisv"]
    assert 0.0 <= decision["eisv"]["E"] <= 1.0
    assert 0.0 <= decision["eisv"]["I"] <= 1.0
    assert 0.0 <= decision["eisv"]["S"] <= 1.0
    assert -1.0 <= decision["eisv"]["V"] <= 1.0  # V is signed valence
    assert decision["body_eisv_projection"] == decision["eisv"]
    assert decision["eisv_source"] == "body_eisv_projection_legacy_alias"
    assert decision["governance_eisv"] is None
    assert decision["governance_eisv_source"] == "not_returned_by_unitares"
    assert decision["state_space_provenance"]["eisv"] == {
        "alias_of": "body_eisv_projection",
        "deprecated": True,
    }


@pytest.mark.asyncio
async def test_margin_calculation():
    """Test margin calculation in local governance."""
    bridge = UnitaresBridge(unitares_url=None)
    
    # Test different margins
    test_cases = [
        (0.5, 0.5, 0.5, 0.5, "comfortable"),  # Middle of range
        (0.55, 0.45, 0.1, 0.1, "tight"),      # Near thresholds
        (0.6, 0.4, 0.05, 0.15, "critical"),   # At thresholds
    ]
    
    for warmth, clarity, stability, presence, expected_margin in test_cases:
        anima = Anima(
            warmth=warmth,
            clarity=clarity,
            stability=stability,
            presence=presence,
            readings=create_test_readings()
        )
        readings = create_test_readings()
        
        decision = await bridge.check_in(anima, readings)
        
        # Margin should be calculated (may not match exactly due to thresholds)
        assert decision["margin"] in ["comfortable", "tight", "critical"]


def test_get_mcp_url_with_mcp():
    """Test _get_mcp_url when URL already contains /mcp."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp")
    assert bridge._get_mcp_url() == "http://localhost:8767/mcp/"


def test_get_mcp_url_with_sse():
    """Test _get_mcp_url converts /sse to /mcp."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/sse")
    assert bridge._get_mcp_url() == "http://localhost:8767/mcp/"


def test_get_mcp_url_bare():
    """Test _get_mcp_url appends /mcp to bare URL."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767")
    assert bridge._get_mcp_url() == "http://localhost:8767/mcp/"


def test_get_health_url_from_mcp_url():
    """Availability checks should hit /health, not /mcp/health."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp/")
    assert bridge._get_health_url() == "http://localhost:8767/health"


def test_parse_mcp_response_json():
    """Test parsing a plain JSON response."""
    result = UnitaresBridge._parse_mcp_response(
        '{"result": {"content": []}}',
        "application/json"
    )
    assert result == {"result": {"content": []}}


def test_parse_mcp_response_sse():
    """Test parsing an SSE response."""
    sse_text = 'event: message\ndata: {"result": "ok"}\n\n'
    result = UnitaresBridge._parse_mcp_response(sse_text, "text/event-stream")
    assert result == {"result": "ok"}


def test_parse_mcp_response_sse_no_data():
    """Test parsing SSE response with no valid data lines."""
    result = UnitaresBridge._parse_mcp_response(
        "event: message\n\n", "text/event-stream"
    )
    assert result is None


def test_parse_mcp_response_sse_bad_json():
    """Test parsing SSE response with invalid JSON falls through."""
    sse_text = 'data: not-json\ndata: {"ok": true}\n'
    result = UnitaresBridge._parse_mcp_response(sse_text, "text/event-stream")
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_resolve_caller_identity_no_url():
    """Test resolve_caller_identity returns None when no URL configured."""
    bridge = UnitaresBridge(unitares_url=None)
    result = await bridge.resolve_caller_identity()
    assert result is None


@pytest.mark.asyncio
async def test_resolve_caller_identity_no_session():
    """Test resolve_caller_identity returns None with no session ID."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp")
    bridge._available = True
    result = await bridge.resolve_caller_identity()
    assert result is None


@pytest.mark.asyncio
async def test_resolve_caller_identity_unavailable():
    """Test resolve_caller_identity returns None when UNITARES is known unavailable."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp")
    bridge._available = False
    bridge.set_session_id("test-session")
    result = await bridge.resolve_caller_identity()
    assert result is None


@pytest.mark.asyncio
async def test_check_availability_staleness_recheck():
    """Test that availability is rechecked after 5 minutes."""
    # Use an unreachable URL so the recheck actually fails
    bridge = UnitaresBridge(unitares_url="http://127.0.0.1:19999/mcp")
    try:
        bridge._available = True
        import time
        # Recent check — should return True without rechecking
        bridge._last_availability_check = time.time()
        assert await bridge.check_availability() is True

        # Stale check (6 min ago) — should fall through to recheck (and fail, no server)
        bridge._last_availability_check = time.time() - 360
        result = await bridge.check_availability()
        assert result is False
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_check_availability_handles_401_and_opens_circuit():
    """401 responses should mark unavailable and trip circuit logic."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp")
    bridge._circuit_threshold = 1

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=_mock_http_response(status=401, body="unauthorized"))

    with patch.object(bridge, "_get_session", return_value=mock_session):
        available = await bridge.check_availability()

    assert available is False
    assert bridge._available is False
    assert bridge._circuit_open_until > 0


@pytest.mark.asyncio
async def test_check_availability_resets_session_on_connection_failure():
    """A connection/DNS failure should drop the pooled session so the next
    probe rebuilds the connector and re-resolves DNS (self-heal)."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp")
    bridge._circuit_threshold = 1

    mock_session = AsyncMock()
    mock_session.get = MagicMock(side_effect=OSError("Name or service not known"))
    mock_session.post = MagicMock(side_effect=OSError("Name or service not known"))

    reset_spy = AsyncMock()
    with patch.object(bridge, "_get_session", return_value=mock_session), \
         patch.object(bridge, "_reset_session", reset_spy):
        available = await bridge.check_availability()

    assert available is False
    assert bridge._circuit_open_until > 0
    reset_spy.assert_awaited()


@pytest.mark.asyncio
async def test_check_availability_401_does_not_reset_session():
    """An auth (401) failure is not a connectivity problem — it should NOT
    churn the session, since re-resolving DNS won't fix authorization."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp")
    bridge._circuit_threshold = 1

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=_mock_http_response(status=401, body="unauthorized"))

    reset_spy = AsyncMock()
    with patch.object(bridge, "_get_session", return_value=mock_session), \
         patch.object(bridge, "_reset_session", reset_spy):
        available = await bridge.check_availability()

    assert available is False
    reset_spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_unitares_returns_parsed_governance_payload():
    """_call_unitares should unwrap MCP tool content and map fields."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp", agent_id="agent-123")
    bridge.set_session_id("sess-1")

    governance_text = json.dumps(
        {
            "action": "pause",
            "margin": "tight",
            "reason": "edge case",
            "resolved_agent_id": "abc-123",
            "metrics": {"E": 0.21, "I": 0.82, "S": 0.17, "V": -0.61},
            "primary_eisv_source": "behavioral",
        }
    )
    mcp_result = json.dumps(
        {"result": {"content": [{"type": "text", "text": governance_text}]}}
    )
    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=_mock_http_response(body=mcp_result))

    anima = create_test_anima()
    readings = create_test_readings()
    eisv = EISVMetrics(energy=0.4, integrity=0.5, entropy=0.3, valence=0.2)

    with patch.object(bridge, "_get_session", return_value=mock_session), patch(
        "anima_mcp.unitares_bridge.estimate_complexity", return_value=0.2
    ), patch(
        "anima_mcp.unitares_bridge.generate_status_text", return_value="status"
    ), patch(
        "anima_mcp.unitares_bridge.compute_ethical_drift", return_value=0.01
    ), patch(
        "anima_mcp.unitares_bridge.compute_confidence", return_value=0.8
    ):
        decision = await bridge._call_unitares(anima, readings, eisv)

    assert decision["source"] == "unitares"
    assert decision["action"] == "pause"
    assert decision["margin"] == "tight"
    assert decision["unitares_agent_id"] == "abc-123"
    assert decision["body_eisv_projection"] == eisv.to_dict()
    assert decision["eisv"] == eisv.to_dict()
    assert decision["governance_eisv"] == {
        "E": 0.21,
        "I": 0.82,
        "S": 0.17,
        "V": -0.61,
    }
    assert decision["governance_eisv_source"] == "behavioral"

    posted = mock_session.post.call_args.kwargs["json"]
    sensor_data = posted["params"]["arguments"]["sensor_data"]
    assert sensor_data["body_eisv_projection"] == eisv.to_dict()
    assert sensor_data["eisv"] == eisv.to_dict()
    assert sensor_data["body_anima"] == sensor_data["anima"]
    assert sensor_data["state_space_provenance"]["anima"] == {
        "alias_of": "body_anima",
        "deprecated": True,
    }


@pytest.mark.asyncio
async def test_call_unitares_raises_on_http_error():
    """_call_unitares wraps HTTP failures in a bridge exception."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp", agent_id="agent-123")
    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=_mock_http_response(status=500, body="boom"))

    anima = create_test_anima()
    readings = create_test_readings()
    eisv = EISVMetrics(energy=0.4, integrity=0.5, entropy=0.3, valence=0.2)

    with patch.object(bridge, "_get_session", return_value=mock_session), patch(
        "anima_mcp.unitares_bridge.estimate_complexity", return_value=0.2
    ), patch(
        "anima_mcp.unitares_bridge.generate_status_text", return_value="status"
    ), patch(
        "anima_mcp.unitares_bridge.compute_ethical_drift", return_value=0.01
    ), patch(
        "anima_mcp.unitares_bridge.compute_confidence", return_value=0.8
    ):
        with pytest.raises(Exception, match="HTTP 500"):
            await bridge._call_unitares(anima, readings, eisv)


def _tool_result(payload=None, *, is_error=False, text=None):
    """A JSON-RPC tools/call response body shaped like UNITARES /mcp/ output."""
    result = {"content": [{"type": "text", "text": text if text is not None else json.dumps(payload)}]}
    if is_error:
        result["isError"] = True
    return json.dumps({"jsonrpc": "2.0", "id": 1, "result": result})


def _sync_identity():
    identity = MagicMock()
    identity.born_at = datetime(2026, 1, 11, 0, 19, 14)
    identity.total_awakenings = 82
    identity.total_alive_seconds = 1200.0
    identity.alive_ratio.return_value = 0.3
    identity.name_history = ["Anima", "Lumen"]
    identity.current_awakening_at = datetime.now()
    identity.name = "Lumen"
    identity.creature_id = "creature-abcdef"
    return identity


async def _sync_identity_metadata(body):
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp", agent_id="agent-123")
    bridge.set_session_id("sess-1")
    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=_mock_http_response(body=body))
    with patch.object(bridge, "_get_session", return_value=mock_session):
        ok = await bridge.sync_identity_metadata(_sync_identity())
    return ok, bridge, mock_session.post.call_args.kwargs["json"]


@pytest.mark.asyncio
async def test_sync_identity_metadata_success():
    """Identity metadata sync returns True for successful MCP result."""
    ok, _, _ = await _sync_identity_metadata(
        _tool_result({"success": True, "message": "Agent metadata updated", "agent_id": "69a1a4f7"})
    )
    assert ok is True


@pytest.mark.asyncio
async def test_sync_identity_metadata_calls_canonical_agent_update():
    """The call uses a name /mcp/ registers, bound by the bridge's session key."""
    _, bridge, posted = await _sync_identity_metadata(_tool_result({"success": True}))

    assert posted["params"]["name"] == "agent"
    args = posted["params"]["arguments"]
    assert args["action"] == "update"
    assert args["client_session_id"] == bridge.client_session_id()
    assert args["notes"] == (
        "Lumen identity: creature_id=creature-abcdef, born=2026-01-11T00:19:14, awakenings=82"
    )


@pytest.mark.asyncio
async def test_sync_identity_metadata_never_sends_tags():
    """tags REPLACES the server-side list, so sending a subset strips the rest.

    Lumen's row holds server-granted tags (pinned, pioneer, persistent) that
    gate archival immunity and delete refusal. purpose/preferences are not in
    the /mcp/ schema and would be dropped silently, so they are not sent either.
    """
    _, _, posted = await _sync_identity_metadata(_tool_result({"success": True}))

    args = posted["params"]["arguments"]
    assert "tags" not in args
    assert "purpose" not in args
    assert "preferences" not in args


@pytest.mark.asyncio
async def test_sync_identity_metadata_unknown_tool_is_failure():
    """The exact 2026-09-13 /mcp/ refusal for a pre-consolidation name."""
    ok, _, _ = await _sync_identity_metadata(
        _tool_result(text="Unknown tool: update_agent_metadata", is_error=True)
    )
    assert ok is False


@pytest.mark.asyncio
async def test_sync_identity_metadata_handler_refusal_is_failure():
    """Handler refusals are normal results whose text payload says success: false."""
    ok, _, _ = await _sync_identity_metadata(
        _tool_result({"success": False, "error": "Write operations require session binding"})
    )
    assert ok is False


class TestMcpToolRefusal:
    def _parsed(self, body):
        return json.loads(body)

    def test_success_payload_is_not_a_refusal(self):
        assert mcp_tool_refusal(self._parsed(_tool_result({"success": True, "discovery_id": "d1"}))) is None

    def test_non_json_text_is_not_a_refusal(self):
        assert mcp_tool_refusal(self._parsed(_tool_result(text="ok"))) is None

    def test_is_error_carries_the_server_text(self):
        body = _tool_result(text="Unknown tool: store_knowledge_graph", is_error=True)
        assert mcp_tool_refusal(self._parsed(body)) == "Unknown tool: store_knowledge_graph"

    def test_success_false_carries_the_error(self):
        body = _tool_result({"success": False, "error": "Agent not found", "error_code": "NOT_FOUND"})
        assert mcp_tool_refusal(self._parsed(body)) == "Agent not found"

    def test_typed_identity_refusal_has_no_success_key_and_still_refuses(self):
        body = _tool_result({"status": "identity_required", "hint": "call identity(resume=true)"})
        assert mcp_tool_refusal(self._parsed(body)) == "call identity(resume=true)"

    def test_jsonrpc_error(self):
        response = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "Invalid params"}}
        assert mcp_tool_refusal(response) == "Invalid params"

    @pytest.mark.parametrize("response", [None, {}, {"result": None}, "not a dict"])
    def test_missing_response_or_result(self, response):
        assert mcp_tool_refusal(response) is not None


@pytest.mark.asyncio
async def test_resolve_caller_identity_uses_label_fallback():
    """resolve_caller_identity should return label when display_name absent."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp")
    bridge._available = True
    bridge.set_session_id("sess-1")

    payload = {"result": {"content": [{"type": "text", "text": '{"label":"Visitor 7"}'}]}}
    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=_mock_http_response(body=json.dumps(payload)))

    with patch.object(bridge, "_get_session", return_value=mock_session):
        name = await bridge.resolve_caller_identity()

    assert name == "Visitor 7"


def test_anima_snapshot_module_level():
    """Test that _AnimaSnapshot is defined at module level and reusable."""
    from anima_mcp.unitares_bridge import _AnimaSnapshot
    snap1 = _AnimaSnapshot(0.5, 0.6, 0.7, 0.8)
    snap2 = _AnimaSnapshot(0.1, 0.2, 0.3, 0.4)
    assert isinstance(snap1, _AnimaSnapshot)
    assert isinstance(snap2, _AnimaSnapshot)
    assert snap1.warmth == 0.5
    assert snap2.clarity == 0.2


@pytest.mark.asyncio
async def test_sync_name_handles_sse_response():
    """Test sync_name uses _parse_mcp_response instead of response.json()."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp")
    bridge._available = True
    bridge._agent_id = "test-creature"

    sse_body = 'event: message\ndata: {"result": {"content": [{"text": "ok"}]}}\n\n'
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.text = AsyncMock(return_value=sse_body)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)

    with patch.object(bridge, '_get_session', return_value=mock_session):
        result = await bridge.sync_name("Lumen")
    assert result is True


@pytest.mark.asyncio
async def test_report_outcome_handles_sse_response():
    """Test report_outcome uses _parse_mcp_response instead of response.json()."""
    bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp")
    bridge._available = True
    bridge._agent_id = "test-creature"

    sse_body = 'event: message\ndata: {"result": {"content": [{"text": "ok"}]}}\n\n'
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.text = AsyncMock(return_value=sse_body)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)

    with patch.object(bridge, '_get_session', return_value=mock_session):
        result = await bridge.report_outcome("drawing_complete", outcome_score=0.8)
    assert result is True


def test_server_bridge_uses_get_identity():
    """Test that _get_server_bridge uses get_identity() not .identity."""
    from anima_mcp.server import _get_server_bridge
    import anima_mcp.server as server_mod
    import anima_mcp.ctx_ref as ctx_ref_mod
    from types import SimpleNamespace

    # Save and reset globals
    old_ctx = server_mod._ctx
    old_cr_ctx = ctx_ref_mod._ctx
    try:
        mock_identity = MagicMock()
        mock_identity.creature_id = "test-creature-id-1234"

        mock_store = MagicMock()
        mock_store.get_identity = MagicMock(return_value=mock_identity)
        del mock_store.identity  # Ensure .identity would fail if accessed

        ctx = SimpleNamespace(server_bridge=None, store=mock_store)
        server_mod._ctx = ctx
        ctx_ref_mod._ctx = ctx

        with patch.dict('os.environ', {'UNITARES_URL': 'http://localhost:8767/mcp'}):
            bridge = _get_server_bridge()

        if bridge is not None:
            assert bridge._agent_id == "test-creature-id-1234"
            mock_store.get_identity.assert_called_once()
    finally:
        server_mod._ctx = old_ctx
        ctx_ref_mod._ctx = old_cr_ctx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
