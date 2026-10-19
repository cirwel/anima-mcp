"""Targeted tests for REST endpoint helpers and gallery endpoints."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from starlette.requests import Request

from anima_mcp import rest_api


@pytest.fixture(autouse=True)
def _default_auth_mode(monkeypatch):
    """Keep legacy permissive mode in tests unless a test overrides it."""
    monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", None)
    monkeypatch.setattr(rest_api, "_ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN", True)


def _make_request(
    method: str = "GET",
    path: str = "/",
    *,
    headers: dict[str, str] | None = None,
    query: str = "",
    body: dict | None = None,
    client_host: str = "8.8.8.8",
    path_params: dict[str, str] | None = None,
) -> Request:
    """Create a Starlette request object for direct endpoint calls."""
    raw_headers = [(k.lower().encode("utf-8"), v.encode("utf-8")) for k, v in (headers or {}).items()]
    body_bytes = b""
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        raw_headers.append((b"content-type", b"application/json"))

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "headers": raw_headers,
        "query_string": query.encode("utf-8"),
        "client": (client_host, 12345),
        "path_params": path_params or {},
    }

    sent = {"done": False}

    async def receive():
        if sent["done"]:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent["done"] = True
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    return Request(scope, receive)


class TestRestAuthHelpers:
    def test_is_trusted_network_uses_client_host(self):
        request = _make_request(client_host="100.64.0.20")
        assert rest_api._is_trusted_network(request) is True

    def test_is_trusted_network_ignores_x_forwarded_for_from_untrusted_peer(self):
        request = _make_request(headers={"x-forwarded-for": "100.64.0.20"}, client_host="8.8.8.8")
        assert rest_api._is_trusted_network(request) is False

    def test_is_trusted_network_uses_x_forwarded_for_from_trusted_proxy(self, monkeypatch):
        monkeypatch.setattr(
            rest_api,
            "_TRUSTED_PROXY_NETWORKS",
            [rest_api.ipaddress.ip_network("127.0.0.0/8")],
        )
        request = _make_request(headers={"x-forwarded-for": "100.64.0.20"}, client_host="127.0.0.1")
        assert rest_api._is_trusted_network(request) is True

    def test_is_trusted_network_rejects_invalid_ip(self):
        request = _make_request(headers={"x-forwarded-for": "not-an-ip"})
        assert rest_api._is_trusted_network(request) is False

    def test_check_rest_auth_rejects_same_origin_header(self, monkeypatch):
        """sec-fetch-site: same-origin is forgeable and should not bypass auth."""
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "secret")
        request = _make_request(headers={"sec-fetch-site": "same-origin"})
        assert rest_api._check_rest_auth(request) is False

    def test_check_rest_auth_requires_valid_bearer(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "secret")
        request = _make_request(headers={"authorization": "Bearer wrong"})
        assert rest_api._check_rest_auth(request) is False

        request_ok = _make_request(headers={"authorization": "Bearer secret"})
        assert rest_api._check_rest_auth(request_ok) is True

    def test_check_rest_auth_rejects_untrusted_without_token_by_default(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", None)
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN", False)
        request = _make_request(headers={})
        assert rest_api._check_rest_auth(request) is False

    def test_check_rest_auth_allows_legacy_mode_without_token(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", None)
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN", True)
        request = _make_request(headers={})
        assert rest_api._check_rest_auth(request) is True

    def test_check_rest_auth_rejects_non_bearer_value(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "secret")
        request = _make_request(headers={"authorization": "Basic abc"})
        assert rest_api._check_rest_auth(request) is False


@pytest.mark.asyncio
class TestRestToolCall:
    async def test_missing_name_returns_400(self):
        request = _make_request(method="POST", path="/v1/tools/call", body={"arguments": {}})
        response = await rest_api.rest_tool_call(request)
        assert response.status_code == 400
        assert json.loads(response.body)["error"] == "Missing 'name' field"

    async def test_unknown_tool_returns_404(self):
        request = _make_request(method="POST", path="/v1/tools/call", body={"name": "not-a-tool"})
        response = await rest_api.rest_tool_call(request)
        assert response.status_code == 404
        assert "Unknown tool" in json.loads(response.body)["error"]

    async def test_returns_parsed_json_result(self, monkeypatch):
        async def fake_handler(_args):
            return [SimpleNamespace(text='{"ok": true, "n": 3}')]

        monkeypatch.setattr(rest_api, "HANDLERS", {"demo": fake_handler})
        request = _make_request(method="POST", path="/v1/tools/call", body={"name": "demo", "arguments": {"x": 1}})

        response = await rest_api.rest_tool_call(request)
        data = json.loads(response.body)
        assert response.status_code == 200
        assert data["success"] is True
        assert data["result"] == {"ok": True, "n": 3}

    async def test_returns_plain_text_when_handler_result_not_json(self, monkeypatch):
        async def fake_handler(_args):
            return [SimpleNamespace(text="ok plain text")]

        monkeypatch.setattr(rest_api, "HANDLERS", {"demo": fake_handler})
        request = _make_request(method="POST", path="/v1/tools/call", body={"name": "demo"})

        response = await rest_api.rest_tool_call(request)
        data = json.loads(response.body)
        assert data["success"] is True
        assert data["result"] == "ok plain text"

    async def test_returns_none_when_handler_returns_empty(self, monkeypatch):
        async def fake_handler(_args):
            return []

        monkeypatch.setattr(rest_api, "HANDLERS", {"demo": fake_handler})
        request = _make_request(method="POST", path="/v1/tools/call", body={"name": "demo"})

        response = await rest_api.rest_tool_call(request)
        data = json.loads(response.body)
        assert data["success"] is True
        assert data["result"] is None

    async def test_handler_exception_returns_500(self, monkeypatch):
        async def fake_handler(_args):
            raise RuntimeError("tool failed")

        monkeypatch.setattr(rest_api, "HANDLERS", {"demo": fake_handler})
        request = _make_request(method="POST", path="/v1/tools/call", body={"name": "demo"})

        response = await rest_api.rest_tool_call(request)
        assert response.status_code == 500
        assert "tool failed" in json.loads(response.body)["error"]

    async def test_unauthorized_uses_success_error_shape(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_check_rest_auth", lambda _req: False)
        response = await rest_api.rest_tool_call(
            _make_request(method="POST", path="/v1/tools/call", body={"name": "demo"})
        )
        data = json.loads(response.body)
        assert response.status_code == 401
        assert data == {"success": False, "error": "Unauthorized"}


@pytest.mark.asyncio
class TestRestGalleryImage:
    async def test_unauthorized_request_rejected(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_check_rest_auth", lambda _req: False)
        request = _make_request(path="/gallery/foo.png")
        response = await rest_api.rest_gallery_image(request)
        assert response.status_code == 401

    async def test_rejects_path_traversal_filename(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_check_rest_auth", lambda _req: True)
        request = _make_request(path="/gallery/../secret.txt", path_params={"filename": "../secret.txt"})
        response = await rest_api.rest_gallery_image(request)
        assert response.status_code == 400

    async def test_returns_404_for_missing_image(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rest_api, "_check_rest_auth", lambda _req: True)
        monkeypatch.setenv("HOME", str(tmp_path))
        request = _make_request(path="/gallery/missing.png", path_params={"filename": "missing.png"})
        response = await rest_api.rest_gallery_image(request)
        assert response.status_code == 404

    async def test_serves_png_with_cache_header(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rest_api, "_check_rest_auth", lambda _req: True)
        monkeypatch.setenv("HOME", str(tmp_path))
        drawings = Path(tmp_path) / ".anima" / "drawings"
        drawings.mkdir(parents=True)
        image = drawings / "lumen_drawing_20260207_190001_gestural.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        request = _make_request(path=f"/gallery/{image.name}", path_params={"filename": image.name})
        response = await rest_api.rest_gallery_image(request)

        assert response.status_code == 200
        assert response.media_type == "image/png"
        assert response.headers["Cache-Control"] == "max-age=3600"
        assert response.body.startswith(b"\x89PNG")


@pytest.mark.asyncio
class TestRestGallery:
    async def test_gallery_returns_paginated_drawings(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rest_api, "_check_rest_auth", lambda _req: True)
        monkeypatch.setenv("HOME", str(tmp_path))
        drawings = Path(tmp_path) / ".anima" / "drawings"
        drawings.mkdir(parents=True)
        (drawings / "lumen_drawing_20260207_190001_gestural.png").write_bytes(b"1")
        (drawings / "lumen_drawing_20260207_190002_geometric_manual.png").write_bytes(b"1")

        request = _make_request(path="/gallery", query="limit=1&offset=0")
        response = await rest_api.rest_gallery(request)
        data = json.loads(response.body)

        assert response.status_code == 200
        assert data["total"] == 2
        assert data["limit"] == 1
        assert data["has_more"] is True
        assert len(data["drawings"]) == 1


@pytest.mark.asyncio
class TestRestStateAndLayers:
    async def test_governance_state_fields_reject_incomplete_or_nonfinite_vectors(self):
        incomplete = rest_api._governance_state_fields({
            "body_eisv_projection": {"E": 0.2, "I": 0.3},
            "governance_eisv": {"E": 0.8, "I": 0.7, "S": 0.1},
        })
        nonfinite = rest_api._governance_state_fields({
            "governance_eisv": {
                "E": 0.8,
                "I": 0.7,
                "S": 0.1,
                "V": float("nan"),
            },
        })
        out_of_range = rest_api._governance_state_fields({
            "governance_eisv": {"E": 0.8, "I": 0.7, "S": 1.1, "V": 0.1},
        })

        assert incomplete["body_eisv_projection"] is None
        assert incomplete["governance_eisv"] is None
        assert nonfinite["governance_eisv"] is None
        assert out_of_range["governance_eisv"] is None

    async def test_rest_state_unauthorized(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_check_rest_auth", lambda _req: False)
        response = await rest_api.rest_state(_make_request(path="/state"))
        assert response.status_code == 401

    async def test_rest_state_returns_500_when_sensors_missing(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_check_rest_auth", lambda _req: True)
        with patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)):
            response = await rest_api.rest_state(_make_request(path="/state"))
        assert response.status_code == 500
        assert "Unable to read sensor data" in json.loads(response.body)["error"]

    async def test_rest_state_returns_payload(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_check_rest_auth", lambda _req: True)
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "secret")
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN", False)
        monkeypatch.setattr(rest_api, "_TRUSTED_PROXY_NETWORKS", [])
        feelings = {"mood": "calm"}
        anima = SimpleNamespace(warmth=0.4, clarity=0.5, stability=0.6, presence=0.7, feeling=lambda: feelings)
        readings = SimpleNamespace(
            cpu_temp_c=50,
            ambient_temp_c=22,
            light_lux=100,
            humidity_pct=40,
            pressure_hpa=1012,
            cpu_percent=15,
            memory_percent=35,
            disk_percent=55,
            timestamp="now",
        )
        identity = SimpleNamespace(
            name="Lumen",
            total_awakenings=3,
            total_alive_seconds=3600,
            alive_ratio=lambda: 0.5,
            age_seconds=lambda: 7200,
        )
        store = SimpleNamespace(get_identity=lambda: identity, get_session_alive_seconds=lambda: 600)
        activity = MagicMock()
        activity.get_sleep_summary.return_value = {"sessions": 1}
        eisv = SimpleNamespace(
            to_dict=lambda: {"E": 0.5, "I": 0.6, "S": 0.4, "V": -0.1}
        )

        with patch("anima_mcp.accessors._get_readings_and_anima", return_value=(readings, anima)), \
             patch("anima_mcp.accessors._get_store", return_value=store), \
             patch("anima_mcp.accessors._get_last_governance_decision", return_value={"action": "proceed", "source": "unitares"}), \
             patch("anima_mcp.accessors._get_activity", return_value=activity), \
             patch("anima_mcp.accessors._get_last_shm_data", return_value={
                 "activity": {
                     "level": "drowsy",
                     "brightness_multiplier": 0.5,
                     "reason": "broker body state",
                 },
                 "light_attribution": {
                     "mode": "shadow",
                     "status": "warming",
                     "external_lux_residual": None,
                     "used_by_clarity": False,
                 },
                 "clarity_attribution": {
                     "schema": "anima.clarity_attribution.v1",
                     "status": "ready",
                     "raw_value": 0.5,
                     "published_value": 0.5,
                     "components": {},
                     "attention": {
                         "light_salience": 1.3,
                         "used_in_measurement": False,
                     },
                 },
             }), \
             patch("anima_mcp.rest_api.extract_neural_bands", return_value={"beta": 0.2}), \
             patch("anima_mcp.rest_api.anima_to_body_eisv_projection", return_value=eisv):
            response = await rest_api.rest_state(_make_request(path="/state"))
            data = json.loads(response.body)

        assert data["name"] == "Lumen"
        assert data["mood"] == "calm"
        assert data["eisv"]["E"] == 0.5
        assert data["body_eisv_projection"] == data["eisv"]
        assert data["eisv_source"] == "body_eisv_projection_legacy_alias"
        assert data["body_anima"]["warmth"] == 0.4
        assert data["anima"] == data["body_anima"]
        assert data["state_space_provenance"]["anima"]["alias_of"] == "body_anima"
        assert data["governance"]["governance_eisv"] is None
        assert data["governance"]["connected"] is True
        assert data["api_security"]["token_configured"] is True
        assert data["api_security"]["mode"] == "token"
        assert "age_seconds" in data["governance"]
        assert data["activity"]["level"] == "drowsy"
        assert data["light_attribution"]["status"] == "warming"
        assert data["light_attribution"]["external_lux_residual"] is None
        assert data["light_attribution"]["used_by_clarity"] is False
        assert data["clarity_attribution"]["status"] == "ready"
        assert (
            data["clarity_attribution"]["attention"]["used_in_measurement"]
            is False
        )
        assert data["raw_light_lux"] == data["light"]
        assert data["light_composition"] == "room_light_plus_dotstar_glow"
        assert data["neural_provenance"]["physical_eeg"] is False
        assert data["neural_provenance"]["bands"]["alpha"]["independent"] is False
        activity.get_status.assert_not_called()

    async def test_rest_layers_includes_schema_hub_data(self, monkeypatch):
        feelings = {"mood": "steady"}
        anima = SimpleNamespace(warmth=0.4, clarity=0.5, stability=0.6, presence=0.7, feeling=lambda: feelings)
        readings = SimpleNamespace(
            ambient_temp_c=22, humidity_pct=40, light_lux=100, pressure_hpa=1012,
            cpu_temp_c=50, cpu_percent=20, memory_percent=30, disk_percent=40,
        )
        identity = SimpleNamespace(
            name="Lumen", total_awakenings=2, total_alive_seconds=1800,
            alive_ratio=lambda: 0.5, age_seconds=lambda: 3600,
        )
        store = SimpleNamespace(get_identity=lambda: identity, get_session_alive_seconds=lambda: 120)
        traj = SimpleNamespace(observation_count=25, attractor={"center": [0.1, 0.2, 0.3, 0.4], "variance": [0.01, 0.02, 0.03, 0.04]})
        hub = SimpleNamespace(schema_history=[1, 2], history_size=100, last_trajectory=traj, last_gap_delta=None)
        eisv = SimpleNamespace(
            to_dict=lambda: {"E": 0.4, "I": 0.5, "S": 0.3, "V": -0.1}
        )

        with patch("anima_mcp.accessors._get_readings_and_anima", return_value=(readings, anima)), \
             patch("anima_mcp.accessors._get_store", return_value=store), \
             patch("anima_mcp.accessors._get_last_governance_decision", return_value={}), \
             patch("anima_mcp.accessors._get_activity", return_value=None), \
             patch("anima_mcp.accessors._get_schema_hub", return_value=hub), \
             patch("anima_mcp.rest_api.extract_neural_bands", return_value={"alpha": 0.3}), \
             patch("anima_mcp.rest_api.anima_to_body_eisv_projection", return_value=eisv):
            response = await rest_api.rest_layers(_make_request(path="/layers"))
            data = json.loads(response.body)

        assert data["schema_hub"]["history_size"] == 2
        assert data["schema_hub"]["trajectory"]["observation_count"] == 25
        assert data["eisv"]["E"] == 0.4
        assert data["body_eisv_projection"] == data["eisv"]
        assert data["body_anima"]["warmth"] == data["anima"]["warmth"]
        assert data["neural_provenance"]["independent_views"] == 4
        assert data["physical"]["raw_light_lux"] == data["physical"]["light_lux"]
        assert (
            data["physical"]["light_lux_composition"]
            == "room_light_plus_dotstar_glow"
        )

    async def test_rest_layers_returns_500_when_sensors_missing(self):
        with patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)):
            response = await rest_api.rest_layers(_make_request(path="/layers"))
        assert response.status_code == 500
        assert "Unable to read sensor data" in json.loads(response.body)["error"]

    async def test_rest_state_preserves_null_pressure(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_check_rest_auth", lambda _req: True)
        anima = SimpleNamespace(warmth=0.4, clarity=0.5, stability=0.6, presence=0.7, feeling=lambda: {"mood": "calm"})
        readings = SimpleNamespace(
            cpu_temp_c=50, ambient_temp_c=22, light_lux=100, humidity_pct=40,
            pressure_hpa=None,
            cpu_percent=15, memory_percent=35, disk_percent=55, timestamp="now",
        )
        identity = SimpleNamespace(name="Lumen", total_awakenings=3, total_alive_seconds=3600,
                                   alive_ratio=lambda: 0.5, age_seconds=lambda: 7200)
        store = SimpleNamespace(get_identity=lambda: identity, get_session_alive_seconds=lambda: 600)
        activity = SimpleNamespace(get_status=lambda: {"level": "active"}, get_sleep_summary=lambda: {"sessions": 1})
        eisv = SimpleNamespace(
            to_dict=lambda: {"E": 0.5, "I": 0.6, "S": 0.4, "V": -0.1}
        )
        with patch("anima_mcp.accessors._get_readings_and_anima", return_value=(readings, anima)), \
             patch("anima_mcp.accessors._get_store", return_value=store), \
             patch("anima_mcp.accessors._get_last_governance_decision", return_value={"action": "proceed", "source": "unitares"}), \
             patch("anima_mcp.accessors._get_activity", return_value=activity), \
             patch("anima_mcp.rest_api.extract_neural_bands", return_value={"beta": 0.2}), \
             patch("anima_mcp.rest_api.anima_to_body_eisv_projection", return_value=eisv):
            response = await rest_api.rest_state(_make_request(path="/state"))
            data = json.loads(response.body)
        assert data["pressure"] is None, "offline BMP280 must surface as null, not 0 hPa"

    async def test_rest_layers_preserves_null_pressure(self):
        anima = SimpleNamespace(warmth=0.4, clarity=0.5, stability=0.6, presence=0.7, feeling=lambda: {"mood": "steady"})
        readings = SimpleNamespace(
            ambient_temp_c=22, humidity_pct=40, light_lux=100,
            pressure_hpa=None,
            cpu_temp_c=50, cpu_percent=20, memory_percent=30, disk_percent=40,
        )
        identity = SimpleNamespace(name="Lumen", total_awakenings=2, total_alive_seconds=1800,
                                   alive_ratio=lambda: 0.5, age_seconds=lambda: 3600)
        store = SimpleNamespace(get_identity=lambda: identity, get_session_alive_seconds=lambda: 120)
        hub = SimpleNamespace(schema_history=[], history_size=0, last_trajectory=None, last_gap_delta=None)
        eisv = SimpleNamespace(
            to_dict=lambda: {"E": 0.4, "I": 0.5, "S": 0.3, "V": -0.1}
        )
        with patch("anima_mcp.accessors._get_readings_and_anima", return_value=(readings, anima)), \
             patch("anima_mcp.accessors._get_store", return_value=store), \
             patch("anima_mcp.accessors._get_last_governance_decision", return_value={}), \
             patch("anima_mcp.accessors._get_activity", return_value=None), \
             patch("anima_mcp.accessors._get_schema_hub", return_value=hub), \
             patch("anima_mcp.rest_api.extract_neural_bands", return_value={"alpha": 0.3}), \
             patch("anima_mcp.rest_api.anima_to_body_eisv_projection", return_value=eisv):
            response = await rest_api.rest_layers(_make_request(path="/layers"))
            data = json.loads(response.body)
        assert data["physical"]["pressure_hpa"] is None, "offline BMP280 must surface as null, not 0 hPa"


@pytest.mark.asyncio
class TestRestHandlerDelegates:
    async def test_rest_health_detailed_returns_handler_data(self):
        with patch("anima_mcp.handlers.state_queries.handle_get_health", return_value=[SimpleNamespace(text='{"ok": true}')]):
            response = await rest_api.rest_health_detailed(_make_request(path="/health/detailed"))
            data = json.loads(response.body)
        assert data["ok"] is True

    async def test_rest_self_knowledge_delegates_with_query_params(self):
        with patch("anima_mcp.handlers.knowledge.handle_get_self_knowledge", return_value=[SimpleNamespace(text='{"insights": []}')]) as mock_handler:
            response = await rest_api.rest_self_knowledge(_make_request(path="/self-knowledge", query="category=ENVIRONMENT&limit=7"))
            data = json.loads(response.body)
        args = mock_handler.call_args[0][0]
        assert args["category"] == "ENVIRONMENT"
        assert args["limit"] == 7
        assert "insights" in data

    async def test_rest_growth_delegates_to_growth_handler(self):
        with patch("anima_mcp.handlers.knowledge.handle_get_growth", return_value=[SimpleNamespace(text='{"ok": true}')]):
            response = await rest_api.rest_growth(_make_request(path="/growth"))
            data = json.loads(response.body)
        assert data["ok"] is True

    async def test_rest_health_detailed_returns_no_data_when_handler_empty(self):
        with patch("anima_mcp.handlers.state_queries.handle_get_health", return_value=[]):
            response = await rest_api.rest_health_detailed(_make_request(path="/health/detailed"))
            data = json.loads(response.body)
        assert response.status_code == 500
        assert data["error"] == "no data"

    async def test_rest_self_knowledge_invalid_limit_returns_500(self):
        response = await rest_api.rest_self_knowledge(_make_request(path="/self-knowledge", query="limit=oops"))
        assert response.status_code == 500
        assert "invalid literal" in json.loads(response.body)["error"]

    async def test_rest_growth_returns_no_data_when_handler_empty(self):
        with patch("anima_mcp.handlers.knowledge.handle_get_growth", return_value=[]):
            response = await rest_api.rest_growth(_make_request(path="/growth"))
            data = json.loads(response.body)
        assert response.status_code == 500
        assert data["error"] == "no data"

    @pytest.mark.parametrize(
        "endpoint,args",
        [
            ("rest_qa", (_make_request(path="/qa"),)),
            ("rest_messages", (_make_request(path="/messages"),)),
            ("rest_answer", (_make_request(method="POST", path="/answer", body={"question_id": "q1", "answer": "A1"}),)),
            ("rest_message", (_make_request(method="POST", path="/message", body={"message": "hello"}),)),
            ("rest_learning", (_make_request(path="/learning"),)),
            ("rest_voice", (_make_request(path="/voice"),)),
            ("rest_health_detailed", (_make_request(path="/health/detailed"),)),
            ("rest_self_knowledge", (_make_request(path="/self-knowledge"),)),
            ("rest_growth", (_make_request(path="/growth"),)),
            ("rest_layers", (_make_request(path="/layers"),)),
        ],
    )
    async def test_sensitive_endpoints_reject_unauthorized(self, monkeypatch, endpoint, args):
        monkeypatch.setattr(rest_api, "_check_rest_auth", lambda _req: False)
        handler = getattr(rest_api, endpoint)
        response = await handler(*args)
        assert response.status_code == 401
        assert json.loads(response.body)["error"] == "Unauthorized"


@pytest.mark.asyncio
class TestRestQaAndMessages:
    async def test_rest_qa_builds_pairs_and_unanswered_counts(self):
        q1 = SimpleNamespace(message_id="q1", msg_type="question", text="Q1", answered=False, timestamp=1)
        q2 = SimpleNamespace(message_id="q2", msg_type="question", text="Q2", answered=True, timestamp=2)
        a2 = SimpleNamespace(message_id="a2", msg_type="agent", text="A2", author="agent", timestamp=3, responds_to="q2")
        board = SimpleNamespace(_messages=[q1, q2, a2], _load=MagicMock())

        with patch("anima_mcp.messages.get_board", return_value=board), \
             patch("anima_mcp.messages.MESSAGE_TYPE_QUESTION", "question"):
            response = await rest_api.rest_qa(_make_request(path="/qa", query="limit=10"))
            data = json.loads(response.body)

        assert data["total"] == 2
        assert data["unanswered"] == 1
        assert len(data["questions"]) == 2
        assert data["questions"][0]["id"] == "q2"  # newest first after reverse
        assert data["record_semantics"]["current_state_authority"] == "none"
        assert data["questions"][0]["epistemic_status"] == "answered_record"
        assert (
            data["questions"][0]["answer"]["record_role"]
            == "historical_utterance"
        )
        assert data["questions"][1]["epistemic_status"] == "open_question"

    async def test_rest_qa_distinguishes_expiry_from_an_actual_answer(self):
        expired = SimpleNamespace(
            message_id="expired",
            msg_type="question",
            text="Is anyone there?",
            answered=False,
            expired_at=2,
            timestamp=1,
        )
        board = SimpleNamespace(_messages=[expired], _load=MagicMock())

        with patch("anima_mcp.messages.get_board", return_value=board), patch(
            "anima_mcp.messages.MESSAGE_TYPE_QUESTION",
            "question",
        ):
            response = await rest_api.rest_qa(_make_request(path="/qa"))
            data = json.loads(response.body)

        record = data["questions"][0]
        assert data["unanswered"] == 1
        assert record["answered"] is False
        assert record["status"] == "expired"
        assert record["expired_unanswered"] is True
        assert record["epistemic_status"] == "open_question"

    async def test_rest_qa_keeps_self_answered_question_open_with_self_answer(self):
        q1 = SimpleNamespace(
            message_id="q1",
            msg_type="question",
            text="What changes when the room cools?",
            answered=False,
            expired_at=None,
            timestamp=1,
        )
        self_a1 = SimpleNamespace(
            message_id="sa1",
            msg_type="agent",
            text="Warmth averaged 0.512.",
            author="lumen",
            timestamp=2,
            responds_to="q1",
        )
        board = SimpleNamespace(_messages=[q1, self_a1], _load=MagicMock())

        with patch("anima_mcp.messages.get_board", return_value=board), patch(
            "anima_mcp.messages.MESSAGE_TYPE_QUESTION",
            "question",
        ):
            response = await rest_api.rest_qa(_make_request(path="/qa"))
            data = json.loads(response.body)

        record = data["questions"][0]
        assert data["unanswered"] == 1
        assert record["answered"] is False
        assert record["answer"] is None
        assert record["self_answer"]["author"] == "lumen"
        assert record["self_answer"]["text"] == "Warmth averaged 0.512."
        assert record["epistemic_status"] == "open_question"

    async def test_rest_messages_returns_serialized_messages(self):
        m1 = SimpleNamespace(message_id="m1", text="hello", msg_type="user", author="u", timestamp=1, responds_to=None)
        m2 = SimpleNamespace(message_id="m2", text="world", msg_type="agent", author="a", timestamp=2, responds_to="m1")
        board = SimpleNamespace(_messages=[m1, m2])

        with patch("anima_mcp.messages.get_recent_messages", return_value=[m1, m2]), \
             patch("anima_mcp.messages.get_board", return_value=board):
            response = await rest_api.rest_messages(_make_request(path="/messages", query="limit=2"))
            data = json.loads(response.body)

        assert data["total"] == 2
        assert data["returned"] == 2
        assert data["messages"][1]["responds_to"] == "m1"


@pytest.mark.asyncio
class TestRestLearningAndSchemaData:
    async def test_rest_learning_reads_db_and_computes_metrics(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import datetime, timedelta

        db_path = tmp_path / "anima.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE identity (name TEXT, total_awakenings INTEGER, total_alive_seconds REAL, born_at TEXT)")
        conn.execute("CREATE TABLE state_history (timestamp TEXT, warmth REAL, clarity REAL, stability REAL, presence REAL)")
        now = datetime.now()
        conn.execute(
            "INSERT INTO identity VALUES (?, ?, ?, ?)",
            ("Lumen", 4, 7200.0, (now - timedelta(days=2)).isoformat()),
        )
        conn.execute(
            "INSERT INTO state_history VALUES (?, ?, ?, ?, ?)",
            ((now - timedelta(hours=20)).isoformat(), 0.4, 0.5, 0.6, 0.7),
        )
        conn.execute(
            "INSERT INTO state_history VALUES (?, ?, ?, ?, ?)",
            ((now - timedelta(hours=2)).isoformat(), 0.5, 0.6, 0.7, 0.8),
        )
        conn.commit()
        conn.close()

        monkeypatch.setenv("ANIMA_DB", str(db_path))
        response = await rest_api.rest_learning(_make_request(path="/learning"))
        data = json.loads(response.body)

        assert data["name"] == "Lumen"
        assert data["samples_24h"] >= 2
        assert data["awakenings"] == 4

    async def test_rest_schema_data_returns_history_and_trajectory(self):
        schema_obj = SimpleNamespace(
            to_dict=lambda: {"nodes": []},
            timestamp=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
            nodes=[],
            edges=[],
        )
        traj_obj = SimpleNamespace(
            summary=lambda: {"stability": 0.8},
            preferences={"p": 1},
            beliefs={"b": 1},
            attractor={"a": 1},
            recovery={"r": 1},
            relational={"rel": 1},
        )
        gap_obj = SimpleNamespace(duration_seconds=3600, was_gap=True, was_restore=False, anima_delta=0.2, beliefs_decayed=True)
        hub = SimpleNamespace(schema_history=[schema_obj], last_trajectory=traj_obj, last_gap_delta=gap_obj)

        with patch("anima_mcp.rest_api._check_rest_auth", return_value=True), \
             patch("anima_mcp.accessors._get_schema_hub", return_value=hub), \
             patch("anima_mcp.accessors._get_store", return_value=None), \
             patch("anima_mcp.accessors._get_growth", return_value=None), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)):
            response = await rest_api.rest_schema_data(_make_request(path="/schema-data"))
            data = json.loads(response.body)

        assert data["schema"] == {"nodes": []}
        assert data["trajectory"]["stability"] == 0.8
        assert len(data["history"]) == 1
        assert data["gap"]["was_gap"] is True

    async def test_rest_learning_returns_500_when_db_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANIMA_DB", str(tmp_path / "missing.db"))
        monkeypatch.setattr(rest_api.Path, "home", staticmethod(lambda: tmp_path))
        response = await rest_api.rest_learning(_make_request(path="/learning"))
        assert response.status_code == 500
        assert "No identity database" in json.loads(response.body)["error"]

    async def test_rest_schema_data_unauthorized(self):
        with patch("anima_mcp.rest_api._check_rest_auth", return_value=False):
            response = await rest_api.rest_schema_data(_make_request(path="/schema-data"))
        assert response.status_code == 401


@pytest.mark.asyncio
class TestRestMessageAnswerVoice:
    async def test_rest_answer_delegates_and_normalizes_author(self):
        with patch(
            "anima_mcp.growth.normalize_visitor_identity",
            return_value=("id", "Kenny", {}),
        ), patch(
            "anima_mcp.handlers.communication.handle_lumen_qa",
            return_value=[SimpleNamespace(text='{"success": true, "answered": true}')],
        ):
            response = await rest_api.rest_answer(
                _make_request(
                    method="POST",
                    path="/answer",
                    body={"question_id": "q1", "answer": "A1", "author": "dashboard"},
                )
            )
            data = json.loads(response.body)
        assert data["success"] is True

    async def test_rest_message_includes_responds_to(self):
        with patch(
            "anima_mcp.growth.normalize_visitor_identity",
            return_value=("id", "Visitor", {}),
        ), patch(
            "anima_mcp.handlers.communication.handle_post_message",
            return_value=[SimpleNamespace(text='{"success": true}')],
        ) as handler:
            response = await rest_api.rest_message(
                _make_request(
                    method="POST",
                    path="/message",
                    body={"message": "hello", "author": "dashboard", "responds_to": "q1"},
                )
            )
            data = json.loads(response.body)
        payload = handler.call_args[0][0]
        assert payload["responds_to"] == "q1"
        assert data["success"] is True

    async def test_rest_voice_defaults_when_handler_returns_empty(self):
        with patch("anima_mcp.handlers.communication.handle_configure_voice", return_value=[]):
            response = await rest_api.rest_voice(_make_request(path="/voice"))
            data = json.loads(response.body)
        assert data["mode"] == "text"


class TestExternalHostGate:
    """The tunnel path must never qualify as a trusted network.

    Regression lock for the 2026-08-25 lockout/bypass pair: uvicorn ran with
    ``forwarded_allow_ips="*"``, so ``request.client.host`` came from
    ``X-Forwarded-For[0]`` -- a value the caller writes. Anyone could send
    ``X-Forwarded-For: 127.0.0.1`` and read Lumen's whole REST surface,
    ``/v1/tools/call`` included, while the operator's browser (carrying its
    real address) was refused. Host is checked instead of IP because
    cloudflared routes by hostname.
    """

    def test_external_host_is_never_trusted(self):
        request = _make_request(
            client_host="127.0.0.1",
            headers={"host": "lumen.cirwel.org"},
        )
        assert rest_api._is_trusted_network(request) is False

    def test_external_host_ignores_spoofed_forwarded_for(self, monkeypatch):
        monkeypatch.setattr(
            rest_api, "_TRUSTED_PROXY_NETWORKS", rest_api._parse_networks("127.0.0.1/32")
        )
        request = _make_request(
            client_host="127.0.0.1",
            headers={"host": "lumen.cirwel.org", "x-forwarded-for": "127.0.0.1"},
        )
        assert rest_api._is_trusted_network(request) is False

    def test_external_host_with_port_still_gated(self):
        request = _make_request(
            client_host="127.0.0.1",
            headers={"host": "LUMEN.CIRWEL.ORG:443"},
        )
        assert rest_api._is_trusted_network(request) is False

    def test_local_host_still_trusted(self):
        """Tailscale and loopback callers must keep working without a token."""
        request = _make_request(
            client_host="100.64.0.10",
            headers={"host": "100.64.0.10:8766"},
        )
        assert rest_api._is_trusted_network(request) is True

    def test_external_host_with_valid_token_allowed(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        request = _make_request(
            client_host="203.0.113.7",
            headers={"host": "lumen.cirwel.org", "authorization": "Bearer s3cret"},
        )
        assert rest_api._check_rest_auth(request) is True

    def test_external_host_without_token_refused_even_in_legacy_mode(self, monkeypatch):
        """Legacy permissive mode is about *missing config*, not about the
        public internet. With no token configured the tunnel stays shut."""
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_ALLOW_UNAUTH_IF_NO_TOKEN", True)
        request = _make_request(
            client_host="203.0.113.7", headers={"host": "lumen.cirwel.org"}
        )
        assert rest_api._check_rest_auth(request) is False


class TestTokenPresentation:
    def test_cookie_token_accepted(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        request = _make_request(
            client_host="203.0.113.7",
            headers={"host": "lumen.cirwel.org", "cookie": "anima_token=s3cret"},
        )
        assert rest_api._check_rest_auth(request) is True

    def test_query_token_accepted(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        request = _make_request(
            client_host="203.0.113.7",
            headers={"host": "lumen.cirwel.org"},
            query="token=s3cret",
        )
        assert rest_api._check_rest_auth(request) is True

    def test_wrong_token_refused(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        for headers, query in (
            ({"host": "lumen.cirwel.org", "authorization": "Bearer nope"}, ""),
            ({"host": "lumen.cirwel.org", "cookie": "anima_token=nope"}, ""),
            ({"host": "lumen.cirwel.org"}, "token=nope"),
        ):
            request = _make_request(client_host="203.0.113.7", headers=headers, query=query)
            assert rest_api._check_rest_auth(request) is False


@pytest.mark.asyncio
class TestGuardedPages:
    async def test_dashboard_refused_without_token(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        request = _make_request(
            path="/dashboard",
            client_host="203.0.113.7",
            headers={"host": "lumen.cirwel.org"},
        )
        response = await rest_api.dashboard(request)
        assert response.status_code == 401

    async def test_dashboard_query_token_sets_cookie_and_redirects(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        request = _make_request(
            path="/dashboard",
            client_host="203.0.113.7",
            headers={"host": "lumen.cirwel.org"},
            query="token=s3cret",
        )
        response = await rest_api.dashboard(request)
        assert response.status_code == 303
        assert response.headers["location"] == "/dashboard"
        set_cookie = response.headers["set-cookie"]
        assert "anima_token=s3cret" in set_cookie
        assert "HttpOnly" in set_cookie
        assert "Path=/" in set_cookie

    async def test_dashboard_cookie_serves_page(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        monkeypatch.setattr(
            rest_api, "_serve_html_page", lambda *_a, **_k: rest_api.HTMLResponse("page")
        )
        request = _make_request(
            path="/dashboard",
            client_host="203.0.113.7",
            headers={"host": "lumen.cirwel.org", "cookie": "anima_token=s3cret"},
        )
        response = await rest_api.dashboard(request)
        assert response.status_code == 200

    async def test_local_dashboard_needs_no_token(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        monkeypatch.setattr(
            rest_api, "_serve_html_page", lambda *_a, **_k: rest_api.HTMLResponse("page")
        )
        request = _make_request(
            path="/dashboard",
            client_host="100.64.0.10",
            headers={"host": "100.64.0.10:8766"},
        )
        response = await rest_api.dashboard(request)
        assert response.status_code == 200


class TestForwardedAllowIpsNotWildcard:
    """``forwarded_allow_ips="*"`` is what made X-Forwarded-For authoritative
    for any caller. Lock the wildcard out of the source."""

    def test_uvicorn_config_does_not_trust_every_proxy(self):
        source = (Path(rest_api.__file__).parent / "server.py").read_text()
        assert 'forwarded_allow_ips="*"' not in source
        assert "ANIMA_FORWARDED_ALLOW_IPS" in source


class TestPublicReads:
    """ANIMA_HTTP_PUBLIC_READS opens what reports and keeps shut what acts.

    The dashboard's ambient surface -- temperature, drawings, mood -- shared a
    single switch with POST /v1/tools/call, which reaches say(),
    configure_voice(always_listening) and set_calibration. Locking an operator
    out of a thermometer to protect a microphone is the wrong shape.

    These assert against the real call sites, not against
    ``_require_rest_auth`` in isolation: an earlier version of this class
    tested the helper directly and passed happily while /v1/tools/call was
    marked ``read=True``.
    """

    # Endpoint -> may it open under public reads? Every REST route that takes
    # an auth decision belongs here, so adding one forces a ruling.
    READ_ENDPOINTS = frozenset({
        "rest_state", "rest_qa", "rest_messages", "rest_learning", "rest_voice",
        "rest_gallery", "rest_gallery_image", "rest_health_detailed",
        "rest_self_knowledge", "rest_growth", "rest_layers", "rest_schema_data",
    })
    GATED_ENDPOINTS = frozenset({"rest_tool_call", "rest_answer", "rest_message"})

    def test_call_sites_match_the_ruling(self):
        """Pin which functions pass read=True, at the AST level.

        A source-level check because the acting endpoints cannot be exercised
        for real in a unit test, and a mismarked one fails open silently.
        """
        import ast

        tree = ast.parse(Path(rest_api.__file__).read_text())
        marked, unmarked = set(), set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_require_rest_auth"
                ):
                    kwargs = {k.arg for k in call.keywords}
                    (marked if "read" in kwargs else unmarked).add(node.name)

        assert marked == self.READ_ENDPOINTS, (
            f"read-marked set drifted: unexpected={marked - self.READ_ENDPOINTS}, "
            f"missing={self.READ_ENDPOINTS - marked}"
        )
        assert unmarked == self.GATED_ENDPOINTS, (
            f"gated set drifted: unexpected={unmarked - self.GATED_ENDPOINTS}, "
            f"missing={self.GATED_ENDPOINTS - unmarked}"
        )

    @pytest.fixture
    def public(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_PUBLIC_READS", True)
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")

    def _external(self, path="/", **kw):
        return _make_request(
            path=path,
            client_host="203.0.113.7",
            headers={"host": "lumen.cirwel.org"},
            **kw,
        )

    def test_read_opens(self, public):
        assert rest_api._require_rest_auth(self._external(), read=True) is None

    def test_default_is_closed(self, monkeypatch):
        """A fresh install must not serve reads to the internet."""
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        assert rest_api._require_rest_auth(self._external(), read=True) is not None

    @pytest.mark.asyncio
    async def test_tool_call_stays_shut(self, public):
        request = self._external(
            "/v1/tools/call", method="POST", body={"name": "get_state", "arguments": {}}
        )
        response = await rest_api.rest_tool_call(request)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_answer_stays_shut(self, public):
        request = self._external("/answer", method="POST", body={"answer": "hi"})
        response = await rest_api.rest_answer(request)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_message_stays_shut(self, public):
        request = self._external("/message", method="POST", body={"message": "hi"})
        response = await rest_api.rest_message(request)
        assert response.status_code == 401


class TestApiSecurityNotLeaked:
    def _req(self, **headers):
        base = {"host": "lumen.cirwel.org"}
        base.update(headers)
        return _make_request(client_host="203.0.113.7", headers=base)

    def test_anonymous_reader_gets_no_gate_posture(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_PUBLIC_READS", True)
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        assert rest_api._api_security_fields(self._req()) == {}

    def test_authorized_caller_sees_gate_posture(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_PUBLIC_READS", True)
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        fields = rest_api._api_security_fields(
            self._req(authorization="Bearer s3cret")
        )
        assert fields["api_security"]["mode"] == "token"
        assert fields["api_security"]["public_reads"] is True

    def test_local_caller_sees_gate_posture(self, monkeypatch):
        monkeypatch.setattr(rest_api, "_ANIMA_HTTP_API_TOKEN", "s3cret")
        request = _make_request(
            client_host="100.64.0.10", headers={"host": "100.64.0.10:8766"}
        )
        assert "api_security" in rest_api._api_security_fields(request)
