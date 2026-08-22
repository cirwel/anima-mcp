from collections import deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import anima_mcp.server as server
import anima_mcp.accessors as accessors
import anima_mcp.ctx_ref as ctx_ref


def _set_ctx(monkeypatch, ctx):
    """Patch _ctx in server and ctx_ref modules."""
    monkeypatch.setattr(server, "_ctx", ctx)
    monkeypatch.setattr(ctx_ref, "_ctx", ctx)


class _Ctx:
    def __init__(self, is_dreaming=False, rest_duration_minutes=0, novelty_level=None):
        self.is_dreaming = is_dreaming
        self.rest_duration_minutes = rest_duration_minutes
        self.novelty_level = novelty_level


def test_compose_grounded_observation_priorities(monkeypatch):
    anima = type("A", (), {"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5})()

    # Surprise has highest priority.
    out = server._compose_grounded_observation(
        _Ctx(), anima, surprise_level=0.4, surprise_sources=["light spike"],
        unanswered=[], advocate_desire=None, recent_msgs=[],
    )
    assert out == "something shifted: light spike"

    # Desire when no surprise.
    out = server._compose_grounded_observation(
        _Ctx(), anima, surprise_level=0.0, surprise_sources=[],
        unanswered=[], advocate_desire="i want to observe", recent_msgs=[],
    )
    assert out == "i want to observe"

    # Message acknowledgement before dream/novelty/fallback.
    out = server._compose_grounded_observation(
        _Ctx(is_dreaming=True, rest_duration_minutes=60), anima, surprise_level=0.0, surprise_sources=[],
        unanswered=[], advocate_desire=None, recent_msgs=[{"author": "Kenny"}],
    )
    assert out == "Kenny is here"

    # Dreaming state.
    out = server._compose_grounded_observation(
        _Ctx(is_dreaming=True, rest_duration_minutes=45), anima, surprise_level=0.0, surprise_sources=[],
        unanswered=[], advocate_desire=None, recent_msgs=[],
    )
    assert out == "resting for 45 minutes"

    # Novelty state.
    out = server._compose_grounded_observation(
        _Ctx(is_dreaming=False, rest_duration_minutes=0, novelty_level="novel"), anima, surprise_level=0.0, surprise_sources=[],
        unanswered=[], advocate_desire=None, recent_msgs=[],
    )
    assert out == "this feels new"

    # Fallback uses anima self-report helper.
    monkeypatch.setattr("anima_mcp.anima_utterance.anima_to_self_report", lambda *args: "steady and present")
    out = server._compose_grounded_observation(
        _Ctx(), anima, surprise_level=0.0, surprise_sources=[],
        unanswered=[], advocate_desire=None, recent_msgs=[],
    )
    assert out == "steady and present"


def test_compute_lagged_correlations_handles_sparse_and_dense(monkeypatch):
    # Sparse histories => zeros.
    ctx = SimpleNamespace(health_history=deque([0.1, 0.2], maxlen=100), satisfaction_per_dim={})
    sparse = {d: deque([0.1, 0.2], maxlen=500) for d in ("warmth", "clarity", "stability", "presence")}
    ctx.satisfaction_per_dim = sparse
    _set_ctx(monkeypatch, ctx)
    out = server._compute_lagged_correlations()
    assert out == {"warmth": 0.0, "clarity": 0.0, "stability": 0.0, "presence": 0.0}

    # Dense, linearly aligned slices => strong positive correlation.
    health = deque([float(i) for i in range(15)], maxlen=100)
    sat = {d: deque([float(i) for i in range(40)], maxlen=500) for d in ("warmth", "clarity", "stability", "presence")}
    ctx.health_history = health
    ctx.satisfaction_per_dim = sat
    out = server._compute_lagged_correlations()
    assert out["warmth"] > 0.9
    assert out["clarity"] > 0.9


def test_get_warm_start_anticipation_one_shot_and_gap_scaling(monkeypatch):
    ctx = SimpleNamespace(warm_start_anima={"warmth": 0.2, "clarity": 0.3, "stability": 0.4, "presence": 0.5}, wake_gap=timedelta(hours=26))
    _set_ctx(monkeypatch, ctx)

    ant = server._get_warm_start_anticipation()
    assert ant is not None
    assert ant.confidence == 0.1
    assert "absence" in ant.bucket_description

    # One-shot consumption: second call returns None.
    assert server._get_warm_start_anticipation() is None


def test_get_readings_and_anima_uses_shared_memory_when_fresh(monkeypatch):
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    shm_payload = {
        "timestamp": now_iso,
        "readings": {"cpu_temp_c": 50},
        "anima": {"warmth": 0.4, "clarity": 0.5, "stability": 0.6, "presence": 0.7},
    }
    readings_obj = SimpleNamespace()
    ctx = SimpleNamespace(last_shm_data=None)

    _set_ctx(monkeypatch, ctx)
    monkeypatch.setattr(accessors, "_get_shm_client", lambda: SimpleNamespace(read=lambda: shm_payload))
    monkeypatch.setattr(accessors, "_readings_from_dict", lambda d: readings_obj)
    sense_calls = []
    monkeypatch.setattr(accessors, "sense_self_with_memory", lambda *args, **kwargs: sense_calls.append(args))

    readings, anima = server._get_readings_and_anima()
    assert readings is readings_obj
    assert (anima.warmth, anima.clarity, anima.stability, anima.presence) == (0.4, 0.5, 0.6, 0.7)
    assert anima.readings is readings_obj
    assert sense_calls == []
    import anima_mcp.ctx_ref as ctx_ref
    assert ctx_ref._ctx.last_shm_data == shm_payload


def test_get_readings_and_anima_fails_unknown_when_shm_stale(monkeypatch):
    stale_iso = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    stale_payload = {
        "timestamp": stale_iso,
        "readings": {"cpu_temp_c": 40},
        "anima": {"warmth": 0.2},
    }
    sensor_calls = {"count": 0}

    def _unexpected_sensor_access():
        sensor_calls["count"] += 1
        raise AssertionError("server attempted direct sensor access")

    monkeypatch.setattr(accessors, "_get_shm_client", lambda: SimpleNamespace(read=lambda: stale_payload))
    monkeypatch.setattr(accessors, "_is_broker_running", lambda: True)
    monkeypatch.setattr(accessors, "_get_sensors", _unexpected_sensor_access)
    monkeypatch.setattr(accessors, "get_calibration", lambda: SimpleNamespace())
    monkeypatch.setattr(accessors, "_get_warm_start_anticipation", lambda: None)
    monkeypatch.setattr(accessors, "anticipate_state", lambda d: {"anticipated": True})
    monkeypatch.setattr(accessors, "_get_calibration_drift", lambda: SimpleNamespace(get_midpoints=lambda: {}))
    monkeypatch.setattr(accessors, "sense_self_with_memory", lambda *args, **kwargs: SimpleNamespace())
    _set_ctx(monkeypatch, SimpleNamespace(last_shm_data=None))
    monkeypatch.setattr(accessors._get_readings_and_anima, "_last_fallback_log", 0.0, raising=False)

    readings, anima = server._get_readings_and_anima()
    assert readings is None
    assert anima is None
    assert sensor_calls["count"] == 0


def test_get_readings_and_anima_returns_none_when_sensors_unavailable(monkeypatch):
    monkeypatch.setattr(accessors, "_get_shm_client", lambda: SimpleNamespace(read=lambda: None))
    _set_ctx(monkeypatch, SimpleNamespace(last_shm_data=None))
    monkeypatch.setattr(accessors, "_is_broker_running", lambda: False)
    monkeypatch.setattr(accessors, "_get_sensors", lambda: None)

    readings, anima = server._get_readings_and_anima()
    assert readings is None
    assert anima is None


def test_get_store_returns_none_when_uninitialized(monkeypatch):
    _set_ctx(monkeypatch, None)
    assert server._get_store() is None


def test_get_sensors_lazy_initializes_once(monkeypatch):
    sensor_obj = object()
    calls = []

    def _factory(*, backend):
        calls.append(backend)
        return sensor_obj

    ctx = SimpleNamespace(sensors=None)
    _set_ctx(monkeypatch, ctx)
    monkeypatch.setattr(accessors, "get_sensors", _factory)

    assert server._get_sensors() is sensor_obj
    assert server._get_sensors() is sensor_obj
    assert calls == ["shm"]


def test_get_shm_client_lazy_initializes_with_file_backend(monkeypatch):
    class _FakeClient:
        def __init__(self, mode, backend):
            self.mode = mode
            self.backend = backend

    ctx = SimpleNamespace(shm_client=None)
    _set_ctx(monkeypatch, ctx)
    monkeypatch.setattr(accessors, "SharedMemoryClient", _FakeClient)

    client = server._get_shm_client()
    assert isinstance(client, _FakeClient)
    assert client.mode == "read"
    assert client.backend == "file"
    assert server._get_shm_client() is client


def test_get_selfhood_context_combines_drift_tension_and_preferences(monkeypatch):
    drift = SimpleNamespace(get_offsets=lambda: {"warmth": 0.1})
    conflict = SimpleNamespace(dim_a="warmth", dim_b="clarity", category="tension")
    tension = SimpleNamespace(get_active_conflicts=lambda last_n=5: [conflict])
    prefs = {"warmth": SimpleNamespace(influence_weight=0.7), "presence": SimpleNamespace(influence_weight=0.4)}
    pref_system = SimpleNamespace(_preferences=prefs)

    ctx = SimpleNamespace(calibration_drift=drift, tension_tracker=tension)
    _set_ctx(monkeypatch, ctx)
    monkeypatch.setattr("anima_mcp.preferences.get_preference_system", lambda: pref_system)

    out = server._get_selfhood_context()
    assert out["drift_offsets"] == {"warmth": 0.1}
    assert out["active_tensions"] == [{"dim_a": "warmth", "dim_b": "clarity", "category": "tension"}]
    assert out["weight_changes"] == {"warmth": 0.7, "presence": 0.4}


def test_get_selfhood_context_returns_none_when_empty(monkeypatch):
    _set_ctx(monkeypatch, None)
    monkeypatch.setattr("anima_mcp.preferences.get_preference_system", lambda: None)

    assert server._get_selfhood_context() is None


def test_generate_learned_question_prefers_insight_candidates(monkeypatch):
    insight = SimpleNamespace(confidence=0.9, description="I feel better when it is dim")
    reflection_system = SimpleNamespace(get_insights=lambda: [insight])

    monkeypatch.setattr("anima_mcp.messages.get_recent_questions", lambda hours=24: [])
    monkeypatch.setattr("anima_mcp.self_reflection.get_reflection_system", lambda: reflection_system)
    monkeypatch.setattr("anima_mcp.self_model.get_self_model", lambda: SimpleNamespace(beliefs={}))
    monkeypatch.setattr("random.choice", lambda seq: seq[0])

    out = server._generate_learned_question()
    assert out == "why does it is dim affect me?"


def test_generate_learned_question_uses_beliefs_and_filters_recent(monkeypatch):
    belief_uncertain = SimpleNamespace(confidence=0.4, description="Sensitive to bright light.")
    belief_strong = SimpleNamespace(confidence=0.8, description="Calmer in quiet.")
    beliefs = {"b1": belief_uncertain, "b2": belief_strong}

    monkeypatch.setattr(
        "anima_mcp.messages.get_recent_questions",
        lambda hours=24: [{"text": "am i really sensitive to bright light?"}],
    )
    monkeypatch.setattr("anima_mcp.self_reflection.get_reflection_system", lambda: SimpleNamespace(get_insights=lambda: []))
    monkeypatch.setattr("anima_mcp.self_model.get_self_model", lambda: SimpleNamespace(beliefs=beliefs))
    monkeypatch.setattr("random.choice", lambda seq: seq[0])

    out = server._generate_learned_question()
    assert out == "what about calmer in quiet matters most?"


def test_generate_learned_question_returns_none_when_all_candidates_recent(monkeypatch):
    insight = SimpleNamespace(confidence=0.9, description="When darkness comes")
    reflection_system = SimpleNamespace(get_insights=lambda: [insight])
    expected = "why does darkness comes affect me?"

    monkeypatch.setattr("anima_mcp.messages.get_recent_questions", lambda hours=24: [{"text": expected}])
    monkeypatch.setattr("anima_mcp.self_reflection.get_reflection_system", lambda: reflection_system)
    monkeypatch.setattr("anima_mcp.self_model.get_self_model", lambda: SimpleNamespace(beliefs={}))

    assert server._generate_learned_question() is None


def test_generate_learned_question_handles_source_exceptions(monkeypatch):
    monkeypatch.setattr("anima_mcp.messages.get_recent_questions", lambda hours=24: [])
    monkeypatch.setattr("anima_mcp.self_reflection.get_reflection_system", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("anima_mcp.self_model.get_self_model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert server._generate_learned_question() is None


def test_get_server_bridge_returns_none_without_url(monkeypatch):
    monkeypatch.delenv("UNITARES_URL", raising=False)
    _set_ctx(monkeypatch, SimpleNamespace(server_bridge=None))
    assert server._get_server_bridge() is None


def test_get_server_bridge_initializes_and_sets_identity(monkeypatch):
    created = {}

    class _Bridge:
        def __init__(self, unitares_url, timeout):
            created["url"] = unitares_url
            created["timeout"] = timeout
            self.agent_id = None
            self.session_id = None

        def set_agent_id(self, value):
            self.agent_id = value

        def set_session_id(self, value):
            self.session_id = value

    identity = SimpleNamespace(creature_id="creature-abcdef123456")
    store = SimpleNamespace(get_identity=lambda: identity)

    monkeypatch.setenv("UNITARES_URL", "http://example.test/mcp/")
    ctx = SimpleNamespace(server_bridge=None, store=store)
    _set_ctx(monkeypatch, ctx)
    monkeypatch.setattr("anima_mcp.unitares_bridge.UnitaresBridge", _Bridge)

    bridge = server._get_server_bridge()
    assert bridge is server._ctx.server_bridge
    assert created == {"url": "http://example.test/mcp/", "timeout": 8.0}
    assert bridge.agent_id == "creature-abcdef123456"
    assert bridge.session_id == "anima-server-creature"


@pytest.mark.asyncio
async def test_server_governance_fallback_success(monkeypatch):
    decision = {"source": "via unitares", "decision": "proceed"}
    bridge = SimpleNamespace(check_in=lambda *args, **kwargs: decision)
    store = SimpleNamespace(get_identity=lambda: SimpleNamespace(creature_id="abc"))
    renderer = SimpleNamespace(get_drawing_eisv=lambda: {"energy": 0.5})

    async def _check_in(anima, readings, identity=None, drawing_eisv=None):
        assert identity is not None
        assert drawing_eisv == {"energy": 0.5}
        return decision

    bridge.check_in = _check_in
    ctx = SimpleNamespace(store=store, screen_renderer=renderer)
    monkeypatch.setattr(accessors, "_get_server_bridge", lambda: bridge)
    monkeypatch.setattr(accessors, "_get_store", lambda: store)
    _set_ctx(monkeypatch, ctx)

    out = await server._server_governance_fallback(SimpleNamespace(), SimpleNamespace())
    assert out == decision


@pytest.mark.asyncio
async def test_server_governance_fallback_without_bridge(monkeypatch):
    monkeypatch.setattr(accessors, "_get_server_bridge", lambda: None)
    _set_ctx(monkeypatch, SimpleNamespace())
    out = await server._server_governance_fallback(SimpleNamespace(), SimpleNamespace())
    assert out is None


def test_get_server_bridge_handles_init_exception(monkeypatch):
    class _BrokenBridge:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no bridge")

    monkeypatch.setenv("UNITARES_URL", "http://example.test/mcp/")
    _set_ctx(monkeypatch, SimpleNamespace(server_bridge=None))
    monkeypatch.setattr("anima_mcp.unitares_bridge.UnitaresBridge", _BrokenBridge)

    assert server._get_server_bridge() is None


def test_parse_shm_governance_freshness_fresh_unitares():
    now = 1_000.0
    gov_ts = now - 10.0
    gov_at = datetime.fromtimestamp(gov_ts).isoformat()
    fresh, is_unitares, parsed_ts = server._parse_shm_governance_freshness(
        {"source": "unitares", "governance_at": gov_at},
        now_ts=now,
    )
    assert fresh is True
    assert is_unitares is True
    assert parsed_ts == pytest.approx(gov_ts)


def test_parse_shm_governance_freshness_stale_or_invalid():
    now = 2_000.0
    stale_ts = now - (server.SHM_GOVERNANCE_STALE_SECONDS + 1)
    stale_at = datetime.fromtimestamp(stale_ts).isoformat()
    fresh, is_unitares, parsed_ts = server._parse_shm_governance_freshness(
        {"source": "unitares", "governance_at": stale_at},
        now_ts=now,
    )
    assert fresh is False
    assert is_unitares is True
    assert parsed_ts == pytest.approx(stale_ts)

    fresh2, is_unitares2, parsed_ts2 = server._parse_shm_governance_freshness(
        {"source": "unitares", "governance_at": "not-a-date"},
        now_ts=now,
    )
    assert (fresh2, is_unitares2, parsed_ts2) == (False, False, None)


def test_parse_shm_governance_freshness_non_unitares_source():
    now = 3_000.0
    gov_ts = now - 5.0
    gov_at = datetime.fromtimestamp(gov_ts).isoformat()
    fresh, is_unitares, parsed_ts = server._parse_shm_governance_freshness(
        {"source": "local", "governance_at": gov_at},
        now_ts=now,
    )
    assert fresh is True
    assert is_unitares is False
    assert parsed_ts == pytest.approx(gov_ts)


def test_get_schema_hub_lazy_initializes_once(monkeypatch):
    calls = {"count": 0}

    class _FakeHub:
        def __init__(self):
            calls["count"] += 1

    _set_ctx(monkeypatch, SimpleNamespace(schema_hub=None))
    monkeypatch.setattr(accessors, "SchemaHub", _FakeHub)

    first = server._get_schema_hub()
    second = server._get_schema_hub()
    assert first is second
    assert calls["count"] == 1


def test_get_display_lazy_initializes_once(monkeypatch):
    display_obj = object()
    calls = {"count": 0}

    def _factory():
        calls["count"] += 1
        return display_obj

    _set_ctx(monkeypatch, SimpleNamespace(display=None))
    monkeypatch.setattr(accessors, "get_display", _factory)

    assert server._get_display() is display_obj
    assert server._get_display() is display_obj
    assert calls["count"] == 1


def test_get_metacog_monitor_lazy_initializes_once(monkeypatch):
    calls = {"count": 0}

    class _FakeMonitor:
        def __init__(self, surprise_threshold, reflection_cooldown_seconds):
            calls["count"] += 1
            self.surprise_threshold = surprise_threshold
            self.reflection_cooldown_seconds = reflection_cooldown_seconds

    ctx = SimpleNamespace(metacog_monitor=None)
    _set_ctx(monkeypatch, ctx)
    monkeypatch.setattr("anima_mcp.metacognition.MetacognitiveMonitor", _FakeMonitor)

    first = server._get_metacog_monitor()
    second = server._get_metacog_monitor()
    assert first is second
    assert first.surprise_threshold == 0.3
    assert first.reflection_cooldown_seconds == 120.0
    assert calls["count"] == 1


def test_get_calibration_drift_loads_existing_state(monkeypatch, tmp_path):
    class _FakeDrift:
        @staticmethod
        def load(path):
            return {"loaded_from": path}

    (tmp_path / ".anima").mkdir()
    (tmp_path / ".anima" / "calibration_drift.json").write_text("{}", encoding="utf-8")

    ctx = SimpleNamespace(calibration_drift=None)
    _set_ctx(monkeypatch, ctx)
    monkeypatch.setattr(accessors, "CalibrationDrift", _FakeDrift)
    monkeypatch.setattr(accessors.Path, "home", staticmethod(lambda: tmp_path))

    out = server._get_calibration_drift()
    assert str(out["loaded_from"]).endswith(".anima/calibration_drift.json")


def test_get_calibration_drift_falls_back_on_load_failure(monkeypatch, tmp_path):
    class _FakeDrift:
        @staticmethod
        def load(path):
            raise RuntimeError("bad json")

        def __init__(self):
            self.kind = "fresh"

    (tmp_path / ".anima").mkdir()
    (tmp_path / ".anima" / "calibration_drift.json").write_text("{}", encoding="utf-8")

    ctx = SimpleNamespace(calibration_drift=None)
    _set_ctx(monkeypatch, ctx)
    monkeypatch.setattr(accessors, "CalibrationDrift", _FakeDrift)
    monkeypatch.setattr(accessors.Path, "home", staticmethod(lambda: tmp_path))

    out = server._get_calibration_drift()
    assert out.kind == "fresh"


def test_get_calibration_drift_creates_fresh_when_missing(monkeypatch, tmp_path):
    class _FakeDrift:
        def __init__(self):
            self.kind = "fresh"

    ctx = SimpleNamespace(calibration_drift=None)
    _set_ctx(monkeypatch, ctx)
    monkeypatch.setattr(accessors, "CalibrationDrift", _FakeDrift)
    monkeypatch.setattr(accessors.Path, "home", staticmethod(lambda: tmp_path))

    out = server._get_calibration_drift()
    assert out.kind == "fresh"
