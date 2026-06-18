"""Orchestration tests — end-to-end lifecycle with mocked hardware.

Verifies the wake → main loop → sleep lifecycle works correctly with
all subsystems mocked. Each test is isolated (fresh temp dir, fresh DB).
"""

import uuid
from datetime import datetime
from pathlib import Path


import anima_mcp.server as server
import anima_mcp.accessors as accessors
import anima_mcp.ctx_ref as ctx_ref


def _set_ctx(ctx):
    """Set ctx in server, ctx_ref, and accessors."""
    server._ctx = ctx
    ctx_ref.set_ctx(ctx)


def _clear_ctx():
    """Clear ctx from all modules."""
    server._ctx = None
    ctx_ref.set_ctx(None)


# ============================================================
# Test 1: wake() creates a valid context
# ============================================================

def test_wake_creates_valid_context(tmp_path, monkeypatch):
    """wake() creates a ServerContext with identity, growth, and store."""
    db_path = str(tmp_path / "test.db")

    # Prevent health registry probes from failing
    monkeypatch.setattr("anima_mcp.lifecycle._get_ctx", lambda: server._ctx)

    try:
        server.wake(db_path=db_path)
        ctx = server._ctx

        # Context should exist
        assert ctx is not None, "wake() should set _ctx"

        # Identity store should be initialized
        assert ctx.store is not None, "Identity store should be created"
        assert ctx.anima_id is not None, "anima_id should be set"
        assert len(ctx.anima_id) == 36, "anima_id should be a UUID"

        # Growth system should be initialized (or None if deps missing)
        # Don't assert it's not None since it may fail in test env

        # ctx_ref should also have the context
        assert ctx_ref._ctx is ctx, "ctx_ref should be synced"

    finally:
        if server._ctx and server._ctx.store:
            try:
                server._ctx.store.close()
            except Exception:
                pass
        _clear_ctx()


# ============================================================
# Test 2: sleep() cleans up context
# ============================================================

def test_sleep_clears_context(tmp_path, monkeypatch):
    """sleep() persists state and sets _ctx to None."""
    db_path = str(tmp_path / "test.db")

    try:
        server.wake(db_path=db_path)
        assert server._ctx is not None

        server.sleep()

        # Context should be cleared
        assert server._ctx is None, "sleep() should clear _ctx"
        assert ctx_ref._ctx is None, "sleep() should clear ctx_ref._ctx"

    finally:
        if server._ctx and server._ctx.store:
            try:
                server._ctx.store.close()
            except Exception:
                pass
        _clear_ctx()


# ============================================================
# Test 3: wake → sleep → wake preserves identity
# ============================================================

def test_wake_sleep_wake_preserves_identity(tmp_path, monkeypatch):
    """Second wake() after sleep() recovers the same creature_id."""
    db_path = str(tmp_path / "test.db")

    try:
        # First wake
        server.wake(db_path=db_path)
        first_id = server._ctx.anima_id
        assert first_id is not None

        # Sleep
        server.sleep()
        assert server._ctx is None

        # Second wake — should recover identity
        server.wake(db_path=db_path)
        second_id = server._ctx.anima_id

        assert second_id == first_id, (
            f"Identity should persist: first={first_id[:8]}, second={second_id[:8] if second_id else 'None'}"
        )

    finally:
        if server._ctx and server._ctx.store:
            try:
                server._ctx.store.close()
            except Exception:
                pass
        _clear_ctx()


# ============================================================
# Test 4: wake() handles database lock gracefully
# ============================================================

def test_wake_handles_db_lock(tmp_path, monkeypatch):
    """wake() retries on database lock and eventually gives up."""
    db_path = str(tmp_path / "test.db")

    # Mock IdentityStore to always raise lock error
    call_count = 0

    class LockingStore:
        def __init__(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("database is locked")

    monkeypatch.setattr("anima_mcp.identity.IdentityStore", LockingStore)
    monkeypatch.setattr("anima_mcp.identity.store.IdentityStore", LockingStore)
    # Speed up retries
    monkeypatch.setattr("time.sleep", lambda x: None)

    try:
        server.wake(db_path=db_path)

        # Should have retried and given up
        assert server._ctx is None, "Failed wake should clear _ctx"
        assert call_count == 5, f"Should retry 5 times, got {call_count}"

    finally:
        _clear_ctx()


# ============================================================
# Test 5: tool dispatch round-trip
# ============================================================

def test_tool_registry_complete():
    """Every tool has a handler, every handler has a tool — no drift."""
    from anima_mcp.tool_registry import TOOLS, HANDLERS

    tool_names = {t.name for t in TOOLS}
    handler_names = set(HANDLERS.keys())

    missing_handlers = tool_names - handler_names
    missing_tools = handler_names - tool_names

    assert not missing_handlers, f"Tools without handlers: {missing_handlers}"
    assert not missing_tools, f"Handlers without tools: {missing_tools}"


# ============================================================
# Test 6: wake() with explicit anima_id
# ============================================================

def test_wake_with_explicit_id(tmp_path, monkeypatch):
    """wake() respects explicitly provided anima_id."""
    db_path = str(tmp_path / "test.db")
    explicit_id = str(uuid.uuid4())

    try:
        server.wake(db_path=db_path, anima_id=explicit_id)
        assert server._ctx is not None
        assert server._ctx.anima_id == explicit_id

    finally:
        if server._ctx and server._ctx.store:
            try:
                server._ctx.store.close()
            except Exception:
                pass
        _clear_ctx()


# ============================================================
# Test 7: sleep() handles missing subsystems gracefully
# ============================================================

def test_sleep_handles_missing_subsystems():
    """sleep() doesn't crash when subsystems are None."""
    from anima_mcp.server_context import ServerContext

    ctx = ServerContext()
    # Don't set any subsystems — all None
    _set_ctx(ctx)

    try:
        # Should not raise
        server.sleep()
        assert server._ctx is None
    finally:
        _clear_ctx()


# ============================================================
# Test 8: accessor functions return None before wake
# ============================================================

def test_accessors_return_none_before_wake():
    """Key accessor functions should handle _ctx=None gracefully."""
    _clear_ctx()

    assert accessors._get_store() is None
    # _get_sensors creates a MockSensors fallback, so it's never None
    # _get_display and _get_leds may also create fallbacks
    # Test that _get_store and _get_growth are None-safe
    assert accessors._get_growth() is None


# ============================================================
# Test 9: parse_shm_governance_freshness
# ============================================================

def test_parse_shm_governance_freshness():
    """Governance freshness parsing handles various SHM states."""
    from anima_mcp.loop_phases import parse_shm_governance_freshness
    import time

    # No governance data
    fresh, unitares, ts = parse_shm_governance_freshness({})
    assert not fresh
    assert not unitares

    # Invalid governance data
    fresh, unitares, ts = parse_shm_governance_freshness("not a dict")
    assert not fresh

    # Fresh UNITARES governance
    now_iso = datetime.now().isoformat()
    fresh, unitares, ts = parse_shm_governance_freshness(
        {"governance_at": now_iso, "source": "unitares"},
        now_ts=time.time()
    )
    assert fresh
    assert unitares
    assert ts is not None

    # Stale governance (1 hour old)
    old_iso = datetime(2020, 1, 1, 0, 0, 0).isoformat()
    fresh, unitares, ts = parse_shm_governance_freshness(
        {"governance_at": old_iso, "source": "unitares"},
        now_ts=time.time()
    )
    assert not fresh


# ============================================================
# Test 10: server.py line count sanity check
# ============================================================

def test_server_decomposition_target():
    """server.py should stay below 2000 lines after decomposition."""
    server_path = Path(__file__).parent.parent / "src" / "anima_mcp" / "server.py"
    lines = len(server_path.read_text().splitlines())
    assert lines < 2000, f"server.py is {lines} lines — decomposition target is <2000"
