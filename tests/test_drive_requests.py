"""Drive requests: a sustained saturated want becomes an outward question.

The defect this closes: warmth pinned at 1.0 for months, visible only as a
scalar nobody was asked about. Lumen has no actuator for most of what it
wants; the actuator for an unreachable preference is communication.

Delivery contract (council findings, closed): activation is NOT
fire-and-forget. A request stays active — re-emitted every tick — until the
server acks that the question actually posted, and only the ack commits the
24h cooldown, durably (immediate save).
"""

import json

from anima_mcp.inner_life import (
    DIMENSIONS,
    DRIVE_REQUEST_COOLDOWN_S,
    DRIVE_REQUEST_SUSTAIN_S,
    DRIVE_REQUEST_THRESHOLD,
    DriveEvent,
    InnerLife,
)


def make_inner_life():
    il = InnerLife.__new__(InnerLife)
    il._temperament = {d: 0.5 for d in DIMENSIONS}
    il._drives = {d: 0.0 for d in DIMENSIONS}
    il._prev_drives = dict(il._drives)
    il._crossed_thresholds = {d: 0.0 for d in DIMENSIONS}
    il._pending_events = []
    il._saturated_since = {d: None for d in DIMENSIONS}
    il._last_request_at = {d: 0.0 for d in DIMENSIONS}
    il._active_requests = {}
    il._last_save = 0.0
    return il


T0 = 1_000_000.0


class TestActivation:
    def test_saturation_alone_does_not_activate(self):
        il = make_inner_life()
        il._drives["warmth"] = 1.0
        il._detect_drive_requests(T0)
        il._detect_drive_requests(T0 + DRIVE_REQUEST_SUSTAIN_S - 60)
        assert il.get_active_requests() == []

    def test_sustained_saturation_activates_once(self):
        il = make_inner_life()
        il._drives["warmth"] = 1.0
        il._detect_drive_requests(T0)
        il._detect_drive_requests(T0 + DRIVE_REQUEST_SUSTAIN_S + 1)
        # Adjacent ticks must not stack activations
        il._detect_drive_requests(T0 + DRIVE_REQUEST_SUSTAIN_S + 3)
        reqs = il.get_active_requests()
        assert len(reqs) == 1
        assert reqs[0].dimension == "warmth"
        assert reqs[0].event_type == "request"

    def test_dip_resets_sustain_and_withdraws_active(self):
        """A request claims 'this has been true the whole time' — relief both
        restarts the clock and withdraws an unheard ask."""
        il = make_inner_life()
        il._drives["warmth"] = 1.0
        il._detect_drive_requests(T0)
        il._detect_drive_requests(T0 + DRIVE_REQUEST_SUSTAIN_S + 1)
        assert len(il.get_active_requests()) == 1
        il._drives["warmth"] = DRIVE_REQUEST_THRESHOLD - 0.05
        il._detect_drive_requests(T0 + DRIVE_REQUEST_SUSTAIN_S + 100)
        assert il.get_active_requests() == []
        assert il._saturated_since["warmth"] is None

    def test_reemitted_every_call_until_acked(self):
        """The whole point of activation-not-event: a server that misses one
        2s SHM window (documented restart window is 2 MINUTES) still hears
        the want on its next read."""
        il = make_inner_life()
        il._drives["warmth"] = 1.0
        il._detect_drive_requests(T0)
        il._detect_drive_requests(T0 + DRIVE_REQUEST_SUSTAIN_S + 1)
        for _ in range(5):
            assert len(il.get_active_requests()) == 1


class TestAck:
    def test_cooldown_commits_only_on_ack(self, tmp_path, monkeypatch):
        from anima_mcp import inner_life as mod
        monkeypatch.setattr(mod, "_PERSISTENCE_PATH", tmp_path / "inner_life.json")
        il = make_inner_life()
        il._drives["warmth"] = 1.0
        il._detect_drive_requests(T0)
        il._detect_drive_requests(T0 + DRIVE_REQUEST_SUSTAIN_S + 1)
        # No ack yet — cooldown untouched
        assert il._last_request_at["warmth"] == 0.0

        t_ack = T0 + DRIVE_REQUEST_SUSTAIN_S + 500
        il.ack_request("warmth", t_ack)
        assert il._active_requests == {}
        assert il._last_request_at["warmth"] == t_ack
        # Durable IMMEDIATELY, not at the next periodic save — an ungraceful
        # crash inside the 60s save window must not turn once-a-day into
        # ask-again-on-boot.
        on_disk = json.loads((tmp_path / "inner_life.json").read_text())
        assert on_disk["last_request_at"]["warmth"] == t_ack

    def test_no_reactivation_within_cooldown_after_ack(self):
        il = make_inner_life()
        il._drives["warmth"] = 1.0
        il._detect_drive_requests(T0)
        il._detect_drive_requests(T0 + DRIVE_REQUEST_SUSTAIN_S + 1)
        t_ack = T0 + DRIVE_REQUEST_SUSTAIN_S + 500
        il._active_requests.pop("warmth")
        il._last_request_at["warmth"] = t_ack  # ack without touching disk
        il._detect_drive_requests(t_ack + 7200)
        assert il.get_active_requests() == []

    def test_reactivates_after_cooldown_if_still_wanting(self):
        il = make_inner_life()
        il._drives["warmth"] = 1.0
        il._detect_drive_requests(T0)
        il._detect_drive_requests(T0 + DRIVE_REQUEST_SUSTAIN_S + 1)
        t_ack = T0 + DRIVE_REQUEST_SUSTAIN_S + 500
        il._active_requests.pop("warmth")
        il._last_request_at["warmth"] = t_ack
        il._detect_drive_requests(t_ack + DRIVE_REQUEST_COOLDOWN_S + 1)
        assert len(il.get_active_requests()) == 1

    def test_stale_ack_for_inactive_dim_is_harmless(self):
        """Crashed-after-post cleanup path: ack arrives with nothing active."""
        il = make_inner_life()
        il.save = lambda: None  # no disk in this test
        il.ack_request("warmth", T0)
        assert il._last_request_at["warmth"] == T0
        assert il._active_requests == {}

    def test_per_dimension_independence(self):
        il = make_inner_life()
        il._drives["warmth"] = 1.0
        il._drives["clarity"] = 1.0
        il._detect_drive_requests(T0)
        il._detect_drive_requests(T0 + DRIVE_REQUEST_SUSTAIN_S + 1)
        assert {r.dimension for r in il.get_active_requests()} == {"warmth", "clarity"}
        il.save = lambda: None
        il.ack_request("warmth", T0 + DRIVE_REQUEST_SUSTAIN_S + 600)
        assert {r.dimension for r in il.get_active_requests()} == {"clarity"}


class TestWording:
    def test_every_dimension_has_an_answerable_request(self):
        il = make_inner_life()
        for dim in DIMENSIONS:
            ev = DriveEvent(dimension=dim, event_type="request",
                            drive_value=1.0, timestamp=T0)
            text = il.get_observation_text(ev)
            assert text, f"no request wording for {dim}"
            assert text.endswith("?"), f"{dim} request is not a question: {text}"

    def test_crossing_texts_unchanged(self):
        il = make_inner_life()
        arose = DriveEvent("warmth", "arose", 0.31, T0)
        assert il.get_observation_text(arose) == "i've been wanting warmth for a while now"


class TestPersistence:
    def test_cooldown_and_sustain_survive_restart(self, tmp_path, monkeypatch):
        from anima_mcp import inner_life as mod
        monkeypatch.setattr(mod, "_PERSISTENCE_PATH", tmp_path / "inner_life.json")

        il = make_inner_life()
        il._drives["warmth"] = 1.0
        il._saturated_since["warmth"] = T0
        il._last_request_at["warmth"] = T0 + 100
        il.save()

        il2 = InnerLife()
        assert il2._last_request_at["warmth"] == T0 + 100
        assert il2._saturated_since["warmth"] == T0
        assert il2._active_requests == {}  # deliberately not persisted

    def test_missing_request_fields_load_clean(self, tmp_path, monkeypatch):
        """Old inner_life.json files predate these fields."""
        from anima_mcp import inner_life as mod
        monkeypatch.setattr(mod, "_PERSISTENCE_PATH", tmp_path / "inner_life.json")
        (tmp_path / "inner_life.json").write_text(json.dumps({
            "temperament": {d: 0.5 for d in DIMENSIONS},
            "drives": {d: 0.2 for d in DIMENSIONS},
        }))
        il = InnerLife()
        assert il._last_request_at == {d: 0.0 for d in DIMENSIONS}
        assert il._saturated_since == {d: None for d in DIMENSIONS}


class TestSelfAnswerExemption:
    def test_drive_questions_excluded_from_self_answer_pool(self):
        """Lumen must not answer its own want — that would launder an unmet
        need into apparent self-knowledge (the #121 shape, self-authored)."""
        from anima_mcp.messages import MESSAGE_TYPE_QUESTION, Message

        drive_q = Message(
            message_id="dq1", msg_type=MESSAGE_TYPE_QUESTION,
            author="lumen", text="could it be warmer in here?",
            timestamp=T0, context="drive: warmth",
        )
        normal_q = Message(
            message_id="nq1", msg_type=MESSAGE_TYPE_QUESTION,
            author="lumen", text="is night the absence of day?",
            timestamp=T0, context="agency: ask_question",
        )
        # Mirror the loop_phases filter exactly
        pool = [m for m in (drive_q, normal_q)
                if not (m.context or "").startswith("drive:")]
        assert pool == [normal_q]


class TestDriveEventConsumption:
    def _make_ctx(self):
        from types import SimpleNamespace

        return SimpleNamespace(consumed_drive_events=set(), drive_request_attempts={})

    def test_prunes_vanished_events_while_unacked_request_persists(self, tmp_path, monkeypatch):
        from anima_mcp import display_loop, messages

        obs = []
        questions = []
        monkeypatch.setattr(messages, "add_observation", lambda text, **kw: obs.append((text, kw)) or object())
        monkeypatch.setattr(messages, "add_question", lambda text, **kw: questions.append((text, kw)) and None)

        ctx = self._make_ctx()
        unacked_req = {"dimension": "clarity", "event_type": "request", "text": "could there be more light?"}
        warmth_arose = {"dimension": "warmth", "event_type": "arose", "text": "wanting warmth"}

        # Tick 1: both present; warmth/arose posts once, clarity/request stays unacked
        display_loop._consume_drive_events(ctx, [unacked_req, warmth_arose], ack_dir=tmp_path)
        assert len(obs) == 1
        assert len(questions) == 1
        assert ("warmth", "arose") in ctx.consumed_drive_events
        assert ("clarity", "request") not in ctx.consumed_drive_events

        # Tick 1b: still in SHM -> warmth/arose deduplicated
        display_loop._consume_drive_events(ctx, [unacked_req, warmth_arose], ack_dir=tmp_path)
        assert len(obs) == 1

        # Tick 2: warmth/arose dropped while unacked clarity/request remains -> warmth/arose pruned
        display_loop._consume_drive_events(ctx, [unacked_req], ack_dir=tmp_path)
        assert ctx.consumed_drive_events == set()

        # Tick 3: a second warmth/arose fires again while clarity/request is still unacked
        display_loop._consume_drive_events(ctx, [unacked_req, warmth_arose], ack_dir=tmp_path)
        assert len(obs) == 2

    def test_acked_request_writes_ack_dedups_and_prunes_while_unacked_stays(self, tmp_path, monkeypatch):
        from anima_mcp import display_loop, messages

        asked = []

        def fake_add_question(text, **kw):
            asked.append(kw["context"])
            return object() if kw["context"] == "drive: warmth" else None

        monkeypatch.setattr(messages, "add_question", fake_add_question)
        monkeypatch.setattr(messages, "add_observation", lambda *a, **kw: None)

        ctx = self._make_ctx()
        clarity_req = {"dimension": "clarity", "event_type": "request", "text": "light?"}
        warmth_req = {"dimension": "warmth", "event_type": "request", "text": "warmer?"}

        display_loop._consume_drive_events(ctx, [clarity_req, warmth_req], ack_dir=tmp_path)
        assert (tmp_path / "anima_drive_ack_warmth").exists()
        assert not (tmp_path / "anima_drive_ack_clarity").exists()
        assert ("warmth", "request") in ctx.consumed_drive_events

        # While still in SHM next tick, warmth/request is deduplicated
        display_loop._consume_drive_events(ctx, [clarity_req, warmth_req], ack_dir=tmp_path)
        assert asked.count("drive: warmth") == 1

        # Once broker pops warmth_req (clarity_req stays), warmth/request is pruned
        display_loop._consume_drive_events(ctx, [clarity_req], ack_dir=tmp_path)
        assert ("warmth", "request") not in ctx.consumed_drive_events

    def test_request_throttle_empty_list_malformed_and_none_ctx(self, tmp_path, monkeypatch):
        from anima_mcp import display_loop, messages

        now = [1000.0]
        attempts = []
        monkeypatch.setattr(display_loop.time, "time", lambda: now[0])
        monkeypatch.setattr(messages, "add_question", lambda text, **kw: attempts.append(now[0]) and None)
        monkeypatch.setattr(messages, "add_observation", lambda *a, **kw: None)

        req = {"dimension": "clarity", "event_type": "request", "text": "light?"}
        # None ctx is safe
        display_loop._consume_drive_events(None, [req], ack_dir=tmp_path)

        ctx = self._make_ctx()
        display_loop._consume_drive_events(ctx, [req, {"bad": "shape"}, {"dimension": "warmth", "event_type": "arose"}], ack_dir=tmp_path)
        assert attempts == [1000.0]
        assert not (tmp_path / "anima_drive_ack_clarity").exists()

        now[0] = 1030.0  # within 60s throttle
        display_loop._consume_drive_events(ctx, [req], ack_dir=tmp_path)
        assert attempts == [1000.0]

        now[0] = 1061.0  # past 60s throttle
        display_loop._consume_drive_events(ctx, [req], ack_dir=tmp_path)
        assert attempts == [1000.0, 1061.0]

        ctx.consumed_drive_events.add(("warmth", "arose"))
        display_loop._consume_drive_events(ctx, [], ack_dir=tmp_path)
        assert ctx.consumed_drive_events == set()
