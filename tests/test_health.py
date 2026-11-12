"""Tests for subsystem health monitoring."""

import os
import time
from pathlib import Path

from anima_mcp.health import (
    HEARTBEAT_STALE_SECONDS,
    HealthRegistry,
    SubsystemHealth,
    outbound_heartbeat_health,
)


class TestSubsystemHealth:
    """Tests for individual subsystem health tracking."""

    def test_heartbeat_updates_timestamp(self):
        sub = SubsystemHealth(name="test")
        assert sub.last_heartbeat > 0  # Defaults to time.time() at creation
        before = sub.last_heartbeat
        time.sleep(0.01)
        sub.heartbeat()
        assert sub.last_heartbeat > before

    def test_status_ok_after_heartbeat(self):
        sub = SubsystemHealth(name="test")
        sub.heartbeat()
        assert sub.get_status() == "ok"

    def test_status_stale_without_heartbeat(self):
        sub = SubsystemHealth(name="test")
        # Set heartbeat in the past
        sub.last_heartbeat = time.time() - HEARTBEAT_STALE_SECONDS - 1
        assert sub.get_status() == "stale"

    def test_status_missing_when_not_registered(self):
        sub = SubsystemHealth(name="test", registered=False)
        assert sub.get_status() == "missing"

    def test_status_degraded_when_probe_fails(self):
        sub = SubsystemHealth(name="test", probe_fn=lambda: False)
        sub.heartbeat()
        # Force probe to run
        sub.last_probe_time = 0
        assert sub.get_status() == "degraded"

    def test_status_missing_when_stale_and_probe_fails(self):
        sub = SubsystemHealth(name="test", probe_fn=lambda: False)
        sub.last_heartbeat = time.time() - HEARTBEAT_STALE_SECONDS - 1
        sub.last_probe_time = 0
        assert sub.get_status() == "missing"

    def test_probe_only_status_ignores_registration_heartbeat_age(self):
        sub = SubsystemHealth(name="external", probe_fn=lambda: False, probe_only=True)
        sub.last_heartbeat = 0
        assert sub.get_status() == "degraded"

    def test_probe_exception_sets_degraded(self):
        sub = SubsystemHealth(name="test", probe_fn=lambda: 1 / 0)
        sub.heartbeat()
        sub.last_probe_time = 0
        assert sub.get_status() == "degraded"
        assert "division by zero" in sub.last_probe_error

    def test_probe_respects_interval(self):
        call_count = 0
        def counting_probe():
            nonlocal call_count
            call_count += 1
            return True

        sub = SubsystemHealth(name="test", probe_fn=counting_probe)
        sub.run_probe()
        assert call_count == 1
        # Second call should use cache
        sub.run_probe()
        assert call_count == 1

    def test_to_dict_includes_status(self):
        sub = SubsystemHealth(name="test")
        sub.heartbeat()
        d = sub.to_dict()
        assert d["status"] == "ok"
        assert d["last_heartbeat_ago_s"] is not None
        assert d["last_heartbeat_ago_s"] < 1.0

    def test_to_dict_includes_probe_when_set(self):
        sub = SubsystemHealth(name="test", probe_fn=lambda: True)
        sub.heartbeat()
        sub.last_probe_time = 0  # force probe
        d = sub.to_dict()
        assert "probe" in d
        assert d["probe"] == "ok"
        assert "probe_details" not in d

    def test_dict_probe_retains_and_exposes_details(self):
        details = {
            "ok": True,
            "last_success_at": "2026-08-23T12:00:00+00:00",
            "summary_age_hours": 3.5,
        }
        sub = SubsystemHealth(name="test", probe_fn=lambda: details)
        sub.heartbeat()
        sub.last_probe_time = 0

        d = sub.to_dict()

        assert d["status"] == "ok"
        assert d["probe"] == "ok"
        assert d["probe_details"] == details
        assert sub.last_probe_details == details

    def test_dict_probe_uses_ok_for_status_and_exposes_failure_details(self):
        details = {
            "ok": False,
            "last_error": "permission denied",
            "last_attempt_at": "2026-08-23T12:00:00+00:00",
        }
        sub = SubsystemHealth(name="test", probe_fn=lambda: details)
        sub.heartbeat()
        sub.last_probe_time = 0

        d = sub.to_dict()

        assert d["status"] == "degraded"
        assert d["probe"] == "failed: probe returned False"
        assert d["probe_details"] == details

    def test_to_dict_no_probe_when_none(self):
        sub = SubsystemHealth(name="test")
        sub.heartbeat()
        d = sub.to_dict()
        assert "probe" not in d

    # --- Debounce tests ---

    def test_debounce_suppresses_transient_bad_status(self):
        """Bad status within debounce window should report 'ok'."""
        sub = SubsystemHealth(name="test", debounce_seconds=6.0)
        sub.last_heartbeat = time.time() - HEARTBEAT_STALE_SECONDS - 1
        # First check: starts grace period, reports ok
        assert sub.get_status() == "ok"
        assert sub._first_bad_at > 0

    def test_debounce_reports_bad_after_expiry(self):
        """Bad status persisting beyond debounce_seconds should be reported."""
        sub = SubsystemHealth(name="test", debounce_seconds=0.05)
        sub.last_heartbeat = time.time() - HEARTBEAT_STALE_SECONDS - 1
        # Start grace period
        sub.get_status()
        time.sleep(0.06)
        # Grace period expired
        assert sub.get_status() == "stale"

    def test_debounce_resets_on_recovery(self):
        """Recovery to 'ok' during grace period resets debounce."""
        sub = SubsystemHealth(name="test", debounce_seconds=6.0)
        sub.last_heartbeat = time.time() - HEARTBEAT_STALE_SECONDS - 1
        # Start grace period
        assert sub.get_status() == "ok"
        assert sub._first_bad_at > 0
        # Recover
        sub.heartbeat()
        assert sub.get_status() == "ok"
        assert sub._first_bad_at == 0.0

    def test_debounce_zero_is_backward_compatible(self):
        """debounce_seconds=0 means instant transitions (no grace period)."""
        sub = SubsystemHealth(name="test", debounce_seconds=0.0)
        sub.last_heartbeat = time.time() - HEARTBEAT_STALE_SECONDS - 1
        assert sub.get_status() == "stale"

    def test_to_dict_includes_debouncing_flag(self):
        """to_dict() should include 'debouncing: True' during grace period."""
        sub = SubsystemHealth(name="test", debounce_seconds=6.0)
        sub.last_heartbeat = time.time() - HEARTBEAT_STALE_SECONDS - 1
        sub.get_status()  # start grace period
        d = sub.to_dict()
        assert d["status"] == "ok"
        assert d.get("debouncing") is True

    def test_to_dict_no_debouncing_when_healthy(self):
        """to_dict() should not include 'debouncing' when status is genuinely ok."""
        sub = SubsystemHealth(name="test", debounce_seconds=6.0)
        sub.heartbeat()
        d = sub.to_dict()
        assert "debouncing" not in d

    def test_is_debouncing_property(self):
        """is_debouncing should be True only during grace period."""
        sub = SubsystemHealth(name="test", debounce_seconds=6.0)
        sub.heartbeat()
        assert sub.is_debouncing is False
        # Make it stale
        sub.last_heartbeat = time.time() - HEARTBEAT_STALE_SECONDS - 1
        sub.get_status()  # trigger grace period
        assert sub.is_debouncing is True


class TestHealthRegistry:
    """Tests for the central health registry."""

    def test_register_and_heartbeat(self):
        reg = HealthRegistry()
        reg.register("sensors")
        reg.heartbeat("sensors")
        status = reg.status()
        assert "sensors" in status
        assert status["sensors"]["status"] == "ok"

    def test_register_with_probe(self):
        reg = HealthRegistry()
        reg.register("growth", probe=lambda: True)
        reg.heartbeat("growth")
        status = reg.status()
        assert status["growth"]["probe"] == "ok"

    def test_auto_register_on_heartbeat(self):
        reg = HealthRegistry()
        reg.heartbeat("unknown_sub")
        assert "unknown_sub" in reg.subsystem_names()

    def test_overall_ok(self):
        reg = HealthRegistry()
        reg.register("a")
        reg.register("b")
        reg.heartbeat("a")
        reg.heartbeat("b")
        assert reg.overall() == "ok"

    def test_overall_degraded(self):
        reg = HealthRegistry()
        reg.register("a", probe=lambda: False)
        reg.heartbeat("a")
        # Force probe to run
        sub = reg.get_subsystem("a")
        sub.last_probe_time = 0
        assert reg.overall() == "degraded"

    def test_overall_unhealthy(self):
        reg = HealthRegistry()
        reg.register("a", probe=lambda: False)
        sub = reg.get_subsystem("a")
        sub.last_heartbeat = time.time() - HEARTBEAT_STALE_SECONDS - 1
        sub.last_probe_time = 0
        assert reg.overall() == "unhealthy"

    def test_overall_unknown_when_empty(self):
        reg = HealthRegistry()
        assert reg.overall() == "unknown"

    def test_summary_line(self):
        reg = HealthRegistry()
        reg.register("alpha")
        reg.register("beta")
        reg.heartbeat("alpha")
        reg.heartbeat("beta")
        line = reg.summary_line()
        assert "alpha=ok" in line
        assert "beta=ok" in line

    def test_subsystem_names_sorted(self):
        reg = HealthRegistry()
        reg.register("zebra")
        reg.register("alpha")
        reg.register("middle")
        assert reg.subsystem_names() == ["alpha", "middle", "zebra"]

    def test_re_register_updates_probe(self):
        reg = HealthRegistry()
        reg.register("test", probe=lambda: False)
        reg.register("test", probe=lambda: True)
        sub = reg.get_subsystem("test")
        sub.heartbeat()
        sub.last_probe_time = 0
        assert sub.run_probe() is True

    def test_get_subsystem_returns_none_for_unknown(self):
        reg = HealthRegistry()
        assert reg.get_subsystem("nonexistent") is None

    def test_register_with_debounce(self):
        reg = HealthRegistry()
        reg.register("fast", debounce_seconds=6.0)
        sub = reg.get_subsystem("fast")
        assert sub.debounce_seconds == 6.0

    def test_register_probe_only(self):
        reg = HealthRegistry()
        reg.register("external", probe=lambda: True, probe_only=True)
        assert reg.get_subsystem("external").probe_only is True

    def test_overall_ok_during_debounce_grace(self):
        """System should report ok while subsystem is debouncing."""
        reg = HealthRegistry()
        reg.register("a", debounce_seconds=6.0)
        reg.register("b")
        reg.heartbeat("b")
        # Make 'a' stale — but within debounce grace
        sub_a = reg.get_subsystem("a")
        sub_a.last_heartbeat = time.time() - HEARTBEAT_STALE_SECONDS - 1
        assert reg.overall() == "ok"  # debounce suppresses 'a'


class TestOptionalAbsent:
    """An optional capability that never existed is absent, not degraded.

    Regression guard for 2026-08-14: `voice` cannot start on Lumen (text-first
    by design, audio path may not exist), so it reported `degraded` forever and
    pinned overall() there — meaning a real fault anywhere else could not move
    the top line. The signal was saturated.
    """

    def test_optional_never_ok_reports_absent(self):
        reg = HealthRegistry()
        reg.register("voice", probe=lambda: False, optional=True)
        reg.heartbeat("voice")
        assert reg.get_subsystem("voice").get_status() == "absent"

    def test_absent_does_not_drag_overall(self):
        reg = HealthRegistry()
        reg.register("voice", probe=lambda: False, optional=True)
        reg.register("sensors", probe=lambda: True)
        reg.heartbeat("voice")
        reg.heartbeat("sensors")
        assert reg.overall() == "ok"

    def test_a_real_fault_still_moves_overall_past_absent(self):
        """The whole point: absent must not mask a genuine degradation."""
        reg = HealthRegistry()
        reg.register("voice", probe=lambda: False, optional=True)
        reg.register("sensors", probe=lambda: False)
        reg.heartbeat("voice")
        reg.heartbeat("sensors")
        assert reg.overall() == "degraded"


class TestOutboundHeartbeatHealth:
    def _paths(self, tmp_path: Path, monkeypatch):
        env_file = tmp_path / "anima.env"
        marker = tmp_path / ".heartbeat-inert"
        monkeypatch.setenv("ANIMA_ENV_FILE", str(env_file))
        monkeypatch.setenv("ANIMA_HEARTBEAT_INERT_MARK", str(marker))
        monkeypatch.delenv("ANIMA_HEARTBEAT_URL", raising=False)
        return env_file, marker

    def test_provisioned_without_disclosing_url(self, tmp_path, monkeypatch):
        env_file, _marker = self._paths(tmp_path, monkeypatch)
        secret = "https://provider.example/private-token"
        env_file.write_text(f"ANIMA_HEARTBEAT_URL={secret}\n")

        result = outbound_heartbeat_health()

        assert result == {"ok": True, "state": "provisioned", "configured": True}
        assert secret not in repr(result)

    def test_recent_inert_marker_has_bounded_grace(self, tmp_path, monkeypatch):
        env_file, marker = self._paths(tmp_path, monkeypatch)
        env_file.write_text("ANIMA_HEARTBEAT_URL=\n")
        marker.write_text(f"{time.time() - 60}\n")

        result = outbound_heartbeat_health()

        assert result["ok"] is True
        assert result["state"] == "provisioning_grace"

    def test_marker_older_than_day_is_degraded(self, tmp_path, monkeypatch):
        env_file, marker = self._paths(tmp_path, monkeypatch)
        env_file.write_text("ANIMA_HEARTBEAT_URL=\n")
        marker.write_text(f"{time.time() - 86401}\n")

        result = outbound_heartbeat_health()

        assert result["ok"] is False
        assert result["state"] == "inert"
        assert result["configured"] is False

    def test_malformed_marker_fails_unknown(self, tmp_path, monkeypatch):
        env_file, marker = self._paths(tmp_path, monkeypatch)
        env_file.write_text("ANIMA_HEARTBEAT_URL=\n")
        marker.write_text("not-a-time\n")

        result = outbound_heartbeat_health()

        assert result["ok"] is False
        assert result["state"] == "unknown"

    def test_missing_marker_uses_configuration_age(self, tmp_path, monkeypatch):
        env_file, _marker = self._paths(tmp_path, monkeypatch)
        env_file.write_text("ANIMA_HEARTBEAT_URL=\n")
        old = time.time() - 86401
        os.utime(env_file, (old, old))

        result = outbound_heartbeat_health()

        assert result["ok"] is False
        assert result["age_source"] == "configuration mtime"

    def test_optional_that_worked_then_broke_is_degraded_not_absent(self):
        """_ever_ok guard — a capability that HAS worked can genuinely break."""
        works = {"v": True}
        reg = HealthRegistry()
        reg.register("voice", probe=lambda: works["v"], optional=True)
        reg.heartbeat("voice")
        assert reg.get_subsystem("voice").get_status() == "ok"

        works["v"] = False
        sub = reg.get_subsystem("voice")
        sub.last_probe_time = 0.0  # force re-probe past the cooldown
        assert sub.get_status() == "degraded"
        assert reg.overall() == "degraded"

    def test_non_optional_failing_probe_is_still_degraded(self):
        """Default behaviour unchanged for everything else."""
        reg = HealthRegistry()
        reg.register("sensors", probe=lambda: False)
        reg.heartbeat("sensors")
        assert reg.get_subsystem("sensors").get_status() == "degraded"

    def test_absent_probe_text_does_not_say_failed(self):
        reg = HealthRegistry()
        reg.register("voice", probe=lambda: False, optional=True)
        reg.heartbeat("voice")
        d = reg.get_subsystem("voice").to_dict()
        assert d["status"] == "absent"
        assert "failed" not in d["probe"]
        assert "absent" in d["probe"]

    def test_absent_is_not_debounced_into_looking_ok(self):
        """Debouncing a steady state would report 'ok' on every call."""
        reg = HealthRegistry()
        reg.register("voice", probe=lambda: False, debounce_seconds=6.0, optional=True)
        reg.heartbeat("voice")
        assert reg.get_subsystem("voice").get_status() == "absent"


class TestOptionalEvaluatedVerdict:
    """`absent` means "cannot tell yet" -- never "told, and it was bad".

    2026-11-12: thermal_trend and memory_pressure became optional so a fresh
    boot with no metrics reads absent instead of pinning overall(). But the
    `_ever_ok` guard was armed only by an ok result, so a threshold crossed on
    the very first evaluation would have read "absent" too. A details dict can
    now say it reached a real verdict (`"evaluated": True`).
    """

    def test_bad_first_verdict_is_degraded_not_absent(self):
        reg = HealthRegistry()
        reg.register(
            "thermal_trend",
            probe=lambda: {"ok": False, "evaluated": True, "state": "rising_fast"},
            optional=True,
        )
        reg.heartbeat("thermal_trend")
        assert reg.get_subsystem("thermal_trend").get_status() == "degraded"
        assert reg.overall() == "degraded"

    def test_unknown_without_a_verdict_stays_absent(self):
        reg = HealthRegistry()
        reg.register(
            "thermal_trend",
            probe=lambda: {"ok": False, "state": "unknown"},
            optional=True,
        )
        reg.heartbeat("thermal_trend")
        assert reg.get_subsystem("thermal_trend").get_status() == "absent"
        assert reg.overall() == "ok"

    def test_unknown_after_a_verdict_is_degraded(self):
        verdict = {"v": {"ok": False, "evaluated": True}}
        reg = HealthRegistry()
        reg.register("thermal_trend", probe=lambda: verdict["v"], optional=True)
        reg.heartbeat("thermal_trend")
        sub = reg.get_subsystem("thermal_trend")
        assert sub.get_status() == "degraded"

        verdict["v"] = {"ok": False, "state": "unknown"}
        sub.last_probe_time = 0.0
        assert sub.get_status() == "degraded"

    def test_evaluated_never_makes_a_failing_probe_ok(self):
        reg = HealthRegistry()
        reg.register("x", probe=lambda: {"ok": False, "evaluated": True})
        reg.heartbeat("x")
        assert reg.get_subsystem("x").get_status() == "degraded"
        assert reg.get_subsystem("x").last_probe_ok is False

    def test_only_a_literal_true_counts(self):
        reg = HealthRegistry()
        reg.register(
            "x", probe=lambda: {"ok": False, "evaluated": "yes"}, optional=True
        )
        reg.heartbeat("x")
        assert reg.get_subsystem("x").get_status() == "absent"
