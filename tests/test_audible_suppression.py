"""Tests for making swallowed instrumentation failures audible.

`except Exception: pass` around optional instrumentation is the mechanism behind
Lumen's standing failure mode: a channel that stops working becomes a channel
that is silently ABSENT, indistinguishable from one that had nothing to report.

Every defect found on 2026-08-02 had that shape — the barometer publishing a
stale constant, interaction_level pinned at 0.0, the density grid reporting no
structure at all. One was found only because a swallowed ImportError made a
test fail; nothing logged, nothing degraded, the value simply was not there.
"""

import pytest

from anima_mcp.error_recovery import (
    note_suppressed,
    reset_suppressed_counts,
    suppressed_counts,
)


@pytest.fixture(autouse=True)
def clean_counters():
    reset_suppressed_counts()
    yield
    reset_suppressed_counts()


class TestCounting:
    def test_healthy_case_is_an_empty_dict(self):
        assert suppressed_counts() == {}

    def test_a_swallowed_failure_is_counted(self):
        note_suppressed("site.a", ValueError("boom"))
        assert suppressed_counts() == {"site.a": 1}

    def test_every_occurrence_counts_even_when_the_log_is_throttled(self):
        for _ in range(25):
            note_suppressed("site.a", ValueError("boom"))
        # Throttling must silence the log, never the count — a rising count is
        # the whole signal.
        assert suppressed_counts()["site.a"] == 25

    def test_sites_are_counted_separately(self):
        note_suppressed("site.a", ValueError("a"))
        note_suppressed("site.b", KeyError("b"))
        note_suppressed("site.b", KeyError("b"))
        assert suppressed_counts() == {"site.a": 1, "site.b": 2}

    def test_counts_are_a_copy_not_the_live_dict(self):
        note_suppressed("site.a", ValueError("a"))
        snapshot = suppressed_counts()
        snapshot["site.a"] = 999
        assert suppressed_counts()["site.a"] == 1


class TestLoggingBehaviour:
    def test_first_occurrence_logs(self, capsys):
        note_suppressed("site.a", ValueError("boom"))
        err = capsys.readouterr().err
        assert "site.a" in err
        assert "ValueError" in err
        assert "boom" in err

    def test_repeats_are_throttled(self, capsys):
        for _ in range(10):
            note_suppressed("site.a", ValueError("boom"))
        assert capsys.readouterr().err.count("site.a") == 1

    def test_throttle_expiry_lets_it_speak_again(self, capsys):
        note_suppressed("site.a", ValueError("boom"), throttle_seconds=0.0)
        note_suppressed("site.a", ValueError("boom"), throttle_seconds=0.0)
        assert capsys.readouterr().err.count("site.a") == 2

    def test_the_running_total_is_reported(self, capsys):
        for _ in range(4):
            note_suppressed("site.a", ValueError("boom"), throttle_seconds=0.0)
        assert "x4" in capsys.readouterr().err

    def test_each_site_gets_its_own_throttle(self, capsys):
        note_suppressed("site.a", ValueError("a"))
        note_suppressed("site.b", ValueError("b"))
        err = capsys.readouterr().err
        assert "site.a" in err and "site.b" in err


class TestItNeverBreaksTheCaller:
    """A logging helper must not become the thing that breaks a reading."""

    def test_returns_none_and_does_not_raise(self):
        assert note_suppressed("site.a", ValueError("boom")) is None

    def test_survives_an_exception_with_a_hostile_repr(self):
        class Nasty(Exception):
            def __str__(self):
                raise RuntimeError("cannot stringify")

        note_suppressed("site.a", Nasty())  # must not propagate

    def test_survives_a_non_string_site(self):
        note_suppressed(None, ValueError("boom"))  # type: ignore[arg-type]

    def test_survives_a_base_exception(self):
        note_suppressed("site.a", KeyboardInterrupt())


class TestWiredIntoTheStateRecordingPath:
    """The sites that matter: a gap in state_history is a gap in Lumen's record."""

    def test_state_history_uses_it(self):
        """The loop's single writer (state_history.py) replaced the two read
        handlers' copies of this enrichment; the gaps stay audible there."""
        import anima_mcp.state_history as sh

        src = __import__("inspect").getsource(sh)
        assert "note_suppressed(\"state_history.interaction_level\"" in src
        assert "note_suppressed(\"state_history.led_brightness\"" in src

    def test_the_loop_write_itself_is_audible(self):
        import anima_mcp.state_history as sh

        src = __import__("inspect").getsource(sh.maybe_record_state_history)
        assert "note_suppressed(\"server.state_history\", e)" in src

    def test_no_bare_pass_remains_in_the_converted_blocks(self):
        """Guards the regression: these were `except Exception: pass`."""
        import inspect

        import anima_mcp.state_history as sh

        src = inspect.getsource(sh)
        for marker in ("interaction_level", "led_brightness"):
            idx = src.find(f'sensors["{marker}"]')
            assert idx != -1, f"{marker} block not found in {sh.__name__}"
            after = src[idx: idx + 400]
            assert "except Exception:\n            pass" not in after
            assert "except Exception:\n                pass" not in after


class TestDiagnosticsSurface:
    def test_counts_are_exposed_for_diagnostics(self):
        import inspect

        import anima_mcp.handlers.display_ops as dops

        src = inspect.getsource(dops)
        assert "suppressed_counts" in src
        assert '"suppressed"' in src


class TestTheCounterSurvivesLoggingFailure:
    """The tally must outlive the message. A swallow-detector that can itself
    swallow is the exact defect this module exists to remove."""

    def test_a_hostile_str_still_increments_the_count(self):
        class Nasty(Exception):
            def __str__(self):
                raise RuntimeError("cannot stringify")

        note_suppressed("site.hostile", Nasty())
        assert suppressed_counts()["site.hostile"] == 1

    def test_a_hostile_str_degrades_to_the_type_name(self, capsys):
        class Nasty(Exception):
            def __str__(self):
                raise RuntimeError("cannot stringify")

        note_suppressed("site.hostile", Nasty())
        err = capsys.readouterr().err
        assert "site.hostile" in err, "the line was lost entirely"
        assert "Nasty" in err

    def test_a_broken_stderr_still_leaves_the_count(self, monkeypatch):
        import builtins

        def exploding_print(*a, **k):
            raise OSError("stderr is closed")

        monkeypatch.setattr(builtins, "print", exploding_print)
        note_suppressed("site.nostderr", ValueError("boom"))
        monkeypatch.undo()
        # Logging failed; the signal survives where a diagnostics call can find it.
        assert suppressed_counts()["site.nostderr"] == 1

    def test_an_unprintable_site_key_is_still_counted(self):
        class NoRepr:
            def __str__(self):
                raise RuntimeError("nope")

        note_suppressed(NoRepr(), ValueError("boom"))  # type: ignore[arg-type]
        assert suppressed_counts().get("<unprintable-site>") == 1
