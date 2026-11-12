"""
Tests for db_paths — store-path resolution.

Regression cover for #123: a bare ``db_path="anima.db"`` default used to bind
persistence to the process's working directory, which is how the broker's
agency learning ended up in ~/anima-mcp/anima.db while every backup covered
~/.anima only. The invariant these tests hold down is narrow and absolute:
resolution never yields a relative path.
"""

from pathlib import Path

import pytest

from anima_mcp.db_paths import (
    BROKER_AGENCY_DB,
    DEFAULT_DB_NAME,
    resolve_db_path,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Resolution must be decided by the test, not the developer's shell."""
    monkeypatch.delenv("ANIMA_DB", raising=False)


class TestResolution:
    """Order is explicit > $ANIMA_DB > ~/.anima/anima.db."""

    def test_env_used_when_caller_says_nothing(self, monkeypatch, tmp_path):
        target = tmp_path / "from_env.db"
        monkeypatch.setenv("ANIMA_DB", str(target))
        assert resolve_db_path() == str(target)

    def test_bare_default_is_treated_as_saying_nothing(self, monkeypatch, tmp_path):
        """The historical default is not a real relative path."""
        target = tmp_path / "from_env.db"
        monkeypatch.setenv("ANIMA_DB", str(target))
        assert resolve_db_path(DEFAULT_DB_NAME) == str(target)

    def test_explicit_beats_env(self, monkeypatch, tmp_path):
        """This is what lets the broker pin its own agency store (#123)."""
        monkeypatch.setenv("ANIMA_DB", str(tmp_path / "from_env.db"))
        explicit = str(tmp_path / "pinned.db")
        assert resolve_db_path(explicit) == explicit

    def test_falls_back_to_home_store(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert resolve_db_path() == str(tmp_path / ".anima" / DEFAULT_DB_NAME)

    def test_home_store_directory_is_created(self, monkeypatch, tmp_path):
        """A fresh install must not fail to connect for want of a directory."""
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        resolve_db_path()
        assert (tmp_path / ".anima").is_dir()


class TestNeverRelative:
    """The one guarantee. A relative result is the #123 bug returning."""

    @pytest.mark.parametrize("supplied", [None, "", DEFAULT_DB_NAME])
    def test_absent_or_default_never_resolves_relative(
        self, monkeypatch, tmp_path, supplied
    ):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert Path(resolve_db_path(supplied)).is_absolute()

    def test_result_does_not_depend_on_cwd(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        first = resolve_db_path()
        monkeypatch.chdir(tmp_path)
        assert resolve_db_path() == first


class TestBrokerPin:
    """The broker keeps its own agency table — deliberately, not by accident."""

    def test_broker_store_is_absolute(self):
        assert BROKER_AGENCY_DB.is_absolute()

    def test_broker_store_is_not_the_home_store(self, monkeypatch, tmp_path):
        """If these ever converge, the two TD learners have silently merged."""
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert str(BROKER_AGENCY_DB) != resolve_db_path()

    def test_env_does_not_capture_the_broker_store(self, monkeypatch, tmp_path):
        """$ANIMA_DB points both units at ~/.anima; the broker must not follow."""
        monkeypatch.setenv("ANIMA_DB", str(tmp_path / "main.db"))
        assert resolve_db_path(str(BROKER_AGENCY_DB)) == str(BROKER_AGENCY_DB)


class TestActionSelectorBinding:
    """#123's instance: the selector that started it."""

    def test_bare_selector_does_not_bind_to_cwd(self, monkeypatch, tmp_path):
        from anima_mcp import agency

        monkeypatch.setattr(agency, "_action_selector", None)
        monkeypatch.setenv("ANIMA_DB", str(tmp_path / "resolved.db"))
        monkeypatch.chdir(tmp_path)

        selector = agency.get_action_selector()

        assert selector._db_path == Path(tmp_path / "resolved.db")
        assert not (tmp_path / DEFAULT_DB_NAME).exists()

    def test_explicit_pin_survives_env(self, monkeypatch, tmp_path):
        """The broker's call shape: an explicit store, with $ANIMA_DB set."""
        from anima_mcp import agency

        monkeypatch.setattr(agency, "_action_selector", None)
        monkeypatch.setenv("ANIMA_DB", str(tmp_path / "main.db"))
        pinned = tmp_path / "broker_agency.db"

        selector = agency.get_action_selector(db_path=str(pinned))

        assert selector._db_path == pinned

    def test_binding_is_announced(self, monkeypatch, tmp_path, capsys):
        """First-call-wins used to be silent. It has to be greppable."""
        from anima_mcp import agency

        monkeypatch.setattr(agency, "_action_selector", None)
        pinned = tmp_path / "announced.db"

        agency.get_action_selector(db_path=str(pinned))

        err = capsys.readouterr().err
        assert "[Agency] action store:" in err
        assert str(pinned) in err


class TestReflectionStoreFollowsTheRule:
    """`SelfReflectionSystem` was the one store outside the persistence rule
    (found 2026-11-12): an empty path hard-coded ~/.anima/anima.db and ignored
    $ANIMA_DB, and a bare "anima.db" — which `loop_phases` passes when no
    identity store is up yet — bound to the working directory. The singleton
    is first-call-wins, so whichever caller came first decided where Lumen's
    insights lived."""

    def test_env_used_when_caller_says_nothing(self, monkeypatch, tmp_path):
        from anima_mcp.self_reflection import SelfReflectionSystem

        target = tmp_path / "from_env.db"
        monkeypatch.setenv("ANIMA_DB", str(target))
        assert SelfReflectionSystem().db_path == target
        assert SelfReflectionSystem("").db_path == target

    def test_bare_default_does_not_bind_to_cwd(self, monkeypatch, tmp_path):
        from anima_mcp.self_reflection import SelfReflectionSystem

        monkeypatch.delenv("ANIMA_DB", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        resolved = SelfReflectionSystem("anima.db").db_path
        assert resolved.is_absolute()
        assert resolved == tmp_path / "home" / ".anima" / "anima.db"

    def test_explicit_path_wins(self, monkeypatch, tmp_path):
        from anima_mcp.self_reflection import SelfReflectionSystem

        monkeypatch.setenv("ANIMA_DB", str(tmp_path / "from_env.db"))
        explicit = tmp_path / "explicit.db"
        assert SelfReflectionSystem(str(explicit)).db_path == explicit
