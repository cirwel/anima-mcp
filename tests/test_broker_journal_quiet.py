"""The broker's ASCII face is drawn only on a terminal, never into the journal."""

import inspect
import io
import sys

from anima_mcp import stable_creature
from anima_mcp.stable_creature import _stdout_is_terminal


def test_a_pipe_or_journal_is_not_a_terminal(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert _stdout_is_terminal() is False


def test_a_closed_stdout_is_not_a_terminal(monkeypatch):
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(sys, "stdout", closed)
    assert _stdout_is_terminal() is False


def test_every_face_print_is_behind_the_terminal_guard():
    """On Lumen this frame was ~81% of journal lines (retention ~32h)."""
    source = inspect.getsource(stable_creature.run_creature)
    guard = source.index("if terminal_face:")
    for line in ('print(ascii_face)', 'print(f"W: {anima.warmth', 'print("\\033[2J\\033[H"'):
        assert source.count(line) == 1, line
        assert source.index(line) > guard, line
