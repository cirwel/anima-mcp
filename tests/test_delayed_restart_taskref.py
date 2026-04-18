"""Ensure _delayed_restart tasks are retained so they aren't GC'd mid-sleep.

Python's asyncio.create_task() keeps only a weak reference to the task. A
bare `asyncio.create_task(_delayed_restart())` would let the garbage
collector reap the task before its sleep completes, silently turning
restart=true into a no-op. _spawn_delayed_restart pins the task in
_BACKGROUND_TASKS until it finishes.
"""

from __future__ import annotations

import asyncio
import pytest

from anima_mcp.handlers import system_ops


@pytest.mark.asyncio
async def test_spawn_adds_task_to_tracked_set():
    async def _noop():
        await asyncio.sleep(0.01)

    system_ops._BACKGROUND_TASKS.clear()
    # Patch _delayed_restart so the test doesn't actually restart services.
    saved = system_ops._delayed_restart
    system_ops._delayed_restart = _noop
    try:
        system_ops._spawn_delayed_restart()
        assert len(system_ops._BACKGROUND_TASKS) == 1
        (task,) = tuple(system_ops._BACKGROUND_TASKS)
        assert not task.done()
        await asyncio.sleep(0.05)
        assert task.done()
    finally:
        system_ops._delayed_restart = saved


@pytest.mark.asyncio
async def test_spawn_removes_task_on_completion():
    async def _noop():
        await asyncio.sleep(0)

    system_ops._BACKGROUND_TASKS.clear()
    saved = system_ops._delayed_restart
    system_ops._delayed_restart = _noop
    try:
        system_ops._spawn_delayed_restart()
        # Give the event loop a tick for the task and its done-callback.
        await asyncio.sleep(0.05)
        assert len(system_ops._BACKGROUND_TASKS) == 0
    finally:
        system_ops._delayed_restart = saved


@pytest.mark.asyncio
async def test_background_tasks_is_set_of_tasks():
    assert isinstance(system_ops._BACKGROUND_TASKS, set)
