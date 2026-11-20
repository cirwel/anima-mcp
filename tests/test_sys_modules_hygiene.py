"""Regression guards against runtime sys.modules lookup shims in anima_mcp."""

from __future__ import annotations

import ast
from pathlib import Path


def _sys_modules_accesses(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sys_aliases: set[str] = {"sys"}
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sys":
                    sys_aliases.add(alias.asname or "sys")
        elif isinstance(node, ast.ImportFrom) and node.module == "sys":
            for alias in node.names:
                if alias.name == "modules":
                    lines.append(node.lineno)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "modules"
            and isinstance(node.value, ast.Name)
            and node.value.id in sys_aliases
        ):
            lines.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in sys_aliases
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "modules"
        ):
            lines.append(node.lineno)
    return sorted(set(lines))


def test_no_runtime_sys_modules_shims_in_anima_mcp():
    src_root = Path(__file__).resolve().parents[1] / "src" / "anima_mcp"
    py_files = sorted(src_root.rglob("*.py"))
    assert py_files, f"Expected Python files under {src_root}"
    offenders: dict[str, list[int]] = {}
    for py_file in py_files:
        lines = _sys_modules_accesses(py_file)
        if lines:
            offenders[str(py_file.relative_to(src_root))] = lines
    assert offenders == {}, f"Unexpected sys.modules lookups in src/anima_mcp: {offenders}"


def test_creature_runtime_learning_modules_all_import_cleanly():
    """Ensure relocating learning imports to creature_runtime introduced no hidden cycles."""
    from anima_mcp import creature_runtime

    assert creature_runtime._LEARNING_MODULES
    assert all(creature_runtime._LEARNING_MODULES.values()), (
        f"Some learning modules failed to import: {creature_runtime._LEARNING_MODULES}"
    )


def test_drawing_engine_does_not_expose_stale_canvas_path_binding():
    """Patching drawing_engine._get_canvas_path must fail fast rather than silently no-op."""
    from anima_mcp.display import drawing_engine

    assert not hasattr(drawing_engine, "_get_canvas_path")
