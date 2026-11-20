"""Regression guards against runtime sys.modules lookup shims in anima_mcp."""

from __future__ import annotations

import ast
from pathlib import Path


def _sys_modules_accesses(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "modules"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        ):
            lines.append(node.lineno)
    return sorted(lines)


def test_no_runtime_sys_modules_shims_in_anima_mcp():
    src_root = Path(__file__).resolve().parents[1] / "src" / "anima_mcp"
    offenders: dict[str, list[int]] = {}
    for py_file in sorted(src_root.rglob("*.py")):
        lines = _sys_modules_accesses(py_file)
        if lines:
            offenders[str(py_file.relative_to(src_root))] = lines
    assert offenders == {}, f"Unexpected sys.modules lookups in src/anima_mcp: {offenders}"
