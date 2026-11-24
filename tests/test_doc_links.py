"""Guard against broken relative markdown links and dead docs/ references."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
BACKTICK_DOC_RE = re.compile(
    r"`((?:docs/)?(?:operations|features|plans|proposals|guides|archive)/[^`\s]+)`"
)
# Cross-repo citations explicitly labelled in prose as living in `unitares`.
CROSS_REPO_ALLOW = {
    ("docs/proposals/hearing-wire.md", "docs/proposals/archive/eisv-maths-roadmap-v0.md"),
}


def _iter_doc_files() -> list[Path]:
    root_mds = sorted(REPO_ROOT.glob("*.md"))
    docs_mds = sorted((REPO_ROOT / "docs").rglob("*.md"))
    return root_mds + docs_mds


def _strip_fenced_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def test_markdown_relative_links_resolve() -> None:
    broken: list[str] = []
    for fpath in _iter_doc_files():
        rel = fpath.relative_to(REPO_ROOT).as_posix()
        stripped = _strip_fenced_blocks(fpath.read_text(encoding="utf-8"))
        for lineno, line in enumerate(stripped.splitlines(), 1):
            for match in LINK_RE.finditer(line):
                raw = match.group(2).strip().split("#")[0].strip()
                if not raw:
                    continue
                target = raw.split()[0]
                if target.startswith(("http://", "https://", "mailto:", "file://")):
                    continue
                if not (fpath.parent / target).resolve().exists():
                    broken.append(f"{rel}:{lineno}: [{match.group(1)}]({match.group(2)})")
    assert not broken, "Broken relative markdown links:\n  " + "\n  ".join(broken)


def test_backtick_doc_paths_resolve() -> None:
    dead: list[str] = []
    for fpath in _iter_doc_files():
        rel = fpath.relative_to(REPO_ROOT).as_posix()
        stripped = _strip_fenced_blocks(fpath.read_text(encoding="utf-8"))
        for lineno, line in enumerate(stripped.splitlines(), 1):
            for match in BACKTICK_DOC_RE.finditer(line):
                cand = match.group(1).rstrip(".,;:)")
                if (rel, cand) in CROSS_REPO_ALLOW:
                    continue
                if "*" in cand or "<" in cand:
                    dead.append(f"{rel}:{lineno}: unresolvable glob/placeholder `{cand}`")
                    continue
                candidates = (
                    (fpath.parent / cand).resolve(),
                    (REPO_ROOT / cand).resolve(),
                    (REPO_ROOT / "docs" / cand).resolve(),
                )
                if not any(p.exists() for p in candidates):
                    dead.append(f"{rel}:{lineno}: `{cand}`")
    assert not dead, "Dead backtick doc references:\n  " + "\n  ".join(dead)
