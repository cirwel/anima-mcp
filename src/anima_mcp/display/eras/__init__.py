"""
Art Era Registry — manages available drawing eras and rotation.

Eras are pluggable modules that define Lumen's visual character per drawing session.
Each drawing belongs to one era. Era selection is manual by default — pick an era
on the art eras screen and Lumen stays in it until you change it.

Auto-rotate can be toggled on from the art eras screen. When on, Lumen picks a
different era after each drawing completes (weighted random across all registered eras).
"""

import random
from typing import Dict, List


# Registry of all available eras
_ERAS: Dict[str, object] = {}

# Auto-rotate toggle — when True, era changes after each drawing.
# When False (default), Lumen stays in the selected era until manually changed.
auto_rotate: bool = False


def register_era(era) -> None:
    """Register an era module."""
    _ERAS[era.name] = era


def get_era(name: str):
    """Get era by name. Falls back to 'gestural' if not found."""
    return _ERAS.get(name) or _ERAS.get("gestural")


def get_era_info(name: str) -> dict:
    """Get era metadata."""
    era = _ERAS.get(name)
    if not era:
        return {}
    return {
        "name": era.name,
        "description": era.description,
    }


def list_eras() -> List[str]:
    """List all registered era names."""
    return list(_ERAS.keys())


def list_all_era_info() -> List[dict]:
    """List all eras with metadata."""
    return [get_era_info(name) for name in _ERAS]


def choose_next_era(current: str, drawings_saved: int) -> str:
    """Choose era for next drawing.

    If auto_rotate is False, returns the current era (no change).
    If auto_rotate is True, picks a random era (lower weight for repeating).
    """
    if not auto_rotate:
        return current

    candidates = [
        name for name, era in _ERAS.items()
        if getattr(era, 'min_drawings', 0) <= drawings_saved
    ]
    if len(candidates) <= 1:
        return candidates[0] if candidates else "gestural"

    weights = [0.3 if name == current else 1.0 for name in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]


# --- Register eras at import time ---
from .gestural import GesturalEra  # noqa: E402
from .pointillist import PointillistEra  # noqa: E402
from .field import FieldEra  # noqa: E402
from .geometric import GeometricEra  # noqa: E402

register_era(GesturalEra())
register_era(PointillistEra())
register_era(FieldEra())
register_era(GeometricEra())

from .resonance import ResonanceEra  # noqa: E402
register_era(ResonanceEra())
