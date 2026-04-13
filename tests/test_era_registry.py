"""Tests for display/eras/__init__.py — era registry and rotation."""

import anima_mcp.display.eras as eras_module
from anima_mcp.display.eras import (
    register_era,
    get_era,
    get_era_info,
    list_eras,
    list_all_era_info,
    choose_next_era,
    _ERAS,
)


class TestRegistryPopulation:
    """Eras self-register on import."""

    def test_five_eras_registered(self):
        assert len(_ERAS) == 5

    def test_expected_era_names(self):
        names = set(_ERAS.keys())
        assert names == {"gestural", "pointillist", "field", "geometric", "resonance"}


class TestGetEra:
    def test_known_era(self):
        era = get_era("gestural")
        assert era is not None
        assert era.name == "gestural"

    def test_unknown_era_falls_back_to_gestural(self):
        era = get_era("nonexistent")
        assert era is not None
        assert era.name == "gestural"

    def test_each_registered_era(self):
        for name in list_eras():
            era = get_era(name)
            assert era is not None
            assert era.name == name


class TestGetEraInfo:
    def test_known_era(self):
        info = get_era_info("pointillist")
        assert info["name"] == "pointillist"
        assert "description" in info

    def test_unknown_era_returns_empty(self):
        info = get_era_info("nonexistent")
        assert info == {}


class TestListEras:
    def test_returns_list(self):
        result = list_eras()
        assert isinstance(result, list)
        assert len(result) == 5

    def test_all_era_info_matches(self):
        names = list_eras()
        infos = list_all_era_info()
        assert len(infos) == len(names)
        for info in infos:
            assert "name" in info
            assert "description" in info


class TestChooseNextEra:
    def test_no_rotate_returns_current(self):
        original = eras_module.auto_rotate
        try:
            eras_module.auto_rotate = False
            result = choose_next_era("geometric", 10)
            assert result == "geometric"
        finally:
            eras_module.auto_rotate = original

    def test_rotate_returns_valid_era(self):
        original = eras_module.auto_rotate
        try:
            eras_module.auto_rotate = True
            for _ in range(20):
                result = choose_next_era("gestural", 5)
                assert result in list_eras()
        finally:
            eras_module.auto_rotate = original

    def test_rotate_can_pick_different_era(self):
        """Over many tries, rotation should pick something other than current."""
        original = eras_module.auto_rotate
        try:
            eras_module.auto_rotate = True
            results = set()
            for _ in range(50):
                results.add(choose_next_era("gestural", 1))
            assert len(results) > 1
        finally:
            eras_module.auto_rotate = original

    def test_rotate_with_single_era(self, monkeypatch):
        """With only one era, returns that era."""
        original = eras_module.auto_rotate
        saved_eras = dict(_ERAS)
        try:
            eras_module.auto_rotate = True
            _ERAS.clear()
            _ERAS["only_one"] = type("FakeEra", (), {"name": "only_one", "description": "test"})()
            result = choose_next_era("only_one", 0)
            assert result == "only_one"
        finally:
            eras_module.auto_rotate = original
            _ERAS.clear()
            _ERAS.update(saved_eras)

    def test_rotate_with_empty_registry(self, monkeypatch):
        """With no eras, returns 'gestural' fallback."""
        original = eras_module.auto_rotate
        saved_eras = dict(_ERAS)
        try:
            eras_module.auto_rotate = True
            _ERAS.clear()
            result = choose_next_era("anything", 0)
            assert result == "gestural"
        finally:
            eras_module.auto_rotate = original
            _ERAS.clear()
            _ERAS.update(saved_eras)


class TestRegisterEra:
    def test_register_custom_era(self):
        saved_eras = dict(_ERAS)
        try:
            fake = type("FakeEra", (), {"name": "test_era", "description": "A test era"})()
            register_era(fake)
            assert "test_era" in _ERAS
            assert get_era("test_era").name == "test_era"
        finally:
            _ERAS.clear()
            _ERAS.update(saved_eras)


class TestMaturityGating:
    def test_era_with_min_drawings_excluded_when_below(self):
        original = eras_module.auto_rotate
        saved_eras = dict(_ERAS)
        try:
            eras_module.auto_rotate = True
            gated = type("GatedEra", (), {"name": "gated", "description": "test", "min_drawings": 50})()
            register_era(gated)
            results = set()
            for _ in range(100):
                results.add(choose_next_era("gestural", drawings_saved=10))
            assert "gated" not in results
        finally:
            eras_module.auto_rotate = original
            _ERAS.clear()
            _ERAS.update(saved_eras)

    def test_era_with_min_drawings_included_when_above(self):
        original = eras_module.auto_rotate
        saved_eras = dict(_ERAS)
        try:
            eras_module.auto_rotate = True
            gated = type("GatedEra", (), {"name": "gated", "description": "test", "min_drawings": 50})()
            register_era(gated)
            results = set()
            for _ in range(100):
                results.add(choose_next_era("gestural", drawings_saved=60))
            assert "gated" in results
        finally:
            eras_module.auto_rotate = original
            _ERAS.clear()
            _ERAS.update(saved_eras)

    def test_era_without_min_drawings_always_available(self):
        original = eras_module.auto_rotate
        try:
            eras_module.auto_rotate = True
            results = set()
            for _ in range(100):
                results.add(choose_next_era("gestural", drawings_saved=0))
            assert len(results) > 1
        finally:
            eras_module.auto_rotate = original

    def test_resonance_registered(self):
        assert "resonance" in _ERAS
        era = _ERAS["resonance"]
        assert era.name == "resonance"
        assert getattr(era, 'min_drawings', 0) == 50
