"""
Tests for EISV mapper module.

Validates anima → EISV mapping accuracy and edge cases.
"""

import pytest
from datetime import datetime
from anima_mcp.eisv_mapper import (
    EISVMetrics,
    anima_to_eisv,
    estimate_complexity,
    generate_status_text,
    compute_eisv_from_readings
)
from anima_mcp.anima import Anima
from anima_mcp.sensors.base import SensorReadings


def create_test_readings(
    with_neural: bool = False,
    alpha: float = None,
    beta: float = None,
    gamma: float = None
) -> SensorReadings:
    """Create test sensor readings."""
    readings = SensorReadings(
        timestamp=datetime.now(),
        cpu_temp_c=45.0,
        ambient_temp_c=22.0,
        humidity_pct=50.0,
        light_lux=300.0,
        cpu_percent=50.0,
        memory_percent=50.0,
        disk_percent=50.0,
    )
    
    if with_neural:
        readings.eeg_alpha_power = alpha
        readings.eeg_beta_power = beta
        readings.eeg_gamma_power = gamma
    
    return readings


def create_test_anima(
    warmth: float = 0.5,
    clarity: float = 0.5,
    stability: float = 0.5,
    presence: float = 0.5
) -> Anima:
    """Create test anima state."""
    readings = create_test_readings()
    return Anima(
        warmth=warmth,
        clarity=clarity,
        stability=stability,
        presence=presence,
        readings=readings
    )


def test_eisv_basic_mapping():
    """Test basic anima → EISV mapping."""
    anima = create_test_anima(
        warmth=0.7,
        clarity=0.6,
        stability=0.8,
        presence=0.7
    )
    readings = create_test_readings()
    
    eisv = anima_to_eisv(anima, readings)
    
    # Energy should map from warmth
    assert 0.6 < eisv.energy < 0.8
    
    # Integrity should map from clarity
    assert 0.5 < eisv.integrity < 0.7
    
    # Entropy should be inverse of stability
    assert abs(eisv.entropy - (1.0 - 0.8)) < 0.1
    
    # Valence is the signed E-I imbalance (+hot / -careful)
    assert abs(eisv.valence - (eisv.energy - eisv.integrity)) < 1e-9
    assert -1.0 <= eisv.valence <= 1.0


def test_eisv_with_neural_signals():
    """Test EISV mapping with neural signals."""
    anima = create_test_anima(
        warmth=0.5,
        clarity=0.5,
        stability=0.5,
        presence=0.5
    )
    readings = create_test_readings(
        with_neural=True,
        alpha=0.8,  # High alpha = high clarity
        beta=0.7,   # High beta = high energy
        gamma=0.6   # High gamma = high presence
    )
    
    eisv = anima_to_eisv(anima, readings, neural_weight=0.3)
    
    # Energy should be boosted by beta/gamma
    assert eisv.energy > 0.5
    
    # Integrity should be boosted by alpha
    assert eisv.integrity > 0.5


def test_eisv_range_clamping():
    """Test that EISV values are clamped to [0, 1]."""
    # Extreme anima states
    anima = create_test_anima(
        warmth=2.0,  # Out of range
        clarity=-0.5,  # Out of range
        stability=0.0,
        presence=1.0
    )
    readings = create_test_readings()
    
    eisv = anima_to_eisv(anima, readings)
    
    assert 0.0 <= eisv.energy <= 1.0
    assert 0.0 <= eisv.integrity <= 1.0
    assert 0.0 <= eisv.entropy <= 1.0
    assert -1.0 <= eisv.valence <= 1.0


def test_eisv_neural_weight_adjustment():
    """Test that neural weight affects EISV mapping."""
    anima = create_test_anima(warmth=0.5, clarity=0.5)
    readings = create_test_readings(
        with_neural=True,
        beta=0.9,  # Very high beta
        gamma=0.9
    )
    
    # Low neural weight
    eisv_low = anima_to_eisv(anima, readings, neural_weight=0.1)
    
    # High neural weight
    eisv_high = anima_to_eisv(anima, readings, neural_weight=0.5)
    
    # High neural weight should result in higher energy
    assert eisv_high.energy > eisv_low.energy


def test_estimate_complexity():
    """Test complexity estimation."""
    # Low complexity: high clarity, high stability
    anima_low = create_test_anima(clarity=0.9, stability=0.9)
    complexity_low = estimate_complexity(anima_low)
    assert complexity_low < 0.3
    
    # High complexity: low clarity, low stability
    anima_high = create_test_anima(clarity=0.2, stability=0.2)
    complexity_high = estimate_complexity(anima_high)
    assert complexity_high > 0.4
    
    # Complexity should be in [0, 1]
    assert 0.0 <= complexity_low <= 1.0
    assert 0.0 <= complexity_high <= 1.0


def test_generate_status_text():
    """Test status text generation."""
    anima = create_test_anima(
        warmth=0.7,
        clarity=0.6,
        stability=0.8,
        presence=0.7
    )
    readings = create_test_readings(
        with_neural=True,
        alpha=0.5,
        beta=0.6,
        gamma=0.4
    )
    
    text = generate_status_text(anima, readings)
    
    assert "Anima state" in text
    assert "Warmth: 0.70" in text
    assert "Neural" in text
    assert "Alpha" in text


def test_compute_eisv_from_readings():
    """Test convenience function."""
    readings = create_test_readings(
        with_neural=True,
        alpha=0.6,
        beta=0.7,
        gamma=0.5
    )
    
    eisv = compute_eisv_from_readings(readings)
    
    assert isinstance(eisv, EISVMetrics)
    assert 0.0 <= eisv.energy <= 1.0
    assert 0.0 <= eisv.integrity <= 1.0


def test_eisv_to_dict():
    """Test EISV serialization."""
    eisv = EISVMetrics(
        energy=0.7,
        integrity=0.6,
        entropy=0.3,
        valence=0.2
    )

    d = eisv.to_dict()
    
    assert d["E"] == 0.7
    assert d["I"] == 0.6
    assert d["S"] == 0.3
    assert d["V"] == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

