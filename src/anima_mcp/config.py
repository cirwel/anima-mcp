"""
Configuration - Lumen's Nervous System Calibration

Configuration values define how Lumen interprets its senses.
These aren't just "settings" - they're the creature's nervous system calibration.

Adapts to:
- Environment (altitude, climate)
- Hardware (Pi model, sensor types)
- Learned preferences over time
"""

import json
import os
import sys
import uuid
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from .atomic_write import atomic_json_write


# === Retired LED Self-Glow Calibration Reference ===
# The VEML7700 light sensor sits next to the DotStar LEDs.
# This fixed quadratic is retained only so old calibration reports remain
# interpretable. Runtime sensing does not call it; the learned gated model in
# light_attribution.py must not use this coefficient as evidence or a prior.
#
# Empirically calibrated 2026-02-18 via manage_display(action="calibrate_leds"):
#   brightness=0.00 → ~15-44 lux (ambient only, LEDs off)
#   brightness=0.12 → ~29-62 lux (LED adds ~13-17 lux)
#   brightness=0.25 → ~99-112 lux (LED adds ~68-83 lux)
#
# Quadratic fit: glow = 1150 * brightness^2
#   At 0.12: 1150 * 0.0144 = 16.6 lux  (calibration: 13-17 ✓)
#   At 0.25: 1150 * 0.0625 = 71.9 lux  (calibration: 68-83 ✓)
#   At 0.04: 1150 * 0.0016 = 1.8 lux   (barely visible LEDs)
#
# Previous linear model (400*b + 8) overcorrected at low brightness:
#   At 0.12: gave 56 lux (actual ~16) — 3.5x too high.
LED_LUX_QUADRATIC: float = 1150.0       # retired diagnostics-only coefficient
WORLD_LIGHT_SMOOTH_WINDOW: int = 4       # rolling average samples (~8s at 2s interval)
LIGHT_SENSOR_EMA_ALPHA: float = 0.2      # Pi light channel smoothing per broker sample
VEML7700_INTEGRATION_SECONDS: float = 0.2  # physical ALS integration window
VEML7700_INTEGRATION_TOLERANCE: float = 0.30  # Vishay application note 84323
VEML7700_CAPTURE_SUPPORT_SECONDS: float = (
    2.0 * VEML7700_INTEGRATION_SECONDS * (1.0 + VEML7700_INTEGRATION_TOLERANCE)
)


def estimated_led_glow(brightness: float) -> float:
    """Return the retired fixed calibration curve for diagnostics only.

    LED glow is quadratic in brightness (non-linear LED response):
      brightness=0.00 → glow=0
      brightness=0.04 → glow≈1.8 lux
      brightness=0.12 → glow≈17 lux
      brightness=0.25 → glow≈72 lux
    """
    return LED_LUX_QUADRATIC * brightness * brightness


@dataclass
class NervousSystemCalibration:
    """
    How Lumen interprets its senses - the creature's nervous system calibration.
    
    These ranges define what Lumen considers "normal" vs "extreme" in its environment.
    """
    
    # Thermal ranges (Celsius)
    cpu_temp_min: float = 40.0      # Below this = cold
    cpu_temp_max: float = 80.0     # Above this = hot
    
    ambient_temp_min: float = 15.0  # Below this = cold environment
    ambient_temp_max: float = 35.0  # Above this = hot environment
    
    # Ideal values (deviation from these = instability)
    humidity_ideal: float = 45.0    # Ideal humidity (%)
    pressure_ideal: float = 1013.25 # Sea level standard (hPa)
    
    # Light perception (lux)
    light_min_lux: float = 1.0      # Unused — clarity uses hardcoded 1.0 floor. Kept for config compat.
    light_max_lux: float = 1000.0

    # Legacy fields — per-dimension weight dicts below are used instead.
    # Kept for config file backward compatibility.
    neural_weight: float = 0.3
    physical_weight: float = 0.7
    
    # Component weights for anima dimensions (must sum to ~1.0)
    # neural is 0.0 by design, not omitted: consumers use .get("neural", 0.0),
    # and a file merge is whole-field, so the key stays present to document the
    # decision. The (beta+gamma)/2 term is CPU% (beta = cpu_percent/100), which
    # made warmth partly busy-ness — the docstring on _sense_warmth already
    # says warmth is thermal state. Backtested over 14d before the flip
    # (2026-08-14): warmth 0.453->0.535, spurious below-comfort episodes
    # 15.9%->2.0%. CPU%'s single remaining path into E is the EISV mapper's
    # explicit neural_energy term (#166).
    warmth_weights: Dict[str, float] = field(default_factory=lambda: {
        "cpu_temp": 0.4375,
        "ambient_temp": 0.5625,
        "neural": 0.0,
    })
    
    # neural is 0.0 by design (see warmth_weights). Alpha = 1 - beta by
    # construction (computational_neural.py), i.e. the SAME CPU reading that
    # feeds E — so "alpha = relaxed awareness" was an idle Pi inflating I by a
    # 0.27 share and suppressing E from one variable, the double-count
    # CLAUDE.md warns neural consumers about. #141/#166 removed it from the
    # EISV mappers; this removes it at the source. Backtested 14d: clarity
    # 0.672->0.576 (honest-lower, more variance), V -0.320->-0.166.
    clarity_weights: Dict[str, float] = field(default_factory=lambda: {
        "prediction_accuracy": 0.625,  # How well I predict my own state = internal seeing
        "neural": 0.0,
        "sensor_coverage": 0.1875,     # Data richness
        "world_light": 0.1875,         # Gated external-lux residual. Raw lux
                                       # stays physical telemetry; this term is
                                       # omitted until attribution is ready.
                                       # Never restore the fixed quadratic.
    })
    
    stability_weights: Dict[str, float] = field(default_factory=lambda: {
        "humidity_dev": 0.25,
        "memory": 0.3,
        "missing_sensors": 0.2,
        "pressure_dev": 0.15,  # Pressure sensor contribution
        "neural": 0.1,
    })
    
    # neural is 0.0 by design (same pattern as warmth/clarity, #173): the
    # neural term here is gamma, and corr(gamma, cpu_percent) = +0.743 measured
    # over 14d — context switches are mostly the same variable as the direct
    # cpu door, so CPU% entered presence twice at an effective ~0.40 share. In
    # degraded mode (no ctx stats) the fallback was gamma = beta*0.5: exact
    # aliasing. Presence is resource headroom, read once per resource.
    # Backtested before the flip: presence 0.733 -> 0.713, sd 0.016 -> 0.012.
    # Face expression thresholds derived from THIS creature's lived
    # distribution (scripts/derive_face_thresholds.py). Empty = use the
    # built-in defaults in display/face.py — fresh installs render identically
    # to before this field existed. Keys override 1:1; the two absolute safety
    # floors (WARMTH_FREEZING, STABILITY_DISTRESSED) are clamped in
    # FaceThresholds and refused by the derivation script.
    face_thresholds: Dict[str, float] = field(default_factory=dict)

    # Clarity cuts that pick a drawing's coverage intention, derived from THIS
    # creature's lived clarity distribution (scripts/derive_drawing_thresholds.py).
    # Empty = use the built-in defaults in display/drawing_engine.py — fresh
    # installs generate goals identically to before this field existed. Keys
    # override 1:1; a non-monotone pair is rejected whole (it would starve
    # "balanced" the way the built-in 0.30 starved "dense").
    drawing_thresholds: Dict[str, float] = field(default_factory=dict)

    presence_weights: Dict[str, float] = field(default_factory=lambda: {
        "disk": 0.3125,
        "memory": 0.375,
        "cpu": 0.3125,
        "neural": 0.0,
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NervousSystemCalibration":
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate calibration values are sensible."""
        if self.cpu_temp_min >= self.cpu_temp_max:
            return False, "cpu_temp_min must be < cpu_temp_max"
        
        if self.ambient_temp_min >= self.ambient_temp_max:
            return False, "ambient_temp_min must be < ambient_temp_max"
        
        if self.light_max_lux < 10.0:
            return False, "light_max_lux must be >= 10.0 (log scale floor)"

        if self.light_min_lux >= self.light_max_lux:
            return False, "light_min_lux must be < light_max_lux"
        
        if not (0 <= self.humidity_ideal <= 100):
            return False, "humidity_ideal must be 0-100"
        
        if self.pressure_ideal < 0:
            return False, "pressure_ideal must be positive"
        
        if not (0 <= self.neural_weight <= 1) or not (0 <= self.physical_weight <= 1):
            return False, "neural_weight and physical_weight must be 0-1"
        
        # Check weights sum to ~1.0 (allow some tolerance)
        for weight_dict in [self.warmth_weights, self.clarity_weights, 
                           self.stability_weights, self.presence_weights]:
            total = sum(weight_dict.values())
            if not (0.9 <= total <= 1.1):  # Allow 10% tolerance
                return False, f"Weights should sum to ~1.0, got {total}"
        
        return True, None


@dataclass
class DisplayConfig:
    """Display system configuration."""
    led_brightness: float = 0.04  # Base brightness (manual control)
    update_interval: float = 2.0
    breathing_enabled: bool = True
    breathing_cycle: float = 12.0  # Match design doc's 12s cycle
    breathing_variation: float = 0.1
    color_transitions_enabled: bool = True
    pattern_mode: str = "standard"  # Only "standard" supported now


@dataclass
class AnimaConfig:
    """Complete configuration for anima-mcp."""
    nervous_system: NervousSystemCalibration = field(default_factory=NervousSystemCalibration)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    
    # Metadata for tracking changes
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "calibration_last_updated": None,
        "calibration_last_updated_by": None,  # "manual", "automatic", "agent"
        "calibration_update_count": 0,
        "calibration_history": [],  # List of recent changes
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nervous_system": self.nervous_system.to_dict(),
            "display": asdict(self.display),
            "metadata": self.metadata.copy(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnimaConfig":
        """Create from dictionary."""
        return cls(
            nervous_system=NervousSystemCalibration.from_dict(
                data.get("nervous_system", {})
            ),
            display=DisplayConfig(**data.get("display", {})),
            metadata=data.get("metadata", {
                "calibration_last_updated": None,
                "calibration_last_updated_by": None,
                "calibration_update_count": 0,
                "calibration_history": [],
            }),
        )
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate entire configuration."""
        valid, error = self.nervous_system.validate()
        if not valid:
            return False, f"Nervous system calibration: {error}"
        
        if not (0 <= self.display.led_brightness <= 1):
            return False, "led_brightness must be 0-1"
        
        if self.display.update_interval <= 0:
            return False, "update_interval must be positive"
        
        return True, None


class ConfigManager:
    """Manages configuration loading, saving, and adaptation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to config file.  Defaults to ``$ANIMA_CONFIG``
                when set, otherwise ``anima_config.yaml`` in the current dir.
        """
        if config_path is None:
            config_path = Path(os.environ.get("ANIMA_CONFIG", "anima_config.yaml"))
        self.config_path = Path(config_path)
        self._config: Optional[AnimaConfig] = None
        self._loaded_signature: Optional[tuple[int, int, int, int]] = None

    def _file_signature(self) -> Optional[tuple[int, int, int, int]]:
        """Return a change token that notices same-size atomic replacements."""
        try:
            stat = self.config_path.stat()
            return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size
        except OSError:
            return None
    
    def _read_from_disk(self) -> AnimaConfig:
        """Parse the file into a fresh object, or defaults. Never touches the cache."""
        if not self.config_path.exists():
            return AnimaConfig()
        try:
            if self.config_path.suffix == ".yaml" or self.config_path.suffix == ".yml":
                with open(self.config_path, "r") as f:
                    data = yaml.safe_load(f)
            else:
                with open(self.config_path, "r") as f:
                    data = json.load(f)

            config = AnimaConfig.from_dict(data)

            # Validate
            valid, error = config.validate()
            if not valid:
                print(f"[Config] Warning: Invalid config, using defaults: {error}", file=sys.stderr, flush=True)
                return AnimaConfig()
            return config
        except Exception as e:
            print(f"[Config] Error loading config, using defaults: {e}", file=sys.stderr, flush=True)
            return AnimaConfig()

    def load(self, force_reload: bool = False) -> AnimaConfig:
        """Load configuration from file or return defaults."""
        current_signature = self._file_signature()
        if (
            self._config is not None
            and not force_reload
            and current_signature == self._loaded_signature
        ):
            return self._config
        
        self._config = self._read_from_disk()

        # Readers in the broker and server are separate processes.  Remember
        # which file revision produced this object so either process notices an
        # atomic calibration update on its next access.
        self._loaded_signature = self._file_signature()
        
        return self._config
    
    def save(self, config: Optional[AnimaConfig] = None, update_source: Optional[str] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Config to save (uses current if None)
            update_source: Source of update ("manual", "automatic", "agent") for tracking
        
        Returns:
            True if saved successfully
        """
        if config is None:
            config = self._config or self.load()
        
        # Validate before saving
        valid, error = config.validate()
        if not valid:
            print(f"[Config] Cannot save invalid config: {error}", file=sys.stderr, flush=True)
            return False
        
        # Track calibration changes
        metadata = None
        if update_source:
            from datetime import datetime
            # Compare against what is on disk, not self.load(): callers mutate
            # the cached object in place (or assign into it) before saving, so
            # the cache already holds the new values and every comparison came
            # out equal — calibration_update_count stayed 0 and
            # calibration_history stayed empty however often calibration moved.
            old_config = self._read_from_disk()
            old_cal = old_config.nervous_system.to_dict()
            new_cal = config.nervous_system.to_dict()
            
            # Detect changes
            changes = {}
            for key in old_cal:
                old_val = old_cal.get(key)
                new_val = new_cal.get(key)
                # Handle float comparison (allow small differences)
                if isinstance(old_val, float) and isinstance(new_val, float):
                    if abs(old_val - new_val) > 0.001:
                        changes[key] = {
                            "old": old_val,
                            "new": new_val,
                        }
                elif old_val != new_val:
                    changes[key] = {
                        "old": old_val,
                        "new": new_val,
                    }
            
            if changes:
                # Built on a copy and committed only after the write lands: a
                # failed save must not leave a phantom history entry in the
                # cached config for the next successful save to persist.
                metadata = dict(config.metadata or {})
                if "calibration_last_updated" not in metadata:
                    metadata = {
                        "calibration_last_updated": None,
                        "calibration_last_updated_by": None,
                        "calibration_update_count": 0,
                        "calibration_history": [],
                    }

                # Update metadata
                metadata["calibration_last_updated"] = datetime.now().isoformat()
                metadata["calibration_last_updated_by"] = update_source
                metadata["calibration_update_count"] = metadata.get("calibration_update_count", 0) + 1

                # Add to history (keep last 10)
                history_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "source": update_source,
                    "changes": changes,
                }
                history = list(metadata.get("calibration_history", []))
                history.append(history_entry)
                metadata["calibration_history"] = history[-10:]  # Keep last 10

        previous_metadata = config.metadata
        if metadata is not None:
            config.metadata = metadata
        try:
            data = config.to_dict()

            if self.config_path.suffix == ".yaml" or self.config_path.suffix == ".yml":
                # Atomic write for YAML (same pattern as atomic_json_write)
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self.config_path.with_name(
                    f"{self.config_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
                )
                try:
                    with open(tmp_path, "w") as f:
                        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                        f.flush()
                        os.fsync(f.fileno())
                    tmp_path.replace(self.config_path)

                    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    directory_fd = os.open(self.config_path.parent, directory_flags)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except BaseException:
                    tmp_path.unlink(missing_ok=True)
                    raise
            else:
                atomic_json_write(self.config_path, data, indent=2)

            self._config = config
            self._loaded_signature = self._file_signature()
            return True
        except Exception as e:
            config.metadata = previous_metadata
            print(f"[Config] Error saving config: {e}", file=sys.stderr, flush=True)
            return False
    
    def reload(self) -> AnimaConfig:
        """Force reload configuration from file."""
        self._config = None
        self._loaded_signature = None
        return self.load()
    
    def get_calibration(self) -> NervousSystemCalibration:
        """Get current nervous system calibration."""
        return self.load().nervous_system
    
    def get_display_config(self) -> DisplayConfig:
        """Get current display configuration."""
        return self.load().display
    
    def adapt_to_environment(
        self,
        observed_temps: list[float],
        observed_pressures: list[float],
        observed_humidity: list[float],
    ) -> NervousSystemCalibration:
        """
        Adapt calibration based on observed environment.
        
        Learns what's "normal" for this environment and adjusts ranges.
        
        Args:
            observed_temps: Observed ambient temperatures
            observed_pressures: Observed barometric pressures
            observed_humidity: Observed humidity values
        
        Returns:
            Adapted calibration
        """
        cal = self.get_calibration()
        
        # Adapt ambient temp range based on observations
        if observed_temps:
            temp_min = min(observed_temps)
            temp_max = max(observed_temps)
            # Expand range by 20% for safety margin
            range_expansion = (temp_max - temp_min) * 0.2
            cal.ambient_temp_min = max(0, temp_min - range_expansion)
            cal.ambient_temp_max = temp_max + range_expansion
        
        # Adapt pressure ideal based on observations
        if observed_pressures:
            cal.pressure_ideal = sum(observed_pressures) / len(observed_pressures)
        
        # Adapt humidity ideal based on observations
        if observed_humidity:
            cal.humidity_ideal = sum(observed_humidity) / len(observed_humidity)
            # Clamp to reasonable range
            cal.humidity_ideal = max(20, min(80, cal.humidity_ideal))
        
        return cal


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """Get global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_calibration() -> NervousSystemCalibration:
    """Get current nervous system calibration."""
    return get_config_manager().get_calibration()


def get_display_config() -> DisplayConfig:
    """Get current display configuration."""
    return get_config_manager().get_display_config()
