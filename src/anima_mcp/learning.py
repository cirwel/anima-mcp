"""
Adaptive Learning - Lumen learns from accumulated experience.

Longer persistence = more observations = better calibration.

The creature accumulates sensor readings over time and adapts its
nervous system calibration to match its actual environment.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import sqlite3
from pathlib import Path

from .config import NervousSystemCalibration, ConfigManager
from .db_paths import resolve_db_path


class AdaptiveLearner:
    """
    Learns calibration from accumulated sensor observations.
    
    The longer Lumen persists, the more it learns about its environment.
    """
    
    def __init__(self, db_path: str = "anima.db", learning_window_days: int = 7):
        """
        Initialize adaptive learner.
        
        Args:
            db_path: Path to identity database (contains state_history)
            learning_window_days: How many days of history to use for learning
        """
        self.db_path = Path(resolve_db_path(db_path))
        self.learning_window_days = learning_window_days
        self._conn: Optional[sqlite3.Connection] = None
    
    def _connect(self) -> sqlite3.Connection:
        """Connect to database."""
        if self._conn is None:
            # Use timeout and WAL mode for better concurrency
            # Shorter timeout for faster failure (5s instead of 30s)
            self._conn = sqlite3.connect(self.db_path, timeout=5.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")  # 5 seconds instead of 30
            self._conn.execute("PRAGMA read_uncommitted=1")  # Better concurrency with WAL
        return self._conn
    
    def detect_gap(self) -> Optional[timedelta]:
        """
        Detect time gap since Lumen was last alive.

        Uses the persisted heartbeat timestamp (updated every ~30s during
        operation) or the most recent sleep event as the best estimate of
        when Lumen was last running.  Falls back to state_history only if
        neither source is available (legacy databases).

        Returns:
            Time since Lumen was last alive, or None if no data exists
        """
        if not self.db_path.exists():
            return None

        conn = self._connect()

        # Best source: persisted heartbeat timestamp (accurate to ~30s)
        try:
            row = conn.execute(
                "SELECT last_heartbeat_at FROM identity LIMIT 1"
            ).fetchone()
            if row and row["last_heartbeat_at"]:
                last_alive = datetime.fromisoformat(row["last_heartbeat_at"])
                return datetime.now() - last_alive
        except Exception:
            pass  # Column may not exist yet (pre-migration)

        # Second best: most recent sleep event (graceful shutdown timestamp)
        try:
            row = conn.execute(
                "SELECT timestamp FROM events WHERE event_type = 'sleep' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row:
                last_sleep = datetime.fromisoformat(row["timestamp"])
                return datetime.now() - last_sleep
        except Exception:
            pass  # Events table may not exist

        # Legacy fallback: state_history (the server loop writes it every few minutes)
        row = conn.execute(
            """SELECT timestamp FROM state_history
               ORDER BY timestamp DESC
               LIMIT 1"""
        ).fetchone()

        if row is None:
            return None

        last_obs = datetime.fromisoformat(row["timestamp"])
        return datetime.now() - last_obs
    
    def get_recent_observations(
        self,
        days: Optional[int] = None
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Get recent sensor observations from state history.

        Handles gaps by expanding window if needed to get enough observations.

        Returns:
            Tuple of (temperatures, pressures, humidities, light_readings) lists
        """
        if not self.db_path.exists():
            return ([], [], [], [])

        conn = self._connect()
        days = days or self.learning_window_days

        # Detect gap - if there's a significant gap, expand window to get more data
        gap = self.detect_gap()
        if gap and gap.days > days:
            # Gap longer than window - expand window to include pre-gap data
            days = min(gap.days + 7, 30)  # Expand up to 30 days max

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Query state history for sensor readings
        rows = conn.execute(
            """SELECT sensors FROM state_history
               WHERE timestamp > ?
               ORDER BY timestamp DESC
               LIMIT 1000""",
            (cutoff,)
        ).fetchall()

        temps = []
        pressures = []
        humidities = []
        light_readings = []

        import json
        for row in rows:
            try:
                sensors = json.loads(row["sensors"])

                # Collect ambient temperatures
                if "ambient_temp_c" in sensors and sensors["ambient_temp_c"] is not None:
                    temps.append(float(sensors["ambient_temp_c"]))

                # Collect pressures
                if "pressure_hpa" in sensors and sensors["pressure_hpa"] is not None:
                    pressures.append(float(sensors["pressure_hpa"]))

                # Collect humidity
                if "humidity_pct" in sensors and sensors["humidity_pct"] is not None:
                    humidities.append(float(sensors["humidity_pct"]))

                # Calibrate the clarity light scale only from the gated
                # environmental residual. Raw VEML7700 lux includes DotStar
                # self-glow and would bake Lumen's own output into the scale.
                if (
                    "external_light_lux" in sensors
                    and sensors["external_light_lux"] is not None
                ):
                    lux = float(sensors["external_light_lux"])
                    if lux > 0:
                        light_readings.append(lux)

            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        return (temps, pressures, humidities, light_readings)
    
    def learn_calibration(
        self,
        current_calibration: NervousSystemCalibration,
        min_observations: int = 50
    ) -> Optional[NervousSystemCalibration]:
        """
        Learn calibration from accumulated observations.
        
        Args:
            current_calibration: Current calibration (starting point)
            min_observations: Minimum observations needed to learn
        
        Returns:
            Learned calibration, or None if not enough data
        """
        temps, pressures, humidities, light_readings = self.get_recent_observations()

        # Need minimum observations to learn
        if len(temps) < min_observations and len(pressures) < min_observations:
            return None

        learned = NervousSystemCalibration.from_dict(current_calibration.to_dict())

        # Learn ambient temperature range
        if len(temps) >= min_observations:
            temp_min = min(temps)
            temp_max = max(temps)
            # Expand range by 20% for safety margin
            range_expansion = (temp_max - temp_min) * 0.2
            learned.ambient_temp_min = max(-10, temp_min - range_expansion)
            learned.ambient_temp_max = temp_max + range_expansion
            # Ensure a comfortable baseline range (room temp should feel normal, not cold)
            # Minimum range: 15-35°C so 20-25°C feels comfortable (mid-range)
            learned.ambient_temp_min = min(learned.ambient_temp_min, 15.0)
            learned.ambient_temp_max = max(learned.ambient_temp_max, 35.0)

        # Learn pressure baseline (mean of observations)
        if len(pressures) >= min_observations:
            learned.pressure_ideal = sum(pressures) / len(pressures)

        # Learn humidity ideal (mean of observations)
        if len(humidities) >= min_observations:
            learned.humidity_ideal = sum(humidities) / len(humidities)
            # Clamp to reasonable range
            learned.humidity_ideal = max(20, min(80, learned.humidity_ideal))

        # Learn light max (95th percentile of observations)
        if len(light_readings) >= min_observations:
            sorted_light = sorted(light_readings)
            p95_idx = int(len(sorted_light) * 0.95)
            learned.light_max_lux = max(10.0, sorted_light[p95_idx])

        return learned
    
    def should_adapt(
        self,
        current_calibration: NervousSystemCalibration,
        learned_calibration: NervousSystemCalibration,
        threshold: float = 0.1
    ) -> bool:
        """
        Determine if calibration should be adapted.
        
        Args:
            current_calibration: Current calibration
            learned_calibration: Learned calibration
            threshold: Minimum change required to adapt (10% default)
        
        Returns:
            True if adaptation is significant enough
        """
        # Check if pressure changed significantly
        # Use absolute threshold (5 hPa) — percentage-based is too coarse for pressure
        # since a 20 hPa shift (only 2.5%) significantly affects stability
        pressure_abs_change = abs(
            learned_calibration.pressure_ideal - current_calibration.pressure_ideal
        )
        pressure_changed = pressure_abs_change > 5.0  # 5 hPa absolute threshold

        # Check if ambient temp range changed significantly
        temp_range_current = current_calibration.ambient_temp_max - current_calibration.ambient_temp_min
        temp_range_learned = learned_calibration.ambient_temp_max - learned_calibration.ambient_temp_min
        temp_change = abs(temp_range_learned - temp_range_current) / max(1, temp_range_current)

        # Check if humidity ideal changed significantly
        humidity_change = abs(
            learned_calibration.humidity_ideal - current_calibration.humidity_ideal
        ) / max(1, current_calibration.humidity_ideal)

        # Check if light max changed significantly
        light_change = abs(
            learned_calibration.light_max_lux - current_calibration.light_max_lux
        ) / max(1, current_calibration.light_max_lux)

        # Adapt if any change is significant
        return (pressure_changed or
                temp_change > threshold or
                humidity_change > threshold or
                light_change > threshold)
    
    def get_last_adaptation_time(self, config_manager: Optional[ConfigManager] = None) -> Optional[datetime]:
        """
        Get timestamp of last calibration adaptation (from config metadata).
        
        Args:
            config_manager: Config manager (creates if None)
        
        Returns:
            Last adaptation time, or None if never adapted
        """
        if config_manager is None:
            config_manager = ConfigManager()
        
        # Check config file modification time as proxy for last adaptation
        config_path = config_manager.config_path
        if config_path.exists():
            mtime = datetime.fromtimestamp(config_path.stat().st_mtime)
            return mtime
        
        return None
    
    def should_adapt_now(
        self,
        min_time_between_adaptations: timedelta = timedelta(minutes=5)
    ) -> bool:
        """
        Check if enough time has passed since last adaptation.
        
        Prevents redundant adaptations during continuous operation.
        
        Args:
            min_time_between_adaptations: Minimum time between adaptations
        
        Returns:
            True if enough time has passed
        """
        last_adapt = self.get_last_adaptation_time()
        if last_adapt is None:
            return True  # Never adapted before
        
        time_since = datetime.now() - last_adapt
        return time_since >= min_time_between_adaptations
    
    def adapt_calibration(
        self,
        config_manager: Optional[ConfigManager] = None,
        min_observations: int = 50,
        adaptation_threshold: float = 0.1,
        respect_cooldown: bool = True
    ) -> Tuple[bool, Optional[NervousSystemCalibration]]:
        """
        Adapt calibration based on learned observations.
        
        Handles gaps gracefully - will use expanded window if gap detected.
        
        Args:
            config_manager: Config manager (creates if None)
            min_observations: Minimum observations needed
            adaptation_threshold: Minimum change to adapt
            respect_cooldown: If True, respects minimum time between adaptations
        
        Returns:
            Tuple of (adapted: bool, new_calibration: Optional)
        """
        if config_manager is None:
            config_manager = ConfigManager()
        
        # Check cooldown (unless this is startup/resume after gap)
        if respect_cooldown and not self.should_adapt_now():
            return (False, None)
        
        current = config_manager.get_calibration()
        learned = self.learn_calibration(current, min_observations)
        
        if learned is None:
            return (False, None)
        
        if not self.should_adapt(current, learned, adaptation_threshold):
            return (False, None)
        
        # Save adapted calibration
        config = config_manager.load()
        config.nervous_system = learned
        
        if config_manager.save(config, update_source="automatic"):
            return (True, learned)
        
        return (False, None)
    
    def get_observation_count(self, days: Optional[int] = None) -> int:
        """
        Get count of observations in the learning window.
        
        Args:
            days: Learning window (uses default if None)
        
        Returns:
            Number of observations
        """
        if not self.db_path.exists():
            return 0
        
        conn = self._connect()
        days = days or self.learning_window_days
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        count = conn.execute(
            """SELECT COUNT(*) FROM state_history WHERE timestamp > ?""",
            (cutoff,)
        ).fetchone()[0]
        
        return count
    
    def can_learn(self, min_observations: int = 50) -> bool:
        """
        Check if there are enough observations to learn.
        
        Args:
            min_observations: Minimum observations needed
        
        Returns:
            True if enough observations exist
        """
        temps, pressures, humidities, _light = self.get_recent_observations()
        return (len(temps) >= min_observations or
                len(pressures) >= min_observations or
                len(humidities) >= min_observations)


# Global learner instance
_learner: Optional[AdaptiveLearner] = None


def get_learner(db_path: str = "anima.db") -> AdaptiveLearner:
    """Get global adaptive learner instance."""
    global _learner
    if _learner is None:
        _learner = AdaptiveLearner(db_path)
    return _learner
