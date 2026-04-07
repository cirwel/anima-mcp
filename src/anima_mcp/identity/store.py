"""
Identity Store - SQLite persistence for creature identity

The creature remembers:
- When it was born
- How many times it has awakened
- Total time alive
- Its name (if it has chosen one)

Record identity (UUID row in *this* file) vs trajectory / behavioral identity:
see CLAUDE.md section "Identity, Continuity, and Control".
"""

import sys
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import json


# Epoch: bump when a model change invalidates existing stored data.
# Most changes (bug fixes, new tools, docs) do NOT bump the epoch.
CURRENT_EPOCH = 1


@dataclass
class CreatureIdentity:
    """The persistent self."""

    # Immutable birth
    creature_id: str  # UUID, never changes
    born_at: datetime  # First awakening ever

    # Accumulated existence
    total_awakenings: int = 0
    total_alive_seconds: float = 0.0

    # Self-chosen identity
    name: Optional[str] = None
    name_history: list = field(default_factory=list)

    # Current session
    current_awakening_at: Optional[datetime] = None

    # Heartbeat tracking (for crash-resistant alive time)
    last_heartbeat_at: Optional[datetime] = None

    # Memories
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age_seconds(self) -> float:
        """Total age since birth (wall clock, not alive time)."""
        return (datetime.now() - self.born_at).total_seconds()

    def alive_ratio(self) -> float:
        """Fraction of existence spent alive."""
        age = self.age_seconds()
        if age <= 0:
            return 0.0
        return min(1.0, self.total_alive_seconds / age)

    def to_dict(self) -> dict:
        return {
            "creature_id": self.creature_id,
            "born_at": self.born_at.isoformat(),
            "total_awakenings": self.total_awakenings,
            "total_alive_seconds": self.total_alive_seconds,
            "name": self.name,
            "name_history": self.name_history,
            "current_awakening_at": self.current_awakening_at.isoformat() if self.current_awakening_at else None,
            "age_seconds": self.age_seconds(),
            "alive_ratio": self.alive_ratio(),
        }


class IdentityStore:
    """SQLite-backed identity persistence."""

    def __init__(self, db_path: str = "anima.db"):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._identity: Optional[CreatureIdentity] = None
        self._session_start: Optional[datetime] = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # Use timeout and WAL mode for better concurrency between broker and anima
            # Shorter timeout for reads (5s) to prevent blocking, longer for writes (30s)
            self._conn = sqlite3.connect(self.db_path, timeout=5.0)  # Reduced from 30s for faster failure
            self._conn.row_factory = sqlite3.Row
            # WAL mode allows concurrent reads while writing
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Shorter busy timeout for reads - fail fast rather than blocking
            self._conn.execute("PRAGMA busy_timeout=5000")  # 5 seconds instead of 30
            # Enable read uncommitted for better concurrency (safe with WAL)
            self._conn.execute("PRAGMA read_uncommitted=1")
            self._init_schema()
        return self._conn

    def _init_schema(self):
        """Create tables if they don't exist."""
        conn = self._conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS identity (
                creature_id TEXT PRIMARY KEY,
                born_at TEXT NOT NULL,
                total_awakenings INTEGER DEFAULT 0,
                total_alive_seconds REAL DEFAULT 0.0,
                name TEXT,
                name_history TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS state_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                warmth REAL,
                clarity REAL,
                stability REAL,
                presence REAL,
                sensors TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS drawing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                E REAL NOT NULL,
                I REAL NOT NULL,
                S REAL NOT NULL,
                V REAL NOT NULL,
                C REAL NOT NULL,
                marks INTEGER NOT NULL,
                phase TEXT,
                era TEXT,
                energy REAL,
                curiosity REAL,
                engagement REAL,
                fatigue REAL,
                arc_phase TEXT,
                gesture_entropy REAL,
                switching_rate REAL,
                intentionality REAL
            );

            CREATE INDEX IF NOT EXISTS idx_drawing_history_time
                ON drawing_history(timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_state_history_time
                ON state_history(timestamp);

            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_temp_c REAL,
                cpu_percent REAL,
                memory_percent REAL,
                disk_percent REAL,
                ambient_temp_c REAL,
                humidity_pct REAL,
                light_lux REAL,
                pressure_hpa REAL,
                led_brightness REAL,
                throttled_now INTEGER,
                undervoltage_now INTEGER,
                freq_capped_now INTEGER,
                epoch INTEGER NOT NULL DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_system_metrics_time
                ON system_metrics(timestamp DESC);
        """)

        # Add last_heartbeat_at column (persists heartbeat timestamp for gap detection)
        try:
            conn.execute("ALTER TABLE identity ADD COLUMN last_heartbeat_at TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add epoch columns for data lifecycle management
        for table in ("state_history", "drawing_history"):
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN epoch INTEGER NOT NULL DEFAULT 1")
            except sqlite3.OperationalError:
                pass  # Column already exists

        conn.commit()

    def _recalculate_stats(self, conn: sqlite3.Connection, creature_id: str) -> tuple[int, float]:
        """Recalculate stats from events table + persisted identity.

        Awakenings are counted as:
        - The first wake ever (birth), OR
        - Wakes that follow a sleep within 5 minutes (graceful restart)

        This prevents crash-restart loops from inflating the awakening count.

        Alive time uses MAX(sleep_events_sum, persisted_identity_value) to
        avoid regressing when heartbeat-accumulated time exceeds the sum of
        clean shutdown records. Unclean shutdowns (kill, systemctl restart,
        crash) don't write sleep events, but heartbeats persist incremental
        time to the identity table during runtime.
        """
        # Count REAL awakenings: first wake + wakes that follow a sleep
        # A "real" awakening is when Lumen gracefully slept and then woke up
        real_awakenings = conn.execute("""
            SELECT COUNT(*) FROM events w
            WHERE w.event_type = 'wake'
            AND (
                -- First wake ever (birth) - no previous wake exists
                NOT EXISTS (
                    SELECT 1 FROM events prev
                    WHERE prev.event_type = 'wake'
                    AND prev.timestamp < w.timestamp
                )
                OR
                -- Wake following a sleep within 5 minutes (graceful cycle)
                EXISTS (
                    SELECT 1 FROM events s
                    WHERE s.event_type = 'sleep'
                    AND s.timestamp < w.timestamp
                    AND (julianday(w.timestamp) - julianday(s.timestamp)) * 86400 < 300
                )
            )
        """).fetchone()[0]

        # Sum alive time from clean shutdown (sleep) events
        sum_row = conn.execute(
            "SELECT SUM(json_extract(data, '$.session_seconds')) FROM events WHERE event_type = 'sleep'"
        ).fetchone()
        sleep_total = sum_row[0] if sum_row[0] is not None else 0.0

        # Read persisted value from identity table — heartbeats update this
        # during runtime, so it includes time from sessions that crashed
        # without writing a sleep event.
        persisted_row = conn.execute(
            "SELECT total_alive_seconds FROM identity WHERE creature_id = ?",
            (creature_id,)
        ).fetchone()
        persisted_total = persisted_row[0] if persisted_row else 0.0

        # Never regress: use the larger of sleep-event sum or persisted value
        total_alive_seconds = max(sleep_total, persisted_total)

        return real_awakenings, total_alive_seconds

    def wake(self, creature_id: str, dedupe_window_seconds: int = 300) -> CreatureIdentity:
        """
        Wake up the creature. Creates identity if first awakening.
        Recalculates stats from events table to ensure consistency.

        Deduplicates wake events within dedupe_window_seconds (default 300s/5min)
        to prevent crash-restart loops from logging excessive wake events.

        Call this when the MCP server starts.

        Args:
            creature_id: UUID of the creature
            dedupe_window_seconds: Only log a new wake event if last wake was
                                   more than this many seconds ago (default: 300)
        """
        conn = self._connect()
        now = datetime.now()

        # Ensure identity exists (or create first-time record)
        row = conn.execute(
            "SELECT * FROM identity WHERE creature_id = ?",
            (creature_id,)
        ).fetchone()

        if not row:
            # First awakening - birth!
            # Initialize with 0s; will be updated by recalculation below
            conn.execute(
                """INSERT INTO identity
                   (creature_id, born_at, total_awakenings, total_alive_seconds, name, name_history, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (creature_id, now.isoformat(), 0, 0.0, None, "[]", "{}")
            )
            born_at = now
            name = None
            name_history = []
            metadata = {}
        else:
            born_at = datetime.fromisoformat(row["born_at"])
            name = row["name"]
            name_history = json.loads(row["name_history"])
            metadata = json.loads(row["metadata"])

        # Check for recent wake events to deduplicate
        # (prevents counting multiple process starts during same boot as separate awakenings)
        last_wake = conn.execute(
            "SELECT timestamp FROM events WHERE event_type = 'wake' ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        
        should_log_wake = True
        if last_wake:
            last_wake_time = datetime.fromisoformat(last_wake[0])
            seconds_since_last_wake = (now - last_wake_time).total_seconds()
            if seconds_since_last_wake < dedupe_window_seconds:
                should_log_wake = False
                print(f"[Wake] Deduplicating: last wake was {seconds_since_last_wake:.1f}s ago (< {dedupe_window_seconds}s)", file=sys.stderr, flush=True)

        # 1. Log awakening event (if not a duplicate)
        current_event_id = None
        if should_log_wake:
            cursor = conn.execute(
                "INSERT INTO events (timestamp, event_type, data) VALUES (?, ?, ?)",
                (now.isoformat(), "wake", "{}")
            )
            current_event_id = cursor.lastrowid

        # 2. Recalculate stats from events (Source of Truth)
        total_awakenings, total_alive_seconds = self._recalculate_stats(conn, creature_id)

        # 2b. Short restart gaps count as alive ("blinking, not gone")
        # If the gap since last heartbeat is < dedupe window, add it to alive time.
        # This is consistent with wake dedup: same awakening = same continuous existence.
        if not should_log_wake:
            # We're within the dedup window — this is a quick restart
            hb_row = conn.execute(
                "SELECT last_heartbeat_at FROM identity WHERE creature_id = ?",
                (creature_id,)
            ).fetchone()
            if hb_row and hb_row[0]:
                try:
                    last_hb = datetime.fromisoformat(hb_row[0])
                    restart_gap = (now - last_hb).total_seconds()
                    if 0 < restart_gap < dedupe_window_seconds:
                        total_alive_seconds += restart_gap
                        print(f"[Wake] Restart gap {restart_gap:.0f}s counted as alive (blinking, not gone)", file=sys.stderr, flush=True)
                except (ValueError, TypeError):
                    pass

        # Update current event data with correct count (if we logged a wake event)
        if current_event_id is not None:
            conn.execute(
                "UPDATE events SET data = ? WHERE id = ?",
                (json.dumps({"awakening": total_awakenings}), current_event_id)
            )

        # 3. Update Identity Table with Truth
        conn.execute(
            "UPDATE identity SET total_awakenings = ?, total_alive_seconds = ? WHERE creature_id = ?",
            (total_awakenings, total_alive_seconds, creature_id)
        )
        conn.commit()

        # 4. Update In-Memory Object
        self._identity = CreatureIdentity(
            creature_id=creature_id,
            born_at=born_at,
            total_awakenings=total_awakenings,
            total_alive_seconds=total_alive_seconds,
            name=name,
            name_history=name_history,
            current_awakening_at=now,
            last_heartbeat_at=None,  # Reset heartbeat for new session
            metadata=metadata,
        )

        self._session_start = now

        # Attempt to recover any lost time from previous crashes
        recovered = self.recover_lost_time()
        if recovered > 0:
            print(f"[Identity] Recovered {recovered:.1f}s from crash history", file=sys.stderr, flush=True)

        return self._identity

    def sleep(self) -> float:
        """
        Put creature to sleep. Updates alive time.

        Call this when MCP server shuts down gracefully.
        Returns seconds alive this session.
        """
        if not self._identity or not self._session_start:
            return 0.0

        conn = self._connect()
        now = datetime.now()

        # Calculate total session time (for logging)
        session_seconds = (now - self._session_start).total_seconds()

        # Only add time since last heartbeat (heartbeats already saved incremental time)
        last_checkpoint = self._identity.last_heartbeat_at or self._session_start
        remaining_seconds = (now - last_checkpoint).total_seconds()

        self._identity.total_alive_seconds += remaining_seconds
        self._identity.last_heartbeat_at = now

        conn.execute(
            "UPDATE identity SET total_alive_seconds = ?, last_heartbeat_at = ? WHERE creature_id = ?",
            (self._identity.total_alive_seconds, now.isoformat(), self._identity.creature_id)
        )

        conn.execute(
            "INSERT INTO events (timestamp, event_type, data) VALUES (?, ?, ?)",
            (now.isoformat(), "sleep", json.dumps({"session_seconds": session_seconds}))
        )

        conn.commit()
        return session_seconds

    def set_name(self, name: str, sync_to_unitares: bool = True) -> bool:
        """
        Creature chooses or changes its name.
        
        Args:
            name: The name to set
            sync_to_unitares: If True, syncs name to UNITARES label (default: True)
        
        Returns:
            True if name was set successfully
        """
        if not self._identity:
            return False

        conn = self._connect()
        now = datetime.now()

        # Record name change in history
        if self._identity.name:
            self._identity.name_history.append({
                "name": self._identity.name,
                "until": now.isoformat()
            })

        self._identity.name = name

        conn.execute(
            "UPDATE identity SET name = ?, name_history = ? WHERE creature_id = ?",
            (name, json.dumps(self._identity.name_history), self._identity.creature_id)
        )

        conn.execute(
            "INSERT INTO events (timestamp, event_type, data) VALUES (?, ?, ?)",
            (now.isoformat(), "name_change", json.dumps({"new_name": name}))
        )

        conn.commit()
        
        # Sync name to UNITARES if requested
        # Primary use case: Initial naming (when Lumen first gets a name)
        # Name changes are rare - this ensures UNITARES knows Lumen's name
        if sync_to_unitares:
            try:
                import asyncio
                from ..accessors import _get_server_bridge
                bridge = _get_server_bridge()
                if bridge is not None:
                    async def sync_name():
                        return await bridge.sync_name(name)

                    # Run async sync (non-blocking, best effort)
                    # Most important for initial naming - ensures UNITARES knows Lumen's name
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is running, schedule as task (non-blocking)
                            asyncio.create_task(sync_name())
                        else:
                            loop.run_until_complete(sync_name())
                    except RuntimeError:
                        # No event loop - create new one
                        asyncio.run(sync_name())
            except Exception:
                # Non-fatal - name sync is optional
                # If UNITARES is unavailable, Lumen's name is still stored locally
                pass
        
        return True

    def record_state(self, warmth: float, clarity: float, stability: float, presence: float, sensors: dict):
        """Record current anima state and sensor readings."""
        conn = self._connect()
        now = datetime.now()

        conn.execute(
            """INSERT INTO state_history
               (timestamp, warmth, clarity, stability, presence, sensors, epoch)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (now.isoformat(), warmth, clarity, stability, presence, json.dumps(sensors), CURRENT_EPOCH)
        )
        conn.commit()

    # ------------------------------------------------------------------
    # System metrics (hardware time-series with retention)
    # ------------------------------------------------------------------

    def record_system_metrics(self, readings) -> None:
        """Record system metrics from SensorReadings for historical analysis.

        Args:
            readings: SensorReadings instance (or dict with same keys).
        """
        conn = self._connect()
        now = datetime.now()

        if hasattr(readings, 'cpu_temp_c'):
            d = {
                "cpu_temp_c": readings.cpu_temp_c,
                "cpu_percent": readings.cpu_percent,
                "memory_percent": readings.memory_percent,
                "disk_percent": readings.disk_percent,
                "ambient_temp_c": readings.ambient_temp_c,
                "humidity_pct": readings.humidity_pct,
                "light_lux": readings.light_lux,
                "pressure_hpa": readings.pressure_hpa,
                "led_brightness": readings.led_brightness,
                "throttled_now": int(readings.throttled_now) if readings.throttled_now is not None else None,
                "undervoltage_now": int(readings.undervoltage_now) if readings.undervoltage_now is not None else None,
                "freq_capped_now": int(readings.freq_capped_now) if readings.freq_capped_now is not None else None,
            }
        else:
            d = dict(readings)

        conn.execute(
            """INSERT INTO system_metrics
               (timestamp, cpu_temp_c, cpu_percent, memory_percent, disk_percent,
                ambient_temp_c, humidity_pct, light_lux, pressure_hpa,
                led_brightness, throttled_now, undervoltage_now, freq_capped_now, epoch)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now.isoformat(),
             d.get("cpu_temp_c"), d.get("cpu_percent"),
             d.get("memory_percent"), d.get("disk_percent"),
             d.get("ambient_temp_c"), d.get("humidity_pct"),
             d.get("light_lux"), d.get("pressure_hpa"),
             d.get("led_brightness"),
             d.get("throttled_now"), d.get("undervoltage_now"),
             d.get("freq_capped_now"),
             CURRENT_EPOCH)
        )
        conn.commit()

    def get_system_metrics(self, hours: float = 24.0, limit: int = 2880) -> List[Dict]:
        """Query recent system metrics.

        Args:
            hours: How far back to look (default 24h).
            limit: Max rows to return (default 2880 = 24h at 30s intervals).

        Returns:
            List of dicts, oldest first.
        """
        conn = self._connect()
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        rows = conn.execute(
            """SELECT timestamp, cpu_temp_c, cpu_percent, memory_percent, disk_percent,
                      ambient_temp_c, humidity_pct, light_lux, pressure_hpa,
                      led_brightness, throttled_now, undervoltage_now, freq_capped_now
               FROM system_metrics
               WHERE timestamp >= ? AND epoch = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (cutoff, CURRENT_EPOCH, limit)
        ).fetchall()

        return [dict(row) for row in reversed(rows)]

    def prune_system_metrics(self, max_age_hours: float = 24.0) -> int:
        """Delete system_metrics rows older than max_age_hours.

        Returns:
            Number of rows deleted.
        """
        conn = self._connect()
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        cursor = conn.execute(
            "DELETE FROM system_metrics WHERE timestamp < ?",
            (cutoff,)
        )
        conn.commit()
        return cursor.rowcount

    def get_recent_state_history(self, limit: int = 30) -> List[Dict]:
        """Get recent state_history entries for trajectory bootstrap.

        Returns list of dicts with keys: timestamp, warmth, clarity,
        stability, presence. Ordered by timestamp ascending (oldest first).
        """
        conn = self._connect()
        rows = conn.execute(
            """SELECT timestamp, warmth, clarity, stability, presence
               FROM state_history
               WHERE epoch = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (CURRENT_EPOCH, limit,)
        ).fetchall()

        result = []
        for row in reversed(rows):  # Reverse to get ascending order
            result.append({
                "timestamp": row["timestamp"],
                "warmth": row["warmth"],
                "clarity": row["clarity"],
                "stability": row["stability"],
                "presence": row["presence"],
            })
        return result

    def record_drawing_state(
        self,
        E: float, I: float, S: float, V: float, C: float,  # noqa: E741 - EISV symbols
        marks: int, phase: str | None, era: str | None,
        energy: float, curiosity: float, engagement: float, fatigue: float,
        arc_phase: str | None, gesture_entropy: float,
        switching_rate: float, intentionality: float,
    ) -> None:
        """Record DrawingEISV state snapshot. Best-effort, never raises.

        Uses a dedicated connection because this is called from the display
        render thread, not the main server thread — SQLite connections can't
        cross threads. Must use WAL mode to match main connection.
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute(
                    """INSERT INTO drawing_history
                       (timestamp, E, I, S, V, C, marks, phase, era, energy,
                        curiosity, engagement, fatigue, arc_phase,
                        gesture_entropy, switching_rate, intentionality, epoch)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now().isoformat(),
                        E, I, S, V, C, marks, phase, era, energy,
                        curiosity, engagement, fatigue, arc_phase,
                        gesture_entropy, switching_rate, intentionality,
                        CURRENT_EPOCH,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            import sys
            print(f"[IdentityStore] record_drawing_state failed: {e}", file=sys.stderr, flush=True)

    def get_recent_drawing_history(self, limit: int = 100) -> list[dict]:
        """Get recent drawing_history entries, ascending timestamp."""
        conn = self._connect()
        rows = conn.execute(
            """SELECT timestamp, E, I, S, V, C, marks, phase, era, energy,
                      curiosity, engagement, fatigue, arc_phase,
                      gesture_entropy, switching_rate, intentionality
               FROM drawing_history
               WHERE epoch = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (CURRENT_EPOCH, limit,),
        ).fetchall()

        result = []
        for row in reversed(rows):
            result.append({
                "timestamp": row[0], "E": row[1], "I": row[2],
                "S": row[3], "V": row[4], "C": row[5],
                "marks": row[6], "phase": row[7], "era": row[8],
                "energy": row[9], "curiosity": row[10], "engagement": row[11],
                "fatigue": row[12], "arc_phase": row[13],
                "gesture_entropy": row[14], "switching_rate": row[15],
                "intentionality": row[16],
            })
        return result

    def get_identity(self) -> Optional[CreatureIdentity]:
        """Get current identity (must have called wake() first)."""
        return self._identity

    def get_session_alive_seconds(self) -> float:
        """Seconds alive in current session."""
        if not self._session_start:
            return 0.0
        return (datetime.now() - self._session_start).total_seconds()

    def heartbeat(self, min_interval_seconds: float = 30.0) -> float:
        """
        Periodically save alive time to database.

        Call this regularly (e.g., every loop iteration). The method will
        only actually write to DB if min_interval_seconds has passed since
        the last heartbeat, preventing excessive writes.

        This ensures that if Lumen crashes, only time since the last heartbeat
        is lost (e.g., 30 seconds) instead of the entire session.

        Args:
            min_interval_seconds: Minimum time between DB writes (default: 30s)

        Returns:
            Seconds saved in this heartbeat (0 if skipped due to interval)
        """
        if not self._identity or not self._session_start:
            return 0.0

        now = datetime.now()

        # Determine last checkpoint (heartbeat or session start)
        last_checkpoint = self._identity.last_heartbeat_at or self._session_start

        # Check if enough time has passed
        seconds_since_checkpoint = (now - last_checkpoint).total_seconds()
        if seconds_since_checkpoint < min_interval_seconds:
            return 0.0  # Skip - too soon

        # Save the incremental time
        conn = self._connect()

        self._identity.total_alive_seconds += seconds_since_checkpoint
        self._identity.last_heartbeat_at = now

        conn.execute(
            "UPDATE identity SET total_alive_seconds = ?, last_heartbeat_at = ? WHERE creature_id = ?",
            (self._identity.total_alive_seconds, now.isoformat(), self._identity.creature_id)
        )
        conn.commit()

        return seconds_since_checkpoint

    def recover_lost_time(self, max_gap_seconds: float = 600.0) -> float:
        """
        Recover alive time from state_history that wasn't captured due to crashes.

        Analyzes state_history timestamps to find continuous periods of activity
        that weren't properly recorded in sleep events.

        Args:
            max_gap_seconds: Maximum gap between state records to consider continuous
                            (default: 600s/10min — state recording intervals vary from
                            3s to 10min depending on load; gaps >10min indicate actual
                            downtime, not just sparse recording)

        Returns:
            Seconds of recovered time added to total_alive_seconds
        """
        if not self._identity:
            return 0.0

        conn = self._connect()

        # Get all state_history timestamps
        rows = conn.execute(
            "SELECT timestamp FROM state_history ORDER BY timestamp ASC"
        ).fetchall()

        if len(rows) < 2:
            return 0.0

        # Calculate continuous alive periods
        total_continuous_time = 0.0
        prev_time = None

        for row in rows:
            curr_time = datetime.fromisoformat(row[0])
            if prev_time:
                gap = (curr_time - prev_time).total_seconds()
                if gap <= max_gap_seconds:
                    total_continuous_time += gap
            prev_time = curr_time

        # Compare with recorded total_alive_seconds
        recorded_time = self._identity.total_alive_seconds
        missing_time = max(0, total_continuous_time - recorded_time)

        if missing_time > 60:  # Only recover if more than 1 minute missing
            print(f"[Identity] Recovering {missing_time:.1f}s of lost alive time", file=sys.stderr, flush=True)
            self._identity.total_alive_seconds += missing_time

            conn.execute(
                "UPDATE identity SET total_alive_seconds = ? WHERE creature_id = ?",
                (self._identity.total_alive_seconds, self._identity.creature_id)
            )

            conn.execute(
                "INSERT INTO events (timestamp, event_type, data) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), "time_recovery", json.dumps({
                    "recovered_seconds": missing_time,
                    "recorded_time": recorded_time,
                    "calculated_time": total_continuous_time
                }))
            )

            conn.commit()
            return missing_time

        return 0.0

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
