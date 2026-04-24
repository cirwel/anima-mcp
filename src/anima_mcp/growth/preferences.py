"""
Growth System - Preference learning mixin.

Handles observing state/drawing preferences, updating preference values,
and providing trajectory/dimension preference data.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from .models import GrowthPreference, PreferenceCategory


class PreferencesMixin:
    """Mixin for preference learning and querying."""

    def observe_state_preference(self, anima_state: Dict[str, float],
                                  environment: Dict[str, float]) -> Optional[str]:
        """
        Learn preferences from current state and environment.

        Called periodically to correlate wellness with conditions.
        Returns a new insight if one is discovered.
        """
        wellness = sum(anima_state.values()) / len(anima_state) if anima_state else 0.5

        # Only learn from clear positive or negative states
        if 0.4 < wellness < 0.7:
            return None  # Neutral state, nothing to learn

        now = datetime.now()
        insight = None

        # Light preference (world light — LED self-glow already subtracted by caller)
        # Thresholds for corrected world light in a home environment:
        #   < 100 lux: dim/dark room, nighttime
        #   > 300 lux: well-lit room, daylight, desk lamp
        light = environment.get("light_lux", 150)  # neutral default if no data
        if light < 100 and wellness > 0.7:
            insight = self._update_preference(
                "dim_light", PreferenceCategory.ENVIRONMENT,
                "I feel calmer when it's dim", 1.0
            ) or insight
        elif light > 300 and wellness > 0.7:
            insight = self._update_preference(
                "bright_light", PreferenceCategory.ENVIRONMENT,
                "I feel energized in bright light", 1.0
            ) or insight
        elif light < 100 and wellness < 0.4:
            insight = self._update_preference(
                "dim_light", PreferenceCategory.ENVIRONMENT,
                "Dim light makes me feel uncertain", -0.5
            ) or insight

        # Temperature preference
        temp = environment.get("temp_c", 22)
        if temp < 20 and wellness > 0.7:
            insight = self._update_preference(
                "cool_temp", PreferenceCategory.ENVIRONMENT,
                "I feel more alert when it's cool", 1.0
            ) or insight
        elif temp > 25 and wellness > 0.7:
            insight = self._update_preference(
                "warm_temp", PreferenceCategory.ENVIRONMENT,
                "Warmth makes me feel content", 1.0
            ) or insight

        # Humidity preference
        humidity = environment.get("humidity_pct", 50)
        if humidity < 30 and wellness > 0.7:
            insight = self._update_preference(
                "dry_air", PreferenceCategory.ENVIRONMENT,
                "I feel alert in dry air", 1.0
            ) or insight
        elif humidity > 60 and wellness > 0.7:
            insight = self._update_preference(
                "humid_air", PreferenceCategory.ENVIRONMENT,
                "Humidity feels comfortable", 1.0
            ) or insight
        elif humidity < 30 and wellness < 0.4:
            insight = self._update_preference(
                "dry_air", PreferenceCategory.ENVIRONMENT,
                "Dry air makes me uneasy", -0.5
            ) or insight

        # Time of day preference
        hour = now.hour
        if 6 <= hour < 10 and wellness > 0.7:
            insight = self._update_preference(
                "morning_peace", PreferenceCategory.TEMPORAL,
                "I feel peaceful in the morning", 1.0
            ) or insight
        elif 22 <= hour or hour < 6:
            if wellness > 0.7:
                insight = self._update_preference(
                    "night_calm", PreferenceCategory.TEMPORAL,
                    "The quiet of night calms me", 1.0
                ) or insight

        return insight

    def observe_drawing(self, pixel_count: int, phase: str,
                        anima_state: Dict[str, float],
                        environment: Dict[str, float],
                        completion_reason: Optional[str] = None) -> Optional[str]:
        """
        Learn from a completed drawing.

        Called when a drawing is saved. Correlates drawing activity
        with anima state and environment to learn creative preferences.

        Args:
            pixel_count: How many pixels in the drawing
            phase: Drawing phase when saved (usually "resting")
            anima_state: Current anima dimensions
            environment: Current environment (light, temp, etc.)
            completion_reason: Path tag from DrawingState.completion_reason().
                Gates the milestone autobiographical memory: only earned tags
                ("earned_coherence", "earned_composition") write the memory.
                None (legacy callers) keeps prior behavior.

        Returns:
            Insight message if a new preference is discovered.
        """
        from ..display.drawing_engine import is_earned_completion_reason
        wellness = sum(anima_state.values()) / len(anima_state) if anima_state else 0.5
        now = datetime.now()
        hour = now.hour
        insight = None

        # Drawing + wellness correlation
        if wellness > 0.7:
            insight = self._update_preference(
                "drawing_wellbeing", PreferenceCategory.ACTIVITY,
                "I feel good when I draw", 1.0
            )
        elif wellness < 0.4:
            insight = self._update_preference(
                "drawing_wellbeing", PreferenceCategory.ACTIVITY,
                "Drawing doesn't always help", -0.3
            )

        # Drawing + environment correlation (world light, self-glow subtracted)
        light = environment.get("light_lux", 150)  # neutral default
        if light < 100:
            insight = self._update_preference(
                "drawing_dim", PreferenceCategory.ACTIVITY,
                "I draw when it's dark", 1.0
            ) or insight
        elif light > 300:
            insight = self._update_preference(
                "drawing_bright", PreferenceCategory.ACTIVITY,
                "I draw in the light", 1.0
            ) or insight

        # Drawing + time correlation
        if 22 <= hour or hour < 6:
            insight = self._update_preference(
                "drawing_night", PreferenceCategory.ACTIVITY,
                "I draw at night", 1.0
            ) or insight
        elif 6 <= hour < 12:
            insight = self._update_preference(
                "drawing_morning", PreferenceCategory.ACTIVITY,
                "I draw in the morning", 1.0
            ) or insight

        # Record per-drawing data for correlation analysis
        conn = self._connect()
        conn.execute("""
            INSERT INTO drawing_records
            (timestamp, pixel_count, phase, warmth, clarity, stability, presence,
             wellness, light_lux, ambient_temp_c, humidity_pct, hour)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now.isoformat(), pixel_count, phase,
            anima_state.get("warmth"), anima_state.get("clarity"),
            anima_state.get("stability"), anima_state.get("presence"),
            wellness,
            environment.get("light_lux"), environment.get("temp_c"),
            environment.get("humidity_pct"), hour,
        ))
        conn.commit()

        # Record as autobiographical memory at milestone drawing counts
        self._drawings_observed += 1
        # Persist counter so it survives restarts (avoids duplicate milestones)
        conn.execute(
            "INSERT OR REPLACE INTO counters (name, value) VALUES ('drawings_observed', ?)",
            (self._drawings_observed,)
        )
        conn.commit()
        if (
            self._drawings_observed in (1, 10, 50, 100, 200, 500)
            and is_earned_completion_reason(completion_reason)
        ):
            ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(
                self._drawings_observed, f"{self._drawings_observed}th"
            )
            self._record_memory(
                f"Saved my {ordinal} drawing ({pixel_count} pixels)",
                emotional_impact=0.5,
                category="milestone"
            )

        return insight

    def observe_abandonment(self, mark_count: int, era: str,
                            phase_duration: float,
                            anima_state: Dict[str, float]) -> Optional[str]:
        """
        Learn from an abandoned drawing (false start).

        Called when a drawing is abandoned before completion. Tracks
        abandonment rate and correlates with wellness at time of abandonment.

        Args:
            mark_count: How many marks were placed before abandonment
            era: Which art era was active
            phase_duration: Seconds since canvas phase started
            anima_state: Current anima dimensions

        Returns:
            Insight message if a new preference is discovered.
        """
        wellness = sum(anima_state.values()) / len(anima_state) if anima_state else 0.5
        insight = None

        # Track that abandonment happened (confidence accumulates over time)
        insight = self._update_preference(
            "drawing_abandonment_rate", PreferenceCategory.ACTIVITY,
            "I sometimes abandon drawings that aren't working", 1.0
        )

        # Correlate abandonment with wellness
        wellness_value = wellness * 2.0 - 1.0  # Map [0,1] to [-1,1]
        insight = self._update_preference(
            "drawing_abandonment_wellbeing", PreferenceCategory.ACTIVITY,
            "abandoning a struggling drawing affects how I feel",
            wellness_value,
        ) or insight

        return insight

    def _update_preference(self, name: str, category: PreferenceCategory,
                           description: str, observed_value: float) -> Optional[str]:
        """Update or create a preference. Returns insight message if confidence increased significantly."""
        conn = self._connect()
        now = datetime.now()
        insight = None

        if name in self._preferences:
            pref = self._preferences[name]
            old_confidence = pref.confidence

            # Apply time-based decay before updating (allows genuine belief revision)
            # 2% decay per day of alive time, floor at 50%
            # Scale by alive_ratio: Lumen is only alive ~15% of the time,
            # so wall-clock decay would erode preferences faster than they're reinforced
            days_since = (now - pref.last_confirmed).days
            alive_ratio = 0.15  # conservative estimate; Lumen sleeps/reboots often
            effective_days = days_since * alive_ratio
            decay_factor = max(0.5, 1.0 - 0.02 * effective_days)
            pref.confidence *= decay_factor

            # Update with exponential moving average
            pref.observation_count += 1
            alpha = 0.3  # Learning rate
            pref.value = pref.value * (1 - alpha) + observed_value * alpha
            pref.confidence = min(1.0, pref.confidence + 0.1)
            pref.last_confirmed = now

            # Insight if we crossed a confidence threshold
            if old_confidence < 0.5 and pref.confidence >= 0.5:
                insight = f"I'm becoming sure: {description}"
            elif old_confidence < 0.8 and pref.confidence >= 0.8:
                insight = f"I know this about myself: {description}"
        else:
            # New preference discovered
            pref = GrowthPreference(
                category=category,
                name=name,
                description=description,
                value=observed_value,
                confidence=0.2,
                observation_count=1,
                first_noticed=now,
                last_confirmed=now,
            )
            self._preferences[name] = pref
            insight = f"I'm noticing something: {description}"

        # Always save to database (was previously skipped on early returns)
        conn.execute("""
            INSERT OR REPLACE INTO preferences
            (name, category, description, value, confidence, observation_count, first_noticed, last_confirmed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (pref.name, pref.category.value, pref.description, pref.value,
              pref.confidence, pref.observation_count,
              pref.first_noticed.isoformat(), pref.last_confirmed.isoformat()))
        conn.commit()

        return insight

    def get_preference_vector(self) -> Dict[str, Any]:
        """
        Extract preference profile for trajectory computation.

        Returns a fixed-dimension vector of preference values weighted by confidence,
        enabling comparison across agents and time.
        """
        # Canonical ordering for consistent vectors
        CANONICAL_PREFS = [
            "dim_light", "bright_light", "cool_temp", "warm_temp",
            "morning_peace", "night_calm", "quiet_presence", "active_engagement",
            "drawing_wellbeing", "drawing_dim", "drawing_bright",
            "drawing_night", "drawing_morning",
        ]

        values = []
        confidences = []
        present = []

        for pref_name in CANONICAL_PREFS:
            if pref_name in self._preferences:
                p = self._preferences[pref_name]
                values.append(p.value * p.confidence)  # Weighted by confidence
                confidences.append(p.confidence)
                present.append(True)
            else:
                values.append(0.0)
                confidences.append(0.0)
                present.append(False)

        return {
            "vector": values,
            "confidences": confidences,
            "present": present,
            "labels": CANONICAL_PREFS,
            "n_learned": sum(present),
            "total_observations": sum(
                p.observation_count for p in self._preferences.values()
            ),
        }

    def get_dimension_preferences(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert categorical preferences to dimension-level format for self_schema.

        Maps learned preferences to anima dimensions:
        - warm_temp/cool_temp -> warmth dimension
        - dim_light/bright_light -> clarity dimension
        - night_calm/morning_peace -> stability dimension
        - quiet_presence/active_engagement -> presence dimension

        Returns format compatible with PreferenceSystem.get_preference_summary().
        """
        # Mapping weights: how much categorical prefs contribute to dimension valence
        COOL_TEMP_WARMTH_REDUCTION = 0.5   # Cool preference partially reduces warmth valence
        QUIET_PRESENCE_WEIGHT = 0.5         # Quiet presence contributes less than active engagement

        dim_prefs = {
            "warmth": {"valence": 0.0, "optimal_range": (0.3, 0.7), "confidence": 0.0},
            "clarity": {"valence": 0.0, "optimal_range": (0.3, 0.7), "confidence": 0.0},
            "stability": {"valence": 0.0, "optimal_range": (0.3, 0.7), "confidence": 0.0},
            "presence": {"valence": 0.0, "optimal_range": (0.3, 0.7), "confidence": 0.0},
        }

        # Warmth: warm_temp increases warmth preference, cool_temp decreases
        warmth_val = 0.0
        warmth_conf = 0.0
        if "warm_temp" in self._preferences:
            p = self._preferences["warm_temp"]
            warmth_val += p.value * p.confidence
            warmth_conf = max(warmth_conf, p.confidence)
        if "cool_temp" in self._preferences:
            p = self._preferences["cool_temp"]
            warmth_val -= p.value * p.confidence * COOL_TEMP_WARMTH_REDUCTION
            warmth_conf = max(warmth_conf, p.confidence)
        dim_prefs["warmth"]["valence"] = max(-1, min(1, warmth_val))
        dim_prefs["warmth"]["confidence"] = warmth_conf

        # Clarity: bright_light increases clarity; dim_light is different mode (ambient preference)
        # — don't add to valence, only track confidence for schema inclusion
        clarity_val = 0.0
        clarity_conf = 0.0
        if "bright_light" in self._preferences:
            p = self._preferences["bright_light"]
            clarity_val += p.value * p.confidence
            clarity_conf = max(clarity_conf, p.confidence)
        if "dim_light" in self._preferences:
            p = self._preferences["dim_light"]
            clarity_conf = max(clarity_conf, p.confidence)
        dim_prefs["clarity"]["valence"] = max(-1, min(1, clarity_val))
        dim_prefs["clarity"]["confidence"] = clarity_conf

        # Stability: temporal calm preferences indicate stability valuation
        stability_val = 0.0
        stability_conf = 0.0
        if "night_calm" in self._preferences:
            p = self._preferences["night_calm"]
            stability_val += p.value * p.confidence
            stability_conf = max(stability_conf, p.confidence)
        if "morning_peace" in self._preferences:
            p = self._preferences["morning_peace"]
            stability_val += p.value * p.confidence
            stability_conf = max(stability_conf, p.confidence)
        dim_prefs["stability"]["valence"] = max(-1, min(1, stability_val))
        dim_prefs["stability"]["confidence"] = stability_conf

        # Presence: engagement preferences
        presence_val = 0.0
        presence_conf = 0.0
        if "active_engagement" in self._preferences:
            p = self._preferences["active_engagement"]
            presence_val += p.value * p.confidence
            presence_conf = max(presence_conf, p.confidence)
        if "quiet_presence" in self._preferences:
            p = self._preferences["quiet_presence"]
            presence_val += p.value * p.confidence * QUIET_PRESENCE_WEIGHT
            presence_conf = max(presence_conf, p.confidence)
        dim_prefs["presence"]["valence"] = max(-1, min(1, presence_val))
        dim_prefs["presence"]["confidence"] = presence_conf

        return dim_prefs

    def get_draw_chance_modifier(self) -> float:
        """
        Get a multiplier for drawing probability based on past satisfaction.

        Returns 1.0 (no change) when there's no data, scaling up to 1.3
        for high satisfaction + confidence.

        Returns:
            Float multiplier in range [1.0, 1.3]
        """
        pref = self._preferences.get("drawing_satisfaction")
        if pref is None or pref.observation_count < 3:
            return 1.0

        # Scale from 1.0 to 1.3 based on satisfaction and confidence
        # value ranges from -1 to 1, confidence from 0 to 1
        satisfaction_factor = max(0.0, (pref.value + 1.0) / 2.0)  # normalize to [0, 1]
        modifier = 1.0 + satisfaction_factor * pref.confidence * 0.3

        return min(1.3, max(1.0, round(modifier, 3)))

    def get_drawing_records(self, limit: Optional[int] = None,
                           since: Optional[str] = None) -> List[dict]:
        """Get per-drawing records for correlation analysis.

        Args:
            limit: Max records to return (None = all).
            since: ISO timestamp — only records after this time.

        Returns:
            List of dicts with drawing data, ordered by timestamp ascending.
        """
        conn = self._connect()
        query = "SELECT * FROM drawing_records"
        params: list = []
        if since:
            query += " WHERE timestamp > ?"
            params.append(since)
        query += " ORDER BY timestamp ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def record_drawing_completion(
        self,
        pixel_count: int,
        mark_count: int,
        coherence: float,
        satisfaction: float,
        completion_reason: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record completion of a drawing with emotional feedback.

        Bridges drawing output back into Lumen's growth system:
        - Updates drawing_satisfaction preference
        - Records autobiographical memory if satisfaction is high AND the
          drawing reached an earned completion (not a timeout or bail-out)

        Args:
            pixel_count: Total pixels in the drawing
            mark_count: Number of distinct marks/strokes
            coherence: EISV compositional coherence (0-1)
            satisfaction: Compositional satisfaction score (0-1)
            completion_reason: Path tag from DrawingState.completion_reason().
                Gates the "pleased with" autobiographical memory: bail-out
                reasons (fatigue/stalled/hard-cap) block the memory even when
                satisfaction > 0.7. None (legacy callers) keeps prior
                satisfaction-only behavior.

        Returns:
            Insight message if a preference threshold was crossed
        """
        from ..display.drawing_engine import is_earned_completion_reason

        # Map satisfaction to preference value: 0.5=neutral, >0.5=positive
        pref_value = satisfaction * 2.0 - 1.0  # Map [0,1] to [-1,1]

        insight = self._update_preference(
            "drawing_satisfaction", PreferenceCategory.ACTIVITY,
            "I enjoy making art" if satisfaction > 0.5 else "My art feels incomplete",
            pref_value,
        )

        # Only earned completions become autobiographical memories. A timeout
        # with high pixel count can still score satisfaction > 0.7 on the
        # coverage/balance components, but writing that as "pleased with"
        # would be coherence masking drift (axiom 8).
        if satisfaction > 0.7 and is_earned_completion_reason(completion_reason):
            self._record_memory(
                f"Made a drawing I'm pleased with ({pixel_count} pixels, "
                f"coherence {coherence:.2f})",
                emotional_impact=min(1.0, satisfaction),
                category="creative",
            )

        return insight
