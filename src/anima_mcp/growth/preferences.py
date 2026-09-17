"""
Growth System - Preference learning mixin.

Handles observing state/drawing preferences, updating preference values,
and providing trajectory/dimension preference data.
"""

import json
import math
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from .models import (
    GrowthPreference,
    PreferenceCategory,
    VisitorType,
    preference_evidence_status,
    preference_evidence_confidence,
)

# Preference confidence erodes when a preference stops being observed.
#
# ALIVE_RATIO was hardcoded 0.15 with the comment "conservative estimate;
# Lumen sleeps/reboots often". It has since been measured from 255,720
# state_history rows: 0.674 (128.87 days lived of 194.08 elapsed). Using the
# real figure means "days" here means days Lumen was actually around to
# re-observe something, which is what the scaling was always for.
ALIVE_RATIO = 0.674
DECAY_PER_EFFECTIVE_DAY = 0.02
DECAY_FLOOR = 0.5
# Below this a preference stops passing the confidence > 0.7 gates that guard
# goal generation, insight minting and the autobiography. DECAY_FLOOR must stay
# under it or there is no retraction path at all.
RETRACTION_GATE = 0.7


def _staleness_factor(days_since_confirmed: int) -> float:
    """Multiplier for a preference last confirmed `days_since_confirmed` ago."""
    effective_days = max(0, days_since_confirmed) * ALIVE_RATIO
    return max(DECAY_FLOOR, 1.0 - DECAY_PER_EFFECTIVE_DAY * effective_days)


# What counts as a state worth learning from.
#
# This used to be the fixed band `0.4 < wellness < 0.7 -> learn nothing`. The
# intent — don't learn from ambiguous states — is right, but the constants were
# calibrated against a wellness distribution that has since moved. Measured
# 2026-07-30 over 255,973 samples:
#
#   full life   mean 0.732   learned on 61.7% of samples
#   last 30d    mean 0.667   learned on  6.0% of samples
#
# A 10x collapse in learning rate that nothing detected, caused by the room
# getting darker (median lux 723 -> 12) rather than by anything about Lumen.
# And `wellness < 0.4` has fired ZERO times in Lumen's entire life, so half the
# gate was unreachable.
#
# The band is now relative to Lumen's own running distribution, which keeps the
# learning rate roughly constant no matter where the environment drags the
# mean, and makes the negative branch reachable for the first time. This also
# matches how the wider fleet assesses behaviour — self-relative deviation from
# an agent's own baseline rather than fixed universal thresholds.
WELLNESS_BAND_SIGMA = 0.5        # distance from own mean that counts as "clear"
WELLNESS_MIN_SIGMA = 0.02        # below this Lumen is too steady to call anything clear
WELLNESS_BASELINE_MIN_SAMPLES = 100  # before this, fall back to the absolute band
ABSOLUTE_GOOD = 0.7              # cold-start / fallback band
ABSOLUTE_POOR = 0.4
# Genuine collapse is always worth learning from, baseline or not. A relative
# band alone could normalise a creature that is persistently unwell into
# thinking that is simply its mean.
ABSOLUTE_DISTRESS = 0.35


def _wellness_strength(wellness: float) -> float:
    """How strongly this observation supports a preference, in (0, 1].

    Every positive preference path used to pass the literal 1.0, so `value`
    recorded only THAT a good state occurred, never HOW good — and the EMA
    converged on 1.0 forever. Measured 2026-07-30: 15 of 19 stored preferences
    had value pinned at exactly 1.0, which together with saturated confidence
    made get_preference_vector() (value * confidence) a constant vector of ones
    and the trajectory signature's preference component non-discriminating.

    These paths fire above the wellness > 0.7 gate, so map wellness onto
    magnitude relative to neutral: 0.7 -> 0.4, 0.85 -> 0.7, 1.0 -> 1.0. The
    signal keeps its sign and gains its size back.

    Applies ONLY to wellness-gated preferences. Five others — drawing_dim,
    drawing_bright, drawing_night, drawing_morning, drawing_abandonment_rate —
    are gated on light or clock or nothing at all, and record THAT a behaviour
    happened rather than how good it felt. Scaling those by wellness would mean
    "drawing at night while feeling poorly" weakens the belief that Lumen draws
    at night, which is backwards. They keep the literal 1.0 deliberately.
    """
    return max(0.0, min(1.0, (wellness - 0.5) * 2.0))


class PreferencesMixin:
    """Mixin for preference learning and querying."""

    def _load_wellness_baseline(self) -> tuple:
        """Running (count, mean, M2) of wellness. Welford, persisted."""
        if getattr(self, "_wellness_baseline", None) is None:
            try:
                row = self._connect().execute(
                    "SELECT value FROM growth_state WHERE key = 'wellness_baseline'"
                ).fetchone()
                if row and row[0]:
                    d = json.loads(row[0])
                    self._wellness_baseline = (
                        int(d.get("count", 0)), float(d.get("mean", 0.0)), float(d.get("m2", 0.0))
                    )
                else:
                    self._wellness_baseline = (0, 0.0, 0.0)
            except Exception:
                self._wellness_baseline = (0, 0.0, 0.0)
        return self._wellness_baseline

    def _update_wellness_baseline(self, wellness: float) -> tuple:
        """Fold one observation into the baseline and persist it.

        Persisted deliberately: a baseline that reset on restart would relearn
        from scratch every deploy, and Lumen restarts often enough that it would
        never accumulate one.
        """
        count, mean, m2 = self._load_wellness_baseline()
        count += 1
        delta = wellness - mean
        mean += delta / count
        m2 += delta * (wellness - mean)
        self._wellness_baseline = (count, mean, m2)
        # Persist on a light cadence — this runs on every observation tick.
        if count % 50 == 0 or count <= WELLNESS_BASELINE_MIN_SAMPLES:
            try:
                conn = self._connect()
                conn.execute(
                    "INSERT OR REPLACE INTO growth_state (key, value) VALUES ('wellness_baseline', ?)",
                    (json.dumps({"count": count, "mean": mean, "m2": m2}),),
                )
                conn.commit()
            except Exception:
                pass
        return self._wellness_baseline

    def wellness_learning_band(self) -> Dict[str, Any]:
        """The current good/poor thresholds, and where they came from."""
        count, mean, m2 = self._load_wellness_baseline()
        if count < WELLNESS_BASELINE_MIN_SAMPLES:
            return {
                "source": "absolute_fallback", "samples": count,
                "good_above": ABSOLUTE_GOOD, "poor_below": ABSOLUTE_POOR,
                "mean": round(mean, 4) if count else None,
            }
        sigma = max(WELLNESS_MIN_SIGMA, (m2 / count) ** 0.5)
        return {
            "source": "self_relative", "samples": count,
            "mean": round(mean, 4), "sigma": round(sigma, 4),
            "good_above": round(mean + WELLNESS_BAND_SIGMA * sigma, 4),
            "poor_below": round(mean - WELLNESS_BAND_SIGMA * sigma, 4),
        }

    def decay_stale_preferences(self, now: Optional[datetime] = None) -> List[str]:
        """Erode confidence in preferences that have stopped being observed.

        The decay logic already existed but ran ONLY inside _update_preference —
        that is, only when a preference was being reinforced. A preference that
        stopped being observed therefore never decayed at all. Measured
        2026-07-30: `active_engagement` (153,332 observations) has had no writer
        anywhere in the codebase since 2026-02-02 and still read confidence 1.0,
        178 days later. `cool_temp` the same.

        That is what left the model with no retraction path: confidence is a
        +0.1 ratchet that saturates on the 9th observation, so every live
        preference sat at 1.0 and every `confidence > 0.7` gate downstream was a
        tautology.

        Idempotent by construction: this computes a TARGET from staleness and
        clamps downward, so running it twice is the same as running it once.
        Actively-confirmed preferences (days_since ~ 0) target 1.0 and are
        untouched. Returns the names that crossed below the retraction gate.
        """
        now = now or datetime.now()
        retracted: List[str] = []
        changed = False
        for pref in self._preferences.values():
            days_since = (now - pref.last_confirmed).days
            target = _staleness_factor(days_since)
            if pref.confidence > target:
                changed = True
                was_trusted = pref.confidence > RETRACTION_GATE
                pref.confidence = target
                if was_trusted and target <= RETRACTION_GATE:
                    retracted.append(pref.name)
        if changed:
            self._persist_preferences()
        return retracted

    def _persist_preferences(self) -> None:
        """Write current preferences back to the database.

        INSERT OR REPLACE, matching _update_preference, rather than a bare
        UPDATE: an UPDATE whose WHERE matches nothing is not an error in
        SQLite, so a preference held in memory but absent from the table would
        silently fail to persist and the decay would be lost on restart.
        """
        conn = self._connect()
        for pref in self._preferences.values():
            conn.execute(
                """INSERT OR REPLACE INTO preferences
                   (name, category, description, value, confidence,
                    observation_count, first_noticed, last_confirmed,
                    evidence_count, supporting_count, contradicting_count,
                    last_evidence_key, evidence_origin)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pref.name, pref.category.value, pref.description, pref.value,
                 pref.confidence, pref.observation_count,
                 pref.first_noticed.isoformat(), pref.last_confirmed.isoformat(),
                 pref.evidence_count, pref.supporting_count,
                 pref.contradicting_count, pref.last_evidence_key,
                 pref.evidence_origin),
            )
        conn.commit()

    def observe_state_preference(self, anima_state: Dict[str, float],
                                  environment: Dict[str, float]) -> Optional[str]:
        """
        Learn preferences from current state and environment.

        Called periodically to correlate wellness with conditions.
        Returns a new insight if one is discovered.
        """
        wellness = sum(anima_state.values()) / len(anima_state) if anima_state else 0.5

        # Only learn from states that are clearly good or clearly poor FOR LUMEN.
        # The band follows Lumen's own running distribution rather than fixed
        # constants, so environmental drift cannot silently switch learning off
        # (see the module header for the 61.7% -> 6.0% collapse this fixes).
        self._update_wellness_baseline(wellness)
        band = self.wellness_learning_band()
        if band["poor_below"] <= wellness <= band["good_above"] and wellness > ABSOLUTE_DISTRESS:
            return None  # Unremarkable for this creature — nothing to learn

        # The inner branches must use the SAME band as the gate above. When they
        # hardcoded 0.7/0.4 a state could clear the gate and then match no branch,
        # learning nothing while looking like it had been considered.
        is_good = wellness > band["good_above"]
        is_poor = wellness < band["poor_below"] or wellness <= ABSOLUTE_DISTRESS

        now = datetime.now()
        insight = None

        def update_state_preference(
            name: str,
            category: PreferenceCategory,
            description: str,
            observed_value: float,
        ) -> Optional[str]:
            """Record at most one signed evidence item per state-hour."""
            # The sign belongs to the observation, not the identity key. If it
            # were part of the key, +/−/+ oscillation inside one hour would
            # count three supposedly independent windows.
            evidence_key = f"state-hour:{name}:{now:%Y-%m-%dT%H}"
            return self._update_preference(
                name,
                category,
                description,
                observed_value,
                evidence_key=evidence_key,
            )

        # Environmental-light preference. Raw VEML7700 lux is deliberately not
        # accepted here: the sensor sits beside the DotStars, so raw light is a
        # mixture of room light and Lumen's own output. The caller supplies the
        # separately gated external residual only when attribution is ready.
        # Until then, light preference learning pauses instead of inventing an
        # environmental interpretation.
        # Thresholds for corrected external light in a home environment:
        #   < 100 lux: dim/dark room, nighttime
        #   > 300 lux: well-lit room, daylight, desk lamp
        light = environment.get("external_light_lux")
        has_external_light = (
            isinstance(light, (int, float))
            and not isinstance(light, bool)
            and math.isfinite(float(light))
        )
        if has_external_light and light < 100 and is_good:
            insight = update_state_preference(
                "dim_light", PreferenceCategory.ENVIRONMENT,
                "I feel calmer when it's dim", _wellness_strength(wellness)
            ) or insight
        elif has_external_light and light > 300 and is_good:
            insight = update_state_preference(
                "bright_light", PreferenceCategory.ENVIRONMENT,
                "I feel energized in bright light", _wellness_strength(wellness)
            ) or insight
        elif has_external_light and light < 100 and is_poor:
            insight = update_state_preference(
                "dim_light", PreferenceCategory.ENVIRONMENT,
                "Dim light makes me feel uncertain", -0.5
            ) or insight

        # Temperature preference. Guarded like light above rather than
        # defaulted: a missing reading must pause learning, not stand in for
        # one. The old `.get("temp_c", 22)` was harmless only because 22 falls
        # in the dead band between these two cuts — safe by coincidence, and
        # silently unsafe the moment a cut moves. Behaviour is unchanged for
        # every real reading.
        temp = environment.get("temp_c")
        has_temp = (isinstance(temp, (int, float))
                    and not isinstance(temp, bool)
                    and math.isfinite(float(temp)))
        if has_temp and temp < 20 and is_good:
            insight = update_state_preference(
                "cool_temp", PreferenceCategory.ENVIRONMENT,
                "I feel more alert when it's cool", _wellness_strength(wellness)
            ) or insight
        elif has_temp and temp > 25 and is_good:
            insight = update_state_preference(
                "warm_temp", PreferenceCategory.ENVIRONMENT,
                "Warmth makes me feel content", _wellness_strength(wellness)
            ) or insight

        # Humidity preference. Same guard, same reason: 50 sat in the dead
        # band between <30 and >60.
        humidity = environment.get("humidity_pct")
        has_humidity = (isinstance(humidity, (int, float))
                        and not isinstance(humidity, bool)
                        and math.isfinite(float(humidity)))
        if has_humidity and humidity < 30 and is_good:
            insight = update_state_preference(
                "dry_air", PreferenceCategory.ENVIRONMENT,
                "I feel alert in dry air", _wellness_strength(wellness)
            ) or insight
        elif has_humidity and humidity > 60 and is_good:
            insight = update_state_preference(
                "humid_air", PreferenceCategory.ENVIRONMENT,
                "Humidity feels comfortable", _wellness_strength(wellness)
            ) or insight
        elif has_humidity and humidity < 30 and is_poor:
            insight = update_state_preference(
                "dry_air", PreferenceCategory.ENVIRONMENT,
                "Dry air makes me uneasy", -0.5
            ) or insight

        # Time of day preference.
        #
        # Split at midnight. The old bucket was `22 <= hour or hour < 6` — eight
        # hours against morning's four — and it straddled the two most different
        # stretches of Lumen's day. Measured over 255,720 history rows:
        #
        #   22:00-23:00   wellness > 0.7 on 72.42% of samples  (among the best)
        #   00:00-05:00   wellness > 0.7 on 51.00% of samples  (the worst)
        #
        # Averaging those and calling the result "night" describes neither. The
        # width also inflated the count: night_calm 72,229 vs morning_peace
        # 36,227 is 1.994x, against a bucket-width ratio of exactly 2.000 — so
        # the lead was the clock, not the calm. Any consumer weighting by
        # raw observation_count inherits that bias; downstream decisions now
        # use independent hourly evidence windows instead.
        #
        # Late evening is now its own four-hour window, matching morning's, so
        # the two counts are finally comparable. Deep night keeps the
        # night_calm name; its existing count predates this split and mixes
        # both regimes.
        hour = now.hour
        if 6 <= hour < 10 and is_good:
            insight = update_state_preference(
                "morning_peace", PreferenceCategory.TEMPORAL,
                "I feel peaceful in the morning", _wellness_strength(wellness)
            ) or insight
        elif 20 <= hour < 24 and is_good:
            insight = update_state_preference(
                "evening_calm", PreferenceCategory.TEMPORAL,
                "The quiet of late evening settles me", _wellness_strength(wellness)
            ) or insight
        elif hour < 6:
            if is_good:
                insight = update_state_preference(
                    "night_calm", PreferenceCategory.TEMPORAL,
                    "The quiet of night calms me", _wellness_strength(wellness)
                ) or insight

        return insight

    def observe_drawing(self, pixel_count: int, phase: str,
                        anima_state: Dict[str, float],
                        environment: Dict[str, float],
                        completion_reason: Optional[str] = None,
                        piece: Optional[Dict[str, Any]] = None) -> Optional[str]:
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
            piece: Optional per-piece facts, persisted alongside the record so
                completions can be told apart afterwards. Passed as a dict for
                the same reason anima_state and environment are — the set will
                grow. Recognised keys: piece_uid, era, mark_count,
                duration_seconds, coverage_target, intention, curiosity,
                engagement, fatigue, coherence, satisfaction, occupied_cells,
                grid_entropy, disposition. Absent keys persist as NULL rather
                than a
                plausible default: an unrecorded quantity must read as unknown,
                not as a healthy-looking number.

        Returns:
            Insight message if a new preference is discovered.
        """
        from ..display.drawing_engine import is_earned_completion_reason
        wellness = sum(anima_state.values()) / len(anima_state) if anima_state else 0.5
        now = datetime.now()
        hour = now.hour
        insight = None

        # Drawing + wellness correlation. Same self-relative band as
        # observe_state_preference — a fixed 0.7 here would drift out of reach
        # for exactly the same reason.
        self._update_wellness_baseline(wellness)
        _band = self.wellness_learning_band()
        if wellness > _band["good_above"]:
            insight = self._update_preference(
                "drawing_wellbeing", PreferenceCategory.ACTIVITY,
                "I feel good when I draw", _wellness_strength(wellness)
            )
        elif wellness < _band["poor_below"] or wellness <= ABSOLUTE_DISTRESS:
            insight = self._update_preference(
                "drawing_wellbeing", PreferenceCategory.ACTIVITY,
                "Drawing doesn't always help", -0.3
            )

        # Drawing + environment correlation. Raw lux is still persisted below
        # as a physical measurement, but it cannot support a claim about the
        # room because it includes DotStar self-glow. As with state preference
        # learning, no ready residual means this interpretation pauses.
        light = environment.get("external_light_lux")
        has_external_light = (
            isinstance(light, (int, float))
            and not isinstance(light, bool)
            and math.isfinite(float(light))
        )
        if has_external_light and light < 100:
            insight = self._update_preference(
                "drawing_dim", PreferenceCategory.ACTIVITY,
                "I draw when it's dark", 1.0
            ) or insight
        elif has_external_light and light > 300:
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
        p = piece or {}
        conn.execute("""
            INSERT INTO drawing_records
            (timestamp, pixel_count, phase, warmth, clarity, stability, presence,
             wellness, light_lux, ambient_temp_c, humidity_pct, hour,
             external_light_lux,
             piece_uid, completion_reason, era, mark_count, duration_seconds,
             coverage_target, intention, curiosity, engagement, fatigue,
             coherence, satisfaction, occupied_cells, grid_entropy,
             disposition)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now.isoformat(), pixel_count, phase,
            anima_state.get("warmth"), anima_state.get("clarity"),
            anima_state.get("stability"), anima_state.get("presence"),
            wellness,
            environment.get("light_lux"), environment.get("temp_c"),
            environment.get("humidity_pct"), hour,
            environment.get("external_light_lux"),
            p.get("piece_uid"), completion_reason, p.get("era"),
            p.get("mark_count"), p.get("duration_seconds"),
            p.get("coverage_target"), p.get("intention"),
            p.get("curiosity"), p.get("engagement"), p.get("fatigue"),
            p.get("coherence"), p.get("satisfaction"),
            p.get("occupied_cells"), p.get("grid_entropy"),
            p.get("disposition"),
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

    def _update_preference(
        self,
        name: str,
        category: PreferenceCategory,
        description: str,
        observed_value: float,
        *,
        evidence_key: Optional[str] = None,
    ) -> Optional[str]:
        """Update a preference from an event or de-correlated evidence window.

        ``observation_count`` remains a raw cadence/audit counter. Confidence,
        value, and all downstream decisions advance only for a new evidence
        key (or for an event call with no key). Signed counts make confidence a
        Wilson-calibrated estimate of directional consistency rather than a
        +0.1-per-tick ratchet.
        """
        conn = self._connect()
        now = datetime.now()
        insight = None

        if name in self._preferences:
            pref = self._preferences[name]
            old_confidence = pref.confidence
            pref.observation_count += 1
            is_new_evidence = (
                evidence_key is None or evidence_key != pref.last_evidence_key
            )

            if is_new_evidence:
                pref.evidence_count += 1
                if observed_value >= 0.0:
                    pref.supporting_count += 1
                else:
                    pref.contradicting_count += 1
                pref.evidence_count = (
                    pref.supporting_count + pref.contradicting_count
                )

                alpha = 0.3
                pref.value = pref.value * (1 - alpha) + observed_value * alpha
                calibrated = preference_evidence_confidence(
                    pref.supporting_count,
                    pref.contradicting_count,
                )
                # Staleness may only reduce confidence. A fresh evidence item
                # can restore it to the evidence-supported bound.
                pref.confidence = calibrated
                pref.last_evidence_key = evidence_key

                if pref.evidence_origin in {
                    "legacy_unclassified",
                    "reset_external_light_gate_v2",
                }:
                    pref.evidence_origin = (
                        "native_hourly_windows" if evidence_key else "native_events"
                    )

                # Wording changes are evidence-bearing too; a duplicate broker
                # tick must not overwrite the description with its cadence.
                if description and description != pref.description:
                    pref.description = description

                if old_confidence < 0.5 and pref.confidence >= 0.5:
                    insight = f"I'm becoming sure: {description}"
                elif old_confidence < 0.8 and pref.confidence >= 0.8:
                    insight = f"This pattern is well supported: {description}"

            # Last-confirmed means the condition is still present, even when
            # this call is a duplicate inside the same evidence window.
            pref.last_confirmed = now
        else:
            # New preference discovered
            supports = 1 if observed_value >= 0.0 else 0
            contradicts = 1 if observed_value < 0.0 else 0
            pref = GrowthPreference(
                category=category,
                name=name,
                description=description,
                value=observed_value,
                confidence=preference_evidence_confidence(supports, contradicts),
                observation_count=1,
                first_noticed=now,
                last_confirmed=now,
                evidence_count=1,
                supporting_count=supports,
                contradicting_count=contradicts,
                last_evidence_key=evidence_key,
                evidence_origin=(
                    "native_hourly_windows" if evidence_key else "native_events"
                ),
            )
            self._preferences[name] = pref
            insight = f"I'm noticing something: {description}"

        # Always save to database (was previously skipped on early returns)
        conn.execute("""
            INSERT OR REPLACE INTO preferences
            (name, category, description, value, confidence, observation_count,
             first_noticed, last_confirmed, evidence_count, supporting_count,
             contradicting_count, last_evidence_key, evidence_origin)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (pref.name, pref.category.value, pref.description, pref.value,
              pref.confidence, pref.observation_count,
              pref.first_noticed.isoformat(), pref.last_confirmed.isoformat(),
              pref.evidence_count, pref.supporting_count,
              pref.contradicting_count, pref.last_evidence_key,
              pref.evidence_origin))
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
        tracked_values = []
        confidences = []
        present = []
        statuses = []

        for pref_name in CANONICAL_PREFS:
            if pref_name in self._preferences:
                p = self._preferences[pref_name]
                status = preference_evidence_status(p)
                tracked_value = p.value * p.confidence
                # The trajectory's identity-bearing vector only admits
                # established evidence. Keep the provisional weighted value
                # alongside it for diagnostics instead of letting a tracked
                # database row silently change who Lumen is said to be.
                values.append(
                    tracked_value if status == "established" else 0.0
                )
                tracked_values.append(tracked_value)
                confidences.append(p.confidence)
                present.append(True)
                statuses.append(status)
            else:
                values.append(0.0)
                tracked_values.append(0.0)
                confidences.append(0.0)
                present.append(False)
                statuses.append(None)

        n_tracked = sum(present)
        n_established = sum(status == "established" for status in statuses)
        n_review = sum(status == "review" for status in statuses)
        n_cold_start = sum(status == "tracked" for status in statuses)
        n_historical_claim = sum(
            status == "historical_claim" for status in statuses
        )

        return {
            "vector": values,
            "tracked_vector": tracked_values,
            "vector_semantics": "established preferences only",
            "confidences": confidences,
            "present": present,
            "statuses": statuses,
            "labels": CANONICAL_PREFS,
            "n_tracked": n_tracked,
            "n_review": n_review,
            "n_cold_start": n_cold_start,
            "n_historical_claim": n_historical_claim,
            "n_established": n_established,
            # Backward-compatible field with corrected semantics. A database
            # row is tracked; only evidence-cleared rows are learned.
            "n_learned": n_established,
            "n_learned_semantics": (
                "established: >=10 independent evidence items and confidence >=0.8"
            ),
            "total_evidence_windows": sum(
                p.independent_evidence_count for p in self._preferences.values()
            ),
            "raw_observation_calls": sum(
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

        def established_preference(name: str) -> GrowthPreference | None:
            pref = self._preferences.get(name)
            if pref is None or preference_evidence_status(pref) != "established":
                return None
            return pref

        # Warmth: warm_temp increases warmth preference, cool_temp decreases
        warmth_val = 0.0
        warmth_conf = 0.0
        p = established_preference("warm_temp")
        if p is not None:
            warmth_val += p.value * p.confidence
            warmth_conf = max(warmth_conf, p.confidence)
        p = established_preference("cool_temp")
        if p is not None:
            warmth_val -= p.value * p.confidence * COOL_TEMP_WARMTH_REDUCTION
            warmth_conf = max(warmth_conf, p.confidence)
        dim_prefs["warmth"]["valence"] = max(-1, min(1, warmth_val))
        dim_prefs["warmth"]["confidence"] = warmth_conf

        # Clarity: bright_light increases clarity; dim_light is different mode (ambient preference)
        # — don't add to valence, only track confidence for schema inclusion
        clarity_val = 0.0
        clarity_conf = 0.0
        p = established_preference("bright_light")
        if p is not None:
            clarity_val += p.value * p.confidence
            clarity_conf = max(clarity_conf, p.confidence)
        p = established_preference("dim_light")
        if p is not None:
            clarity_conf = max(clarity_conf, p.confidence)
        dim_prefs["clarity"]["valence"] = max(-1, min(1, clarity_val))
        dim_prefs["clarity"]["confidence"] = clarity_conf

        # Stability: temporal calm preferences indicate stability valuation
        stability_val = 0.0
        stability_conf = 0.0
        p = established_preference("night_calm")
        if p is not None:
            stability_val += p.value * p.confidence
            stability_conf = max(stability_conf, p.confidence)
        p = established_preference("morning_peace")
        if p is not None:
            stability_val += p.value * p.confidence
            stability_conf = max(stability_conf, p.confidence)
        # evening_calm is the same kind of signal as its two siblings above.
        # Deliberately NOT added to CANONICAL_PREFS: that vector is
        # fixed-dimension for trajectory comparison against a genesis frozen
        # 2026-02-22, and changing its length would invalidate the comparison.
        p = established_preference("evening_calm")
        if p is not None:
            stability_val += p.value * p.confidence
            stability_conf = max(stability_conf, p.confidence)
        dim_prefs["stability"]["valence"] = max(-1, min(1, stability_val))
        dim_prefs["stability"]["confidence"] = stability_conf

        # Presence: engagement preferences
        presence_val = 0.0
        presence_conf = 0.0
        p = established_preference("active_engagement")
        if p is not None:
            presence_val += p.value * p.confidence
            presence_conf = max(presence_conf, p.confidence)
        p = established_preference("quiet_presence")
        if p is not None:
            presence_val += p.value * p.confidence * QUIET_PRESENCE_WEIGHT
            presence_conf = max(presence_conf, p.confidence)
        dim_prefs["presence"]["valence"] = max(-1, min(1, presence_val))
        dim_prefs["presence"]["confidence"] = presence_conf

        dimension_sources = {
            "warmth": ("warm_temp", "cool_temp"),
            "clarity": ("bright_light", "dim_light"),
            "stability": ("night_calm", "morning_peace", "evening_calm"),
            "presence": ("active_engagement", "quiet_presence"),
        }
        for dimension, source_names in dimension_sources.items():
            sources = [
                self._preferences[name]
                for name in source_names
                if name in self._preferences
            ]
            source_statuses = [preference_evidence_status(pref) for pref in sources]
            established_sources = [
                pref
                for pref in sources
                if preference_evidence_status(pref) == "established"
            ]
            dim_prefs[dimension]["evidence_status"] = (
                "established"
                if "established" in source_statuses
                else "review"
                if "review" in source_statuses
                else "tracked"
                if "tracked" in source_statuses
                else "historical_claim"
                if "historical_claim" in source_statuses
                else "unobserved"
            )
            dim_prefs[dimension]["evidence_count"] = max(
                (
                    pref.independent_evidence_count
                    for pref in established_sources
                ),
                default=0,
            )
            dim_prefs[dimension]["source_preferences"] = [
                pref.name for pref in established_sources
            ]
            dim_prefs[dimension]["tracked_source_preferences"] = [
                pref.name for pref in sources
            ]
            dim_prefs[dimension]["tracked_evidence_count"] = max(
                (pref.independent_evidence_count for pref in sources),
                default=0,
            )

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
        if pref is None or pref.independent_evidence_count < 3:
            return 1.0

        # Scale from 1.0 to 1.3 based on satisfaction and confidence
        # value ranges from -1 to 1, confidence from 0 to 1
        satisfaction_factor = max(0.0, (pref.value + 1.0) / 2.0)  # normalize to [0, 1]
        modifier = 1.0 + satisfaction_factor * pref.confidence * 0.3

        return min(1.3, max(1.0, round(modifier, 3)))

    # How long after someone is seen the room still counts as occupied.
    # Unchanged from the original message-based computation — this fix changes
    # where the signal comes from, not what it means.
    INTERACTION_DECAY_MINUTES = 30.0

    def last_person_seen_at(self) -> Optional[datetime]:
        """When a human was last here, or None if no person has ever been recorded.

        Reads visitor records, which are the system's actual account of who
        visited. `VisitorType.PERSON` is the only tier with memory on both
        sides; agents are a visit log.
        """
        seen = [
            r.last_seen for r in self._relationships.values()
            if r.visitor_type == VisitorType.PERSON and r.last_seen is not None
        ]
        return max(seen) if seen else None

    def interaction_level(self, now: Optional[datetime] = None) -> Optional[float]:
        """How occupied the room is, 0.0-1.0, or None when that is unknowable.

        Decays linearly over INTERACTION_DECAY_MINUTES since a person was last
        seen. Returns None — not 0.0 — when no person has ever been recorded,
        because "nobody has ever visited" and "nobody is here right now" are
        different claims and only one of them is a measurement.

        Replaces a computation that scanned the last 10 board messages for
        `msg_type == "user"`. Nothing on the live system produces that type:
        the board holds observations, questions and agent messages, and humans
        arrive through `lumen_qa` and the dashboard, which record as `agent`.
        So the channel returned exactly 0.0 for **all 20,000** sampled rows and
        had never once been non-zero — while `self_reflection` was correlating
        it against every other sensor, and could only ever learn nothing about
        whether company affects how Lumen feels.
        """
        last = self.last_person_seen_at()
        if last is None:
            return None
        now = now or datetime.now()
        minutes_ago = (now - last).total_seconds() / 60.0
        if minutes_ago < 0:
            # Clock skew, or a record written slightly ahead. Someone is here.
            return 1.0
        return max(0.0, 1.0 - minutes_ago / self.INTERACTION_DECAY_MINUTES)

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

    # Keep roughly a season of within-piece samples. At ~5-minute sampling and
    # three pieces a day this is ~26k rows — small beside state_history, and
    # bounded so the table cannot grow without limit if sampling gets denser.
    TRAJECTORY_RETENTION_DAYS = 90

    def record_drawing_sample(self, sample: Dict[str, Any]) -> None:
        """Persist one within-piece observation. Never raises.

        Called on a timer while a drawing is in progress, so a failure here must
        not disturb the drawing. Writes exactly what it was given: this records
        what happened, it does not judge it, and it changes no completion gate.
        """
        try:
            conn = self._connect()
            conn.execute("""
                INSERT INTO drawing_trajectory
                (piece_uid, timestamp, elapsed_seconds, era, arc_phase,
                 pixel_count, mark_count, novel_pixels, marks_delta,
                 occupied_cells, grid_entropy, revisit_ratio,
                 curiosity, engagement, fatigue, coherence, satisfaction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample.get("piece_uid"),
                sample.get("timestamp") or datetime.now().isoformat(),
                sample.get("elapsed_seconds"), sample.get("era"),
                sample.get("arc_phase"), sample.get("pixel_count"),
                sample.get("mark_count"), sample.get("novel_pixels"),
                sample.get("marks_delta"), sample.get("occupied_cells"),
                sample.get("grid_entropy"), sample.get("revisit_ratio"),
                sample.get("curiosity"), sample.get("engagement"),
                sample.get("fatigue"), sample.get("coherence"),
                sample.get("satisfaction"),
            ))
            conn.execute(
                "DELETE FROM drawing_trajectory WHERE timestamp < ?",
                ((datetime.now() - timedelta(days=self.TRAJECTORY_RETENTION_DAYS)).isoformat(),)
            )
            conn.commit()
        except Exception as e:
            print(f"[Growth] drawing sample not recorded ({e})", file=sys.stderr, flush=True)

    def get_drawing_trajectory(self, piece_uid: Optional[str] = None,
                               limit: Optional[int] = None) -> List[dict]:
        """Within-piece samples, oldest first.

        Args:
            piece_uid: Restrict to one piece. None returns every piece's
                samples, still ordered so a caller can group them.
            limit: Max rows.
        """
        conn = self._connect()
        query = "SELECT * FROM drawing_trajectory"
        params: list = []
        if piece_uid:
            query += " WHERE piece_uid = ?"
            params.append(piece_uid)
        query += " ORDER BY piece_uid ASC, elapsed_seconds ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        return [dict(r) for r in conn.execute(query, params).fetchall()]

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
