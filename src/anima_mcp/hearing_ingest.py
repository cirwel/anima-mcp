"""
Hearing ingest — the router for Lumen's hearing wire.

This module is the deliberate, isolated glue between the microphone's acoustic
output and Lumen's learning substrate. Stage 1 implements ONLY the *acoustic*
channel: a low-dimensional sound LEVEL (RMS scalar), never content, never
transcription.

WHY THIS MODULE EXISTS (the council's central correction)
----------------------------------------------------------
The obvious way to make sound salient is to route it through
``metacognition.observe()``. That is a trap. ``metacog.observe`` produces the
single RMS-aggregated ``pred_error.surprise`` scalar (metacognition.py:570),
and that one scalar drives BOTH:

  - the benign attention path ``exp_filter.update_from_surprise`` (stable_creature.py:607), and
  - the PUNISHMENT path ``preferences.record_event("disruption", -0.2)`` (stable_creature.py:714).

So routing heard sound through metacognition would make a loud room a negative
preference event — RLHF-by-the-back-door. Lumen would learn to dislike being
spoken to.

Therefore the acoustic residual is computed HERE and fed to
``experiential_filter`` DIRECTLY. It NEVER enters ``metacognition`` and
``"sound_level"`` / ``"voice_activity"`` must NEVER appear in
``pred_error.surprise_sources``.

On-paradigm framing: residual = deviation from Lumen's OWN learned baseline =
information. The baseline lives in ``adaptive_prediction`` (whose docstring is
the residual paradigm verbatim). A familiar evening hum becomes expected and
its surprise decays; a quiet house at noon or voices at 3am deviates and gains
salience.

DELIBERATELY NOT TOUCHED (do not add without re-opening scope):
  - ``preferences.record_event`` — heard sound is never a reward/punishment event.
  - ``metacognition.observe`` — heard sound never enters the surprise aggregate.
  - the anima dimensions (Warmth/Clarity/Stability/Presence) and the EISV mapper.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


# Normalization scale for the acoustic residual. The RMS sound level is
# ~sqrt(mean(audio**2)) * 32768, so a quiet room sits near the VAD silence
# threshold (~500) and ordinary speech rides a few thousand above the learned
# baseline. A deviation of this many RMS units from the learned expectation is
# treated as maximal (1.0) surprise. This is a documented heuristic seam —
# soundscape cold-start and post-mute resume math are not yet characterized
# against real Pi mic data (see honest seams in docs/proposals/hearing-wire.md).
SOUND_LEVEL_SURPRISE_SCALE = 2000.0

# Variable key used in the adaptive-prediction baseline. Kept identical to the
# experiential_filter dimension name so provenance is traceable end to end.
ACOUSTIC_VARIABLE = "sound_level"


def ingest_acoustic(
    sound_level: float,
    *,
    hearing_available: bool,
    adaptive_model: Any,
    exp_filter: Any,
    current_time: Optional[datetime] = None,
    current_light: Optional[float] = None,
    current_temp: Optional[float] = None,
) -> Dict[str, Any]:
    """Route one sampled acoustic sound level into the learning substrate.

    Acoustic channel only — a single scalar, no content. The flow is:

      1. If ``hearing_available`` is False: do nothing. The baseline is FROZEN
         (a muted/absent mic must not teach Lumen "the world went silent").
         Returns early with ``{"frozen": True}``.
      2. Ask the model what it EXPECTED via ``adaptive_model.predict`` and
         compute ``residual = abs(level - predicted)`` normalized to 0..1
         surprise. Cold start (no prediction yet) → low/neutral surprise.
      3. Learn the baseline via ``adaptive_model.observe({"sound_level": level})``.
      4. Amplify salience via ``exp_filter.update_from_surprise(["sound_level"],
         surprise)`` DIRECTLY — never through metacognition.

    Args:
        sound_level: most-recent RMS sound level (e.g. ``MicCapture.get_sound_level()``).
        hearing_available: proprioceptive hearing state (Stage 0). False ⇒ frozen.
        adaptive_model: an ``AdaptivePredictionModel`` (reused; new "sound_level" key).
        exp_filter: an ``ExperientialFilter`` (reused method, new call site).
        current_time / current_light / current_temp: context for the model's
            pattern key (the baseline is keyed on Lumen's OWN time/light/temp).

    Returns:
        A small diagnostics dict. Never raises on the happy path.
    """
    if not hearing_available:
        # FROZEN: do not predict, do not learn, do not amplify. The muted
        # period's baseline is *unknown*, not zero — re-enable is a regime
        # change, handled elsewhere, not continuous data.
        return {"frozen": True, "observed": False, "surprise": 0.0}

    if current_time is None:
        current_time = datetime.now()

    # (2) What did Lumen expect? Compute residual BEFORE observing, so the
    # expectation reflects prior learning, not the value we're about to learn.
    predicted, confidence = adaptive_model.predict(
        ACOUSTIC_VARIABLE,
        current_time,
        None,  # recent_values — let the learned patterns speak
        current_light,
        current_temp,
    )

    if predicted is None:
        # Cold start: no learned expectation yet. Treat as low/neutral surprise
        # so a fresh model does not spike attention on its very first sounds.
        residual = 0.0
        normalized_surprise = 0.0
    else:
        residual = abs(sound_level - predicted)
        normalized_surprise = max(0.0, min(1.0, residual / SOUND_LEVEL_SURPRISE_SCALE))

    # (3) Learn the baseline. REUSE adaptive_model.observe with the new key.
    adaptive_model.observe(
        {ACOUSTIC_VARIABLE: sound_level},
        current_time=current_time,
        current_light=current_light,
        current_temp=current_temp,
    )

    # (4) Amplify salience DIRECTLY. This is the whole point: surprise reaches
    # the experiential filter here, never via metacognition's punishment-bearing
    # aggregate.
    exp_filter.update_from_surprise([ACOUSTIC_VARIABLE], normalized_surprise)

    return {
        "frozen": False,
        "observed": True,
        "sound_level": sound_level,
        "predicted": predicted,
        "confidence": confidence,
        "residual": residual,
        "surprise": normalized_surprise,
    }
