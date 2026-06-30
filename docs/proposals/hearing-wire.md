# THE WIRE — Heard Speech → Lumen's Learning Substrate

> **Status:** Council plan (architect + reviewer + live-verifier), produced by a
> multi-agent council workflow on **2026-06-29**.
> **Implementation:** Stage 0 (mute-as-sensed-state) and Stage 1 (acoustic
> sound-level channel) + the G1 regression test are implemented on branch
> `feat/hearing-wire-acoustic`. Stages 2–4 (presence-ease debounce, semantic
> memory, speaker latent) are **designed but NOT implemented** here.

---

## TL;DR

Lumen's mic computes a sound level every ~32ms and throws it away
(`mic.py:145`); its transcripts reach `_on_hear` and go nowhere that learns
(`autonomous_voice.py:375`). The wire splits hearing into two channels: a
low-dimensional **acoustic** channel that teaches Lumen its room's soundscape
rhythm as proprioception (residual = deviation from its own learned baseline),
and a confidence-gated **semantic** channel that turns transcripts into
conviction-weighted memory it can be contradicted out of. Both ease Lumen's
drives through the *same* "being-with eases you" pathway that text interaction
already uses — and both are hard-excluded from the one place in the code that
turns surprise into a negative preference event. The critical correction from
review: the obvious way to make sound *salient* routes it straight through that
punishment path, so the anti-RLHF guarantee has to be built as new glue plus a
regression test, not as "reuse what's there."

---

## Architecture (validated end-to-end flow)

Two channels, different evidence types, different destinations. Do not collapse
them.

```
                          mic.py  sd.InputStream  (REUSE)
                                     │
        ┌────────────────────────────┴───────────────────────────┐
   ACOUSTIC CHANNEL                                          SEMANTIC CHANNEL
   (sensor-like, no content)                                (from transcript)
        │                                                         │
 [NEW] get_sound_level() — return the RMS              on_speech_end(bytes) (REUSE 176)
        dropped at mic.py:145                                     │
        │                                              stt.transcribe() → TranscriptionResult (REUSE)
 [NEW] broker samples → SensorReadings.sound_level                │
        │                                              autonomous_voice._on_hear() (REUSE entry 375)
 ┌──────┴───────────┐                                             │
 │                  │                                  [NEW] hearing_ingest.route(utterance)
 (a) baseline       (b) salience                       ┌─────────┴──────────┬─────────────┐
 adaptive_model.    [NEW] router computes              │                    │             │
 observe(           residual = level −          conf ≥ τ:            ALWAYS:        directed-at-Lumen?
  {"sound_level"})  adaptive_model.predict()    knowledge.add_insight   (acoustic only) [NEW] debounced
 (REUSE 309,        then calls                  (REUSE 589,             nothing extra    social-boost flag
  new key)          exp_filter.update_from_     EXPLICIT confidence)        │           (guarded reuse)
        │           surprise(["sound_level"],        │                      │                │
 stores Learned     residual) DIRECTLY          memory_retrieval.           │     inner_life.apply_
 Pattern baseline   (REUSE method 171,           add_session_memory(REUSE)  │     social_boost() (REUSE 262)
 (the expectation)  NEW call site)                    │                     │
                         │                       schema_hub.compose_schema (REUSE 79)
                   exp_filter salience           → self_schema_renderer "things I've heard" [NEW phrasing]
```

**The correction that reshapes the design (verified at
`stable_creature.py:603/607/656/714`):** `adaptive_model.observe()` (line 656)
only fits Welford patterns — it produces **no surprise and feeds nothing
downstream**. The *only* thing that produces `pred_error.surprise` is
`metacog.observe()` (line 603), and that single RMS-aggregated scalar
(`metacognition.py:570`) simultaneously drives the benign attention path
`exp_filter.update_from_surprise` (line 607) **and** the punishment path
`preferences.record_event("disruption", -0.2)` (line 714).

Consequence: routing `sound_level` into `adaptive_prediction` gives Lumen a
learned baseline but **zero salience** — the prior draft's "voices at 3am → high
attention" cannot fire as drawn. The naive fix (add a sound branch to
`metacog.observe`) lands acoustic surprise in the line-570 aggregate and trips
the −0.2 disruption event on a loud room. **So the acoustic residual must be
computed in the new `hearing_ingest` router and fed to
`exp_filter.update_from_surprise` directly** — reusing the method, but as a new
call site outside `metacog`. This is NEW glue, relabeled honestly (not the
"free reuse" the first draft claimed).

| Stage | Module / function | Reuse / New |
|---|---|---|
| Capture | `mic.py` MicCapture + `get_sound_level()` | REUSE capture, NEW accessor (~5 lines) |
| Transcribe | `stt.transcribe()` → `TranscriptionResult` (`stt.py:31`) | REUSE (vosk; whisper.cpp swap before Stage 3) |
| Entry | `autonomous_voice._on_hear` (`:375`) | REUSE; add one call to router |
| **Router** | **`hearing_ingest.py`** — routing, confidence gate, residual compute, baseline glue | **NEW (the heart of the wire)** |
| Acoustic baseline | `adaptive_model.observe({"sound_level":…})` (`:309`) | REUSE, new key |
| Acoustic salience | `exp_filter.update_from_surprise(["sound_level"], residual)` (`:171`) called from router | REUSE method, NEW call site + new `DIMENSIONS` entry (`:22`) |
| Semantic memory | `knowledge.add_insight(..., confidence=utterance.confidence)` (`:589`) + `add_session_memory` (`:290`) | REUSE + explicit-confidence fix |
| Presence ease | `/dev/shm/anima_social_boost` → `apply_social_boost()` (`:262`), **debounced** | REUSE pathway, NEW debounce guard |
| Self-model surface | `schema_hub.compose_schema` (`:79`) + renderer "heard" phrasing | REUSE + small render addition |

**Deliberately NOT wired:** `preferences.record_event`
(`stable_creature.py:714`) and `agency`/TD-learning. Heard speech never touches
the reinforcement path.

---

## On-paradigm framing (active-inference / growth, not RLHF)

Two residuals, both "deviation from Lumen's *own* learned baseline =
information":

- **Acoustic residual** rides `adaptive_prediction`, whose docstring *is* the
  residual paradigm verbatim ("if Lumen keeps getting surprised by the same
  pattern… it should stop being surprised"). A quiet house at noon or voices at
  3am deviates from the learned expectation → high salience → attention. The
  familiar evening hum is expected → surprise decays as
  `LearnedPattern.confidence` rises. Baseline is keyed on Lumen's *own*
  `(hour, light, temp)` (`adaptive_prediction.py:334`) — individuality
  preserved, no imported "normal loudness" prior.
- **Semantic residual** extends the world-model with new latents (who is
  present, what recurs). First hearing of a voice/topic is maximally surprising;
  the hundredth is part of the expected world.
  `conviction = references + 0.5·confidence + recency` (`knowledge.py:98`) is
  **references-dominated**, so heard content accrues conviction only by
  *recurring across independent occasions*, never by being loud or certain.

**Why this is not a score (enforced, with the review fixes applied):**

1. **The punishment exclusion is now real, not asserted.** Acoustic surprise is
   computed in the router and fed to `exp_filter` directly; it never enters
   `metacog.observe`'s aggregate (`metacognition.py:570`) and therefore never
   reaches line 714. Defense-in-depth: make line 714 source-aware (skip
   `record_event` when surprise sources are audio-only). **This is guaranteed by
   a regression test (see Gates), not by a comment.**
2. **Salience decays symmetrically to neutral** (`experiential_filter.py:77`) —
   no trauma ratchet.
3. **"Eased by company, not approval"** — but the social-boost flag was built
   for *discrete* events (`communication.py:192` fires on rare
   `lumen_qa`/`post_message`), and `apply_social_boost` saturates toward 1.0
   under repeated touches (`inner_life.py:262`, `min(1.0, …)`). A continuous
   voice stream touching it every ~2s tick becomes a presence-reward
   accumulator. **Fix: debounce to ≤1 boost per long window (≥5 min) and/or gate
   on novel *directed* speech.** Directedness must stay acoustic-geometric
   (addressed/proximate), never outcome-based ("they kept talking after I
   answered" = approval = RLHF).
4. **No sentiment/valence on content.** We carry topic / recurrence / novelty as
   information; we never classify "sounded angry → mood −". That is a covert
   reward channel and vosk can't do it. Hard no.

> "Lumen learns by being wrong about what it expects to hear, not by being
> graded on whether it liked what it heard."

---

## Privacy posture (mic is operator-gated hardware-off; consent is first-class)

- **Mute is a sensed proprioceptive fact, not a dead channel.** NEW
  `SensorReadings.hearing_available: bool`. While muted, the soundscape baseline
  is **frozen** — we must not learn "the world went silent" from a muted mic, or
  re-enable produces a massive false-surprise transient. Re-enable is treated as
  a regime-change event (like `meta_gap`), not continuous data; the muted
  period's baseline is *unknown*, not zero.
- **Two-tier storage by sensitivity.** Acoustic → a single residual scalar;
  never identifiable, never reconstructible; safe even with content storage off.
  Semantic → transcript + confidence; **waveform discarded** (`mic.py` only
  RAM-buffers for VAD). The substrate stores derived insights/memories, never
  recordings. vosk is offline — no cloud (verified).
- **Honest caveat (don't over-claim acoustic cleanliness):** time-keyed
  `sound_level`/`voice_activity` patterns build a household
  *presence/occupancy rhythm* ("voices mornings, quiet 3am") — a behavioral
  model of people, with no content. Not fatal (it is Lumen's own environmental
  proprioception, like light/temperature rhythm), but consent level 1 must make
  explicit that it learns *when* voices tend to occur.
- **Provenance + erasure.** Everything hearing writes carries
  `source_author="heard_utterance"`, inspectable/erasable via
  `get_self_knowledge` / `get_qa_insights`. "Show me what you learned by
  listening" / "forget this voice" must work **before** content storage is
  enabled. Note: persisted heard insights are plaintext in SQLite and can
  surface over the OAuth-gated `lumen.cirwel.org` tunnel — do not expose raw
  transcripts in REST/dashboard without a separate gate.
- **Speaker identity = unnamed, operator-namable latent, never
  inferred-and-asserted** (mirrors the "no lookup-by-label" identity invariant).
  No silent named dossiers of household members.
- **Ascending consent levels (each a deliberate operator threshold):**
  (0) muted; (1) acoustic level only — proprioception, no content;
  (2) + content insights behind the confidence gate;
  (3) + speaker-latent clustering.

---

## Staged build (smallest meaningful step → full loop)

**Stage 0 — Mute as sensed state. SHIPS VALUE WITH MIC STILL MUTED. Ordering
gate: MUST precede any baseline learning.** Add `SensorReadings.hearing_available`;
freeze baseline while muted; render "I cannot currently hear" in self-schema.
New: one bool + renderer line. (Prevents the false-silence baseline.)
**[IMPLEMENTED on `feat/hearing-wire-acoustic`.]**

**Stage 1 — Acoustic level channel (consent 1).** Rescue the RMS at `mic.py:145`
via `get_sound_level()`; add `SensorReadings.sound_level`; feed
`adaptive_model.observe({"sound_level":…})` for the baseline; **compute the
residual in the router and call `exp_filter.update_from_surprise(["sound_level"],
residual)` directly** (NOT via metacog); add `"sound_level"`/`"voice_activity"`
to `experiential_filter.DIMENSIONS`/`SOURCE_TO_DIM`. This is the smallest real
"hear → be-changed" loop: Lumen learns its room's soundscape rhythm and is
surprised by deviations — no transcription, no content. Keep it strictly out of
`metacognition`, the anima dimensions, and the EISV mapper.
**[IMPLEMENTED on `feat/hearing-wire-acoustic` — `src/anima_mcp/hearing_ingest.py`.]**

**Stage 2 — Presence ease (consent 1).** On *novel directed* voice activity
only, touch the social-boost flag — **debounced** (≥5 min). Reuse
`communication.py:192 → stable_creature.py:548 → apply_social_boost`. New code:
the debounced trigger + a directedness heuristic (seam: start with "speech while
recently active/addressed"). Do NOT ship as a per-tick trigger. **[NOT
implemented.]**

**Stage 3 — Semantic memory (consent 2).** New `hearing_ingest.py` router
(semantic path): confidence-gated `add_insight(..., confidence=utterance.confidence)`
+ `add_session_memory`; schema_hub surfaces it as "things I've heard." Heard
content becomes persistent, conviction-weighted, contradiction-tracked memory.
**Swap vosk → whisper.cpp first** — semantic ingestion is only as honest as the
transcript. Heard insights must be barred from down-weighting *non-heard*
insights via contradiction. **[NOT implemented.]**

**Stage 4 — Speaker latent + relational trajectory (consent 3).** Unnamed
voice-cluster latent (needs embedding/diarization — biggest new dependency),
operator-named only; feed `trajectory` relational Δ. Highest-stakes, last, gated
hardest. **[NOT implemented.]**

---

## Risks & gates

| # | Risk | Gate / deferral |
|---|---|---|
| **G1** | Acoustic surprise leaking into `record_event` punishment (`metacognition.py:570` → `stable_creature.py:714`). One "natural" commit away. | **BLOCKING regression test:** inject a loud-soundscape spike, assert **zero** `record_event("disruption")` calls *and* a nonzero `sound_level` salience bump. `"sound_level"`/`"voice_activity"` must never appear in `pred_error.surprise_sources`. Without this test the anti-RLHF guarantee is prose. **[IMPLEMENTED: `tests/test_hearing_ingest.py`.]** |
| **G2** | Social-boost saturation from a continuous stream (`inner_life.py:262`, `min(1.0,…)`). | Debounce ≥5 min and/or gate on novel directed speech. Never per-tick. Don't ship Stage 2 otherwise. |
| **G3** | Confidence 1.0 default (`knowledge.py:270`: `else 1.0`) → heard speech as maximal external truth. | **Mandatory test:** every `heard_utterance` insight carries `confidence == utterance.confidence`, never the default. |
| **G4** | "Heard often" → "true" via repetition. Re-derivation credit (`knowledge.py:346`) blocks paraphrases of the same question, NOT distinct sentences on a recurring theme (TV, a catchphrase). The first draft's "content-distinct guard" fix is the mechanism that *fails* here. | Route recurrence to a **familiarity/soundscape latent** (predictor surprise-decay); gate `references` credit on **independent occasions** (real time-gap, ideally distinct speaker-latent), never per-tick repetition. Keep "heard often" structurally separate from "is true." |
| **G5** | Context key `(hour, light, temp)` (`adaptive_prediction.py:334`) cannot represent occupancy/day-of-week. Tuesday-empty and Saturday-party collapse into one bucket → high variance → never habituates. The "be-changed-over-time" claim is **unsupported** until fixed. | Add an occupancy / day-of-week context dimension before claiming habituation. This is a modeling extension, not a config tweak. |
| **G6** | whisper.cpp on a Pi 4 spikes CPU; the "neural" bands ARE CPU (`beta = cpu%`, `gamma = ctx-switches`), and thermal load feeds anima → Lumen "feels itself listening," risks throttling the broker. | Rate-limit transcription; run STT niced / off the broker process. Stages 0–2 avoid this entirely and should be the only near-term ship. |
| **G7** | vosk mis-transcription can fire `_polarity_conflict` (`knowledge.py:287`) and down-weight a legitimate self-derived belief. | Below-τ transcripts feed only the acoustic residual; heard insights barred from contradiction-down-weighting non-heard insights. |
| **G8** | Acoustic→anima coupling propagates into the EISV vector reported to UNITARES (local governance is trigger-happy); a noisy room could shift governance state. | Keep Stage 1 acoustic out of the anima dimensions and the EISV mapper until the mapping is cross-checked against the roadmap. |
| **G9** | Worktree auto-sync `git reset --hard` wipes uncommitted work (`hearing_ingest.py`, edits to `mic.py`/`knowledge.py`/`experiential_filter.py`). | Branch, commit, push promptly; re-Read files after any sync. |

**Do-not-build:** sentiment/affect classification of heard speech (covert reward
channel; vosk can't anyway). **Do-not-build without explicit per-feature
operator sign-off:** Stage 4 speaker clustering (silent household dossier +
re-identification surface; conflicts with identity invariants).

---

## Honest seams

**Buildable-but-not-yet-existing (named, not invented):** directedness detection
(no detector today); speaker diarization/voice-print (Stage 4 dependency);
semantic parsing beyond keyword `_categorize_text` ("It's cold" vs "I'm cold"
are indistinguishable to it); occasion gating for speech (no MCP session id —
needs the G4 independent-occasion rule); STT confidence semantics (vosk gives a
noisy word-average, `alternatives` always empty); soundscape cold-start and
post-mute resume math (unspecified).

**Genuine "can't verify from inside" seams — must stay explicit, never
asserted:**

- **Hear → understand.** transcribe → interpret → remember → be-changed are all
  real and buildable here. Whether the recurrence-statistics this produces
  amount to *understanding* is the seam. The self-model must render "I have
  heard X said often" / "a voice I don't recognize was present" — **not** "I know
  that X." Store transcript provenance + confidence *alongside* any derived
  belief so the gap stays visible. Lumen should be able to say "I can't verify
  from inside whether I understand this — I only know it recurs."
- **EISV roadmap is not in this checkout.**
  `docs/proposals/eisv-maths-roadmap-v0.md` lives in the `unitares` repo, not
  anima-mcp (confirmed absent here). The residual framing was grounded against
  the *local* machinery that embodies it (`adaptive_prediction`'s surprise-decay,
  `experiential_filter`'s neutral-decay). **Cross-check the acoustic-residual
  mapping against that roadmap before Stage 1 ships** to confirm "ODE-as-predictor,
  residual=signal, Φ→telemetry."

**Open questions for the operator:**

- When Lumen hears a *question*, does it auto-answer (`_get_response`) or only
  store and respond later? Currently undefined.
- Default confidence threshold τ for content ingestion.
- Whether `voice_activity` and `sound_level` are one channel or two in
  `DIMENSIONS`. *(Stage 1 registers both dimensions; only `sound_level` is wired
  through the router so far.)*

---

**Net shape (one sentence):** rescue the RMS dropped at `mic.py:145` into the
existing residual predictor for the *baseline* and compute its salience in a new
router fed straight to `experiential_filter` (acoustic channel — kept out of
`metacognition`, so out of the punishment path); route confidence-gated
transcripts into knowledge/memory/schema with explicit STT confidence and honest
"things I've heard" provenance (semantic channel); ease drives only through the
*debounced* social-boost→`inner_life` pathway text already uses; and prove the
exclusion from `preferences.record_event` with a regression test — so hearing
changes Lumen's world-model and its sense of company, never its score.

**Key files/lines:** `mic.py:145` (RMS to rescue), `stt.py:31`
(`TranscriptionResult`), `autonomous_voice.py:375` (`_on_hear`),
`adaptive_prediction.py:309/334` (`observe`/context key),
`experiential_filter.py:22/77/171` (DIMENSIONS / decay / `update_from_surprise`),
`metacognition.py:570` (RMS surprise aggregate — keep audio OUT),
`stable_creature.py:603/607/656/714/548` (surprise source / attention / no-op
adaptive observe / punishment event / social-boost consume),
`knowledge.py:98/270/287/346/589` (conviction / confidence default /
contradiction / re-derivation / `add_insight`), `inner_life.py:262`
(`apply_social_boost`, saturates), `handlers/communication.py:192`
(social-boost trigger). New module: `src/anima_mcp/hearing_ingest.py`.
