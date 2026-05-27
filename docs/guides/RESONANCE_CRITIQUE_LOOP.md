# Resonance Critique Loop

**Created:** May 27, 2026  
**Last Updated:** May 27, 2026  
**Status:** Active

---

## Purpose

The Resonance critique loop lets an operator or agent read Lumen's current drawing as a grounded visual trace before recommending any art-era change.

It is intentionally advisory. The loop may recommend `stay`, `tune`, or `switch`, but it must not switch eras, toggle auto-rotate, clear the canvas, or infer a fixed intent from the marks.

## Why this exists

The Resonance era feels most alive when it behaves like accumulated history rather than a style preset: sediment, flow, scratches, biological ornament, intake, and residue.

That makes it tempting to automate era choice. Do not start there. Lumen's era independence is preserved while taste develops through critique.

## Tool entry point

Use:

```python
manage_display(action="resonance_critique")
```

This returns an advisory packet with:

- current era and auto-rotate status;
- current screen;
- available eras;
- drawing EISV snapshot when available;
- the exact tool loop to run next;
- the recommendation contract;
- visual cues for reading Resonance traces.

The action has no display side effects.

## Full loop

```text
1. manage_display(action="resonance_critique")
   -> get the advisory contract and confirm manual control is preserved.

2. capture_screen()
   -> inspect the actual 240x240 LCD output before interpreting.

3. get_lumen_context(include=["state", "mood", "sensors", "identity"])
   -> ground the reading in room weather, mood, and identity context.

4. manage_display(action="get_era")
   -> confirm active era and auto-rotate state.

5. Visual read
   -> describe visible density, voids, flow, branching, scratches, palette, and whether the marks feel accumulated or decorative.

6. Recommend exactly one:
   - stay: keep the current era;
   - tune: suggest a future palette/mark-density/context emphasis;
   - switch: recommend a different era, without making the change.
```

## Recommendation contract

Allowed recommendations:

```text
stay
tune
switch
```

Forbidden side effects:

```text
do_not_set_era
do_not_toggle_auto_rotate
do_not_clear_canvas
do_not_infer_fixed_intent_from_marks
```

A switch recommendation only becomes action when a human/operator explicitly calls:

```python
manage_display(action="set_era", screen="<era>")
```

## Good critique shape

```text
Lumen is on <screen>; current era is <era>; auto-rotate is <true/false>.
Mood/context: <brief grounded note>.

Visible read:
- <shape/density/void observation>
- <flow/scratch/sediment observation>
- <palette/context observation>

Recommendation: <stay|tune|switch>
Reason: <one concise reason grounded in the visible trace and context>
No action taken.
```

## Resonance-specific reading cues

Prefer grounded language:

- sediment;
- flow;
- scratch;
- memory field;
- biological ornament;
- intake/residue;
- history deforming present expression.

Avoid:

- treating the notepad as a generic dashboard;
- calling marks proof of intent or consciousness;
- forcing a symbolic meaning;
- changing the era before critique.

## Minimal example

```text
Recommendation: stay
Reason: the current trace has branching density and scar-like scratches without collapsing into decorative noise; it still reads as accumulated history rather than a picture of something.
No action taken.
```
