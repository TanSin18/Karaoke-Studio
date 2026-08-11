# Region-based vocal fixing (Track Editor / Takes tab)

## Problem

Today, fixing part of an otherwise-good take requires either re-singing the
entire rest of the song (open-ended punch-in via `teRecFromHere`/"sing-along
continue") or manually building a 3-segment comp by hand in the Track
Editor's comping tool (drag the original take onto the "before" stretch,
drag a replacement take onto the "after" stretch, drag the fix onto the
middle) — because `/comp_multi` requires every stretch of the timeline to be
explicitly covered; nothing defaults to "keep what was already there."

There's also no way to just silence a bad word/breath/ad-lib without singing
a replacement, and no way to tell, later, exactly which take(s) contributed
to which part of a comp (`source_take_ids` is stored as a flat, unordered
set — the actual region boundaries used to build a specific comp aren't
recoverable), so a comp can't be reopened and tweaked — only rebuilt from
scratch.

## Goals

- Select a bounded region on the current take's waveform and either
  **re-record** just that window (auto-punch in/out) or **silence** it —
  without touching anything outside the region.
- Anything not explicitly edited defaults to the take that was active when
  editing started — you only ever specify what changed.
- The original, untouched take is never overwritten or deleted by any of
  this — every edit produces a new take.
- The exact recipe behind a comped take (which source/silence filled which
  region, in order) is persisted, so reopening that take later lets you
  swap out one region's source and rebuild, rather than starting over.
- The take strip visually distinguishes a clean single-take recording from
  a patched/comped one, with a segmented thumbnail showing where the seams
  are.
- Harmony takes are provably unaffected by any lead-vocal region edit.

## Non-goals

- No change to how takes are stored (still flat rendered WAV files per
  take) — see "Approach" below.
- No "cut this span from the whole song" editing (vocal + backing track
  both shortened). Silencing only removes the vocal; the backing track is
  untouched.
- No auto-detection of "bad" regions — the singer picks the region by hand.

## Approach

Extend the existing bake-once comp model (`TE.segments` → `/comp_multi` →
`audio_fx.stitch_multi`) rather than turning takes into a live-editable
recipe that's baked lazily at export time. The bake-once model is what the
rest of the app already assumes (every "Keep" is a new, immutable take);
reopening a comp to tweak it is solved by persisting its recipe as
metadata, not by changing what a take *is*. Revisit a fully live/persistent
segment model only if reopen-and-rebuild proves limiting in practice.

## User flow

1. In the Track Editor, drag on the active vocal lane to select an in/out
   region — the same drag gesture `teAssignSegment` already uses for
   cross-take comping, now also usable against the take you're currently
   viewing (not just to drop a *different* take onto).
2. A side panel appears (anchored, not floating over the waveform) showing:
   - The region's exact bounds (`fmt(start)`–`fmt(end)`).
   - Which take currently fills it.
   - Three actions: **🎤 Re-record**, **🔇 Silence**, and a list of other
     existing takes to assign instead (reuses `teAssignSegment`).
3. **Re-record:**
   - ~2-3s of pre-roll plays immediately before the region's start (mic
     live, MediaRecorder not yet started) so the singer can find the
     pocket — mirrors how mid-song punch-ins already skip the 4-click
     count-in (`startRec`'s existing `if(metro.countIn && !punch)` guard).
   - Recording starts exactly at the region's start (reuses the existing
     `pendingContinueAt` punch-in mechanism).
   - Recording **auto-stops** when `backingAudio.currentTime` reaches the
     region's end (new — today punch-ins only stop when the user hits
     Stop).
   - The captured audio becomes a new, standalone take (new id, `kind`
     tag distinguishing it as a region fix, e.g. `"punch"`) — it is never
     merged into or overwritten on top of the original.
   - That new take is auto-assigned to fill the selected region in the
     current edit session's segment list.
4. **Silence:** adds a `{silence: true, start, end}` entry to the segment
   list — no recording involved.
5. **Build:** when the user is done editing regions and clicks Build (as
   today), the client fills every *unedited* stretch of the timeline with
   the take that was active when editing began (new — today the user must
   drag these manually), producing a fully-specified segment list, then
   POSTs to `/comp_multi` as it does now.
6. The resulting comp take stores its full ordered segment recipe (see
   Data model) so it can be reloaded into the editor later; picking "edit
   this comp" loads that recipe back into `TE.segments`, letting the user
   change or re-source any one region and rebuild (a new take, again).

## Data model

- `stitch_multi` (`audio_fx.py`) gains a `"silence"` pseudo-segment: a
  segment dict without a `path` (or with `path: None`) renders as literal
  zero-samples for `end - start` seconds instead of reading/slicing a
  source file. Crossfade behavior at its boundaries stays the same as any
  other segment-to-segment join.
- `/comp_multi` (`studio.py`) accepts segments with either a `take_id` or
  an explicit `"silence": true` marker in place of `take_id`.
- The take record created by `/comp_multi` (`_add_take(..., kind="comp",
  source_take_ids=...)`) gains a new field, `comp_segments` — the full
  ordered segment recipe (`[{take_id | "silence", start, end}, ...]`) —
  alongside the existing flattened `source_take_ids` (kept for backward
  compatibility with anything that only needs "which takes contributed,"
  e.g. harmony `source_take_ids` linking).
- No schema change to plain (`lead`/`harmony`) takes.

## UI

- **Side panel** (not a floating toolbar) — validated via mockup: shows
  region bounds, current filler, the three actions, and the existing-take
  picker together, so nothing needs a second click to discover.
- **Take strip:** a take with a single-source recipe (or no recipe — a
  plain `lead`/`harmony` take) renders its existing plain waveform
  thumbnail. A take with a multi-entry recipe renders a **segmented
  thumbnail** — visible seams at each region boundary, with silenced spans
  rendered in a visually distinct (e.g. hatched/muted) style — validated
  via mockup over a flat "PATCHED" badge chip.

## Harmony independence

No design change needed here — it already works correctly, and this spec
should not disturb it: every take (lead, harmony, comp) aligns to one
absolute song-time coordinate system (sample 0 = song position 0, per
`alignTake`/`stitch_multi`'s existing convention), so building or rebuilding
a lead comp never requires re-aligning or re-recording any harmony take —
harmonies keep playing against whichever lead/comp is currently active.
Region-editing code must not write to harmony takes' `source_take_ids` or
alignment (`align_ms`) as a side effect of a lead rebuild.

## Error handling

- Reuses `/comp_multi`'s existing out-of-range clamp/rejection (a region
  requested past a source take's actual audio errors out client-side
  before submission, per `teAssignSegment`'s existing duration clamp).
- Auto-stop must not fire mid-recording-start race (mirror the existing
  `recorder.onstart` timing anchor rather than a naive `setTimeout`).
- If pre-roll playback fails to prime in time (rare), fail closed: don't
  silently start recording without pre-roll — surface a toast and let the
  user retry, matching this codebase's existing "toast, don't silently
  skip" convention (see the harmony/karaoke buffer-load error handling).

## Testing

- `stitch_multi`: unit-exercise a segment list containing a `"silence"`
  entry and confirm the rendered span is literal silence (not decoded
  from a source), with correct duration and clean crossfades at both
  edges.
- `/comp_multi`: a segment list mixing take-sourced and silence segments
  round-trips into a take with a persisted recipe.
- Reopen-and-rebuild: build a comp, reload its recipe into the editor,
  change one region's source, rebuild — confirm the new take is a new id
  (original comp untouched) and the recipe reflects the swap.
- Manual/browser: full region re-record flow against a real recording
  session — pre-roll audible, auto-stop lands at the region's out-point,
  harmony take(s) on the same session remain playable/aligned after the
  lead is rebuilt, take-strip thumbnail shows the seam.
