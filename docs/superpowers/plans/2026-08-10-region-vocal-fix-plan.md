# Region-based Vocal Fixing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the singer select a bounded region on their current take's waveform and either re-record just that window (auto punch-in/out) or silence it, while everything outside the region stays exactly as it was — replacing today's all-or-nothing open-ended punch-in and fully-manual 3-segment comping.

**Architecture:** Extends the existing bake-once comp pipeline (`TE.segments` client state → `/comp_multi` → `audio_fx.stitch_multi`) rather than introducing a new live-editable take model. Adds a `"silence"` pseudo-segment to `stitch_multi`, persists the exact segment recipe on the resulting comp take (`comp_segments`) so it can be reloaded and re-edited later, and adds a region-action side panel plus a bounded (auto in/out) punch-in recording flow on the client.

**Tech Stack:** Python stdlib HTTP server (`studio.py`), ffmpeg/librosa/soundfile DSP (`audio_fx.py`), single-file vanilla-JS frontend (`index.html`, Web Audio API). No test framework exists in this repo (confirmed: no `tests/` dir, no pytest/jest in `requirements.txt` or elsewhere) — backend verification uses standalone scripts run via `.venv/bin/python3 -c "..."` (the pattern already used to verify the `mixdown()` fixes in this project's history); frontend verification is manual, driving the real app in a browser and checking the console, matching this repo's existing commit-message convention ("Verified: ... button present, no console errors").

Full design rationale lives in `docs/superpowers/specs/2026-08-10-region-vocal-fix-design.md` — read it first if anything below is unclear.

## Global Constraints

- Silencing a region only removes the vocal there; the backing track is never touched or shortened.
- Re-recording a region **auto-stops at the region's out-point** — no manual stop needed for the common case.
- Anything the user hasn't explicitly edited must default to the take that was active when editing began — never require the user to manually cover the whole timeline just to fix one spot.
- No take is ever overwritten or deleted by any of this — every edit (silence, re-record, rebuild) produces a brand-new take; originals stay selectable.
- A comped take's exact recipe (ordered source/silence list) must be persisted so it can be reloaded and re-edited later, not just its flattened `source_take_ids` set.
- Harmony takes must never be touched, re-aligned, or have their `source_take_ids`/`align_ms` modified as a side effect of editing the lead vocal.
- Follow this repo's existing conventions: vanilla JS (no framework/build step) in `index.html`, Python stdlib HTTP handlers in `studio.py`, ffmpeg `-filter_complex` graphs in `audio_fx.py`.

---

## File Structure

- **Modify `audio_fx.py`** (`stitch_multi`, ~line 874): accept a `"silence"` pseudo-segment (no `path`) that renders as literal zero-samples for its duration.
- **Modify `studio.py`**:
  - `_add_take` (~line 915): new `comp_segments=None` parameter, stored on the take dict.
  - `/comp_multi` handler (~line 3226): accept segments with `"silence": true` instead of `take_id`; build and persist the full ordered recipe via the new `comp_segments` param.
- **Modify `index.html`**:
  - `TE` object (~line 4977): add `baseTakeId` and `selectedRegion` fields.
  - `wireTeLaneEvents`/`teAssignSegment` (~lines 5761-5823): refactor the assign logic into a shared `teAssignRegionEntry` so both take-assignment and silence-assignment reuse the same overlap-clipping code; open the new region panel after every drag.
  - New region-action side panel: markup near `teCompBuildRow` (~line 1444), open/close/populate logic, Silence button, existing-take picker.
  - `teDrawSegmentOverlays`/`teUpdateSegmentsSummary` (~lines 5227-5269): render silence entries distinctly.
  - `teBuildCompBtn` handler (~line 5827): auto-fill unedited gaps from `TE.baseTakeId` before POSTing; send `silence` markers.
  - Bounded punch-in recording: new globals near `pendingContinueAt` (~line 4714), auto-stop watcher in `startRec` (~line 6287), `boundedPunch` flag threaded through `finishRec`/`applyAlignmentAndFinish` (~lines 6405, 6491) to skip the stitch-with-kept-buffer path, post-Keep auto-assignment into `TE.segments`.
  - Reopen-a-comp: new "Edit this comp's regions" button, loads `comp_segments` back into `TE.segments`.
  - `renderTakeStrip` (~line 4938): segmented proportional bar for comp takes with >1 recipe entry.

---

### Task 1: `stitch_multi` silence-segment support

**Files:**
- Modify: `audio_fx.py:874-917` (`stitch_multi`)

**Interfaces:**
- Consumes: nothing new (pure function, no dependency on other tasks).
- Produces: `stitch_multi(segments, out_wav, crossfade_ms=40)` now accepts segment dicts with either `{"path": <wav>, "start": float, "end": float|None}` (existing) or `{"path": None, "start": float, "end": float}` (new — silence; `end` is required, unlike the take-sourced case where `None` means "to the take's own end"). Task 2 depends on this exact shape.

- [ ] **Step 1: Write the verification script (no test framework in this repo — ad-hoc script matching existing project convention)**

Create `/tmp/verify_stitch_silence.py`:

```python
import numpy as np, soundfile as sf
import audio_fx

sr = 48000
tone = (0.3 * np.ones(sr * 1, dtype=np.float32))   # 1s of constant 0.3 amplitude
sf.write('/tmp/verify_tone.wav', tone, sr)

segments = [
    {"path": "/tmp/verify_tone.wav", "start": 0, "end": 0.5},
    {"path": None, "start": 0.5, "end": 1.0},          # 0.5s silence, no path
    {"path": "/tmp/verify_tone.wav", "start": 0.5, "end": 1.0},
]
audio_fx.stitch_multi(segments, "/tmp/verify_out.wav", crossfade_ms=40)

y, sr2 = sf.read("/tmp/verify_out.wav", dtype="float32")
assert sr2 == sr, f"sample rate changed: {sr2}"
# expected length: 0.5+0.5+0.5=1.5s of audio minus 2 crossfade overlaps (40ms each)
xf = int(0.04 * sr)
expected_len = int(1.5 * sr) - 2 * xf
assert abs(len(y) - expected_len) <= 4, f"unexpected length {len(y)} vs {expected_len}"

# the middle third should be near-silent (the silence segment, minus crossfade edges)
mid_start = int(0.5 * sr) + xf
mid_end = int(1.0 * sr) - xf
mid_rms = float(np.sqrt(np.mean(y[mid_start:mid_end] ** 2)))
assert mid_rms < 0.01, f"expected near-silence in the middle, got RMS {mid_rms}"

print("stitch_multi silence support: OK, len=%d, mid_rms=%.5f" % (len(y), mid_rms))
```

- [ ] **Step 2: Run it to confirm it currently fails**

```bash
.venv/bin/python3 /tmp/verify_stitch_silence.py
```

Expected: `KeyError: 'path'` (or similar) from `librosa.load(seg["path"], ...)` when it hits the silence segment, since `stitch_multi` doesn't yet handle a missing/`None` path.

- [ ] **Step 3: Implement silence-segment support**

Replace `audio_fx.py:874-917` (the full body of `stitch_multi`) with:

```python
def stitch_multi(segments, out_wav, crossfade_ms=40):
    """
    Non-destructive multi-region comping: an arbitrary ordered list of
    takes/regions, so you can pick a different take for each stretch of the
    song instead of being limited to one head take + one tail take.

    segments: ordered list of {"path": wav_path, "start": float, "end": float}.
    start/end are in SONG time, not each take's own — every take's sample 0
    already corresponds to song position 0 (see alignTake() client-side), so
    slicing the same [start,end) out of each take's file lines them up with
    no per-segment offset math needed. Adjacent segments are joined with a
    short equal-power crossfade applied at every boundary.

    A segment with no "path" (or path=None) is rendered as literal silence
    for its [start, end) duration instead of being read from a file — used
    to silence a region of the lead vocal without a replacement take. Unlike
    take-sourced segments, a silence segment's "end" is required (there's no
    "to this take's own end" concept for silence).
    """
    if not segments:
        raise ValueError("no segments to comp")

    # Establish the working sample rate from the first real (non-silent)
    # segment — sf.info() is a cheap header read, no full decode needed.
    sr = None
    for seg in segments:
        if seg.get("path"):
            sr = sf.info(seg["path"]).samplerate
            break
    if sr is None:
        raise ValueError("stitch_multi needs at least one non-silent segment to establish a sample rate")

    parts = []
    for seg in segments:
        if not seg.get("path"):
            start_samp = max(0, int(round(float(seg["start"]) * sr)))
            end_samp = max(start_samp, int(round(float(seg["end"]) * sr)))
            parts.append(np.zeros(end_samp - start_samp, dtype=np.float32))
            continue
        y, this_sr = librosa.load(seg["path"], sr=sr, mono=True)
        start = max(0, int(float(seg["start"]) * sr))
        end = len(y) if seg.get("end") is None else int(float(seg["end"]) * sr)
        end = min(max(end, start), len(y))
        parts.append(y[start:end])

    xf = max(1, int(crossfade_ms / 1000.0 * sr))
    out = parts[0]
    for nxt in parts[1:]:
        if len(out) >= xf and len(nxt) >= xf:
            fade_out = np.cos(np.linspace(0, np.pi / 2, xf)) ** 1
            fade_in = np.sin(np.linspace(0, np.pi / 2, xf)) ** 1
            joined_mid = out[-xf:] * fade_out + nxt[:xf] * fade_in
            out = np.concatenate([out[:-xf], joined_mid, nxt[xf:]])
        else:
            out = np.concatenate([out, nxt])

    peak = np.max(np.abs(out)) or 1.0
    if peak > 0.99:
        out = out * (0.99 / peak)

    sf.write(out_wav, out.astype(np.float32), sr)
```

- [ ] **Step 4: Run the verification script again to confirm it passes**

```bash
.venv/bin/python3 /tmp/verify_stitch_silence.py
```

Expected: `stitch_multi silence support: OK, len=..., mid_rms=...` with no assertion errors.

- [ ] **Step 5: Regression-check the existing (non-silence) path still works**

```bash
.venv/bin/python3 -c "
import numpy as np, soundfile as sf, audio_fx
sr=48000
sf.write('/tmp/rt1.wav', (0.2*np.ones(sr*2)).astype(np.float32), sr)
sf.write('/tmp/rt2.wav', (0.4*np.ones(sr*2)).astype(np.float32), sr)
audio_fx.stitch_multi(
    [{'path':'/tmp/rt1.wav','start':0,'end':1},
     {'path':'/tmp/rt2.wav','start':1,'end':2}],
    '/tmp/rt_out.wav')
y,_=sf.read('/tmp/rt_out.wav')
print('regression ok, len=', len(y))
"
```

Expected: `regression ok, len=...` with no exception.

- [ ] **Step 6: Clean up scratch files and commit**

```bash
rm -f /tmp/verify_tone.wav /tmp/verify_out.wav /tmp/rt1.wav /tmp/rt2.wav /tmp/rt_out.wav /tmp/verify_stitch_silence.py
git add audio_fx.py
git commit -m "$(cat <<'EOF'
Add silence-segment support to stitch_multi

A segment with no path now renders as literal silence for its
duration, so a region of the lead vocal can be muted without a
replacement take — needed by the upcoming region-based vocal fixing
feature in the Track Editor.
EOF
)"
```

---

### Task 2: `/comp_multi` accepts silence segments and persists the full recipe

**Files:**
- Modify: `studio.py:915-938` (`_add_take`)
- Modify: `studio.py:3226-3283` (`/comp_multi` handler)

**Interfaces:**
- Consumes: `audio_fx.stitch_multi` segment shape from Task 1 (`{"path": None, "start", "end"}` for silence).
- Produces: `/comp_multi` now accepts request segments as either `{"take_id": str, "start": float, "end": float|None}` (existing) or `{"silence": true, "start": float, "end": float}` (new). The take record it creates gains `comp_segments`: `[{"take_id": str, "start": float, "end": float} | {"silence": true, "start": float, "end": float}, ...]` in request order — Task 4/6 (frontend) consume this exact shape when reloading a comp for further editing.

- [ ] **Step 1: Modify `_add_take` to accept and store `comp_segments`**

In `studio.py`, change the signature and body at lines 915-938 from:

```python
def _add_take(jid, file, duration, kind="lead", fx_snapshot=None,
              punch_sec=None, source_take_ids=None, tid=None, pitch_score=None,
              latency_ms=None):
    tid = tid or uuid.uuid4().hex[:8]
    meta = _read_meta(jid)
    takes = meta.get("takes") or []
    take = {
        "id": tid, "file": file, "duration": duration, "kind": kind,
        "fx_snapshot": fx_snapshot, "punch_sec": punch_sec,
        "source_take_ids": source_take_ids, "deleted": False,
        "pitch_score": pitch_score,
        "latency_ms": latency_ms,
        "align_ms": 0,
    }
    takes.append(take)
    _write_meta(jid, takes=takes)
    return take
```

to:

```python
def _add_take(jid, file, duration, kind="lead", fx_snapshot=None,
              punch_sec=None, source_take_ids=None, tid=None, pitch_score=None,
              latency_ms=None, comp_segments=None):
    tid = tid or uuid.uuid4().hex[:8]
    meta = _read_meta(jid)
    takes = meta.get("takes") or []
    take = {
        "id": tid, "file": file, "duration": duration, "kind": kind,
        "fx_snapshot": fx_snapshot, "punch_sec": punch_sec,
        "source_take_ids": source_take_ids, "deleted": False,
        "pitch_score": pitch_score,
        "latency_ms": latency_ms,
        "align_ms": 0,
        # the exact ordered region recipe that built this take (comp takes
        # only) — [{"take_id"|"silence": ..., "start", "end"}, ...] — lets
        # the Track Editor reload and re-edit a comp later instead of only
        # seeing the flattened source_take_ids set.
        "comp_segments": comp_segments,
    }
    takes.append(take)
    _write_meta(jid, takes=takes)
    return take
```

- [ ] **Step 2: Modify `/comp_multi` to accept silence segments and build the recipe**

Replace `studio.py:3226-3283` (the full `/comp_multi` block) with:

```python
        if p.path == "/comp_multi":
            # Generalizes /comp from exactly 2 takes at 1 punch point to an
            # arbitrary ordered list of (take, region) segments — pick a
            # different take for each stretch of the song instead of one
            # head + one tail take. A segment may also be {"silence": true}
            # instead of a take_id, to mute a region without a replacement.
            body = self._read_body()
            jid = body.get("id")
            raw_segments = body.get("segments") or []
            if len(raw_segments) < 1:
                self._json(400, {"error": "need at least one segment"})
                return
            sdir = os.path.join(SESS_DIR, jid)
            segments = []
            source_ids = []
            recipe = []
            for s in raw_segments:
                try:
                    start = float(s.get("start", 0))
                except (TypeError, ValueError):
                    self._json(400, {"error": "segment start/end must be numbers"})
                    return

                if s.get("silence"):
                    if s.get("end") is None:
                        self._json(400, {"error": "a silence segment needs an explicit end"})
                        return
                    try:
                        end = float(s["end"])
                    except (TypeError, ValueError):
                        self._json(400, {"error": "segment start/end must be numbers"})
                        return
                    segments.append({"path": None, "start": start, "end": end})
                    recipe.append({"silence": True, "start": start, "end": end})
                    continue

                t = _get_take(jid, s.get("take_id"))
                if not t:
                    self._json(404, {"error": f"unknown take {s.get('take_id')}"})
                    return
                try:
                    end = float(s["end"]) if s.get("end") is not None else None
                except (TypeError, ValueError):
                    self._json(400, {"error": "segment start/end must be numbers"})
                    return
                # Reject regions that lie past the end of the take's audio
                # instead of "stitching" them: y[start:end] on an
                # out-of-range slice is silently empty, and a comp built
                # from empty slices writes a ~0-second take that then shows
                # up in the strip looking like the feature just didn't work.
                tdur = t.get("duration") or _probe_duration(
                    os.path.join(sdir, t["file"])) or 0
                if tdur and start >= tdur - 0.05:
                    m, s2 = divmod(int(tdur), 60)
                    self._json(400, {"error":
                        f"A region starts past the end of that take's audio "
                        f"(it ends at {m}:{s2:02d}). Drag regions only where "
                        f"the take actually has a waveform."})
                    return
                if tdur and end is not None:
                    end = min(end, tdur)
                segments.append({"path": os.path.join(sdir, t["file"]), "start": start, "end": end})
                source_ids.append(t["id"])
                recipe.append({"take_id": t["id"], "start": start, "end": end})
            tid = uuid.uuid4().hex[:8]
            fname = f"take_{tid}.wav"
            out_path = os.path.join(sdir, fname)
            try:
                audio_fx.stitch_multi(segments, out_path,
                                       crossfade_ms=float(body.get("crossfade_ms", 40)))
            except Exception as e:
                self._json(500, {"error": "Could not join those takes: " + str(e)[:300]})
                return
            take = _add_take(
                jid, file=fname, duration=_probe_duration(out_path), kind="comp",
                source_take_ids=source_ids, tid=tid, comp_segments=recipe,
            )
            self._json(200, {"take": take})
            return
```

- [ ] **Step 3: Sanity-check the file parses and the server still starts**

```bash
.venv/bin/python3 -c "import ast; ast.parse(open('studio.py').read()); print('syntax ok')"
lsof -ti tcp:8770 | xargs -r kill
sleep 1
nohup .venv/bin/python3 studio.py >/tmp/karaoke-studio.log 2>&1 &
sleep 2
curl -s -o /dev/null -w "http status: %{http_code}\n" http://localhost:8770
```

Expected: `syntax ok` then `http status: 200`.

- [ ] **Step 4: Write and run an end-to-end verification script against the real running server**

This needs a real session with at least one take. Create `/tmp/verify_comp_multi.py`:

```python
import json, urllib.request, base64, wave, struct, io

BASE = "http://localhost:8770"

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())

def make_wav_b64(seconds, amplitude, sr=48000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        n = int(seconds * sr)
        w.writeframes(struct.pack("<%dh" % n, *[int(amplitude * 32767)] * n))
    return base64.b64encode(buf.getvalue()).decode()

# 1. create a bare session (no song needed for this test — we only exercise
#    /take and /comp_multi, not rendering)
sess = post("/session/new", {}) if False else None  # placeholder path not needed;
# most installs create sessions via /load_song; instead reuse an existing
# session id if present, else skip gracefully.
print("This script expects SID to be supplied — see manual step below.")
```

Because creating a full session requires a real song load (network-dependent,
out of scope for a unit-style check), do this step **manually** against a
session you already have open in the browser instead of scripting session
creation:

1. Open the app, load any song, record one take, hit Keep.
2. Open the browser console and run:

```javascript
const takeId = takes[0].id;
fetch('/comp_multi', {method:'POST', headers:{'Content-Type':'application/json'},
  body: JSON.stringify({id: SID, segments: [
    {take_id: takeId, start: 0, end: 2},
    {silence: true, start: 2, end: 3},
    {take_id: takeId, start: 3, end: 5},
  ]})}).then(r=>r.json()).then(d=>console.log('comp result:', d));
```

Expected: `comp result: {take: {..., kind: "comp", comp_segments: [
  {take_id: "...", start: 0, end: 2}, {silence: true, start: 2, end: 3},
  {take_id: "...", start: 3, end: 5}]}}` — no error, and `comp_segments`
present with all three entries in order.

3. Confirm persistence: `fetch('/takes?id='+SID).then(r=>r.json()).then(d=>console.log(d.takes.find(t=>t.kind==='comp').comp_segments))` — should print the same three-entry array after a fresh fetch (not just the POST response), proving it round-trips through `_write_meta`/`_read_meta`.

- [ ] **Step 5: Commit**

```bash
git add studio.py
git commit -m "$(cat <<'EOF'
Persist comp_segments and accept silence markers in /comp_multi

/comp_multi now accepts {"silence": true, start, end} segments (muting
a region with no replacement take) and persists the full ordered
segment recipe on the resulting take, not just the flattened
source_take_ids set — needed to reload and re-edit a comp later.
EOF
)"
```

---

### Task 3: Base-take tracking and gap auto-fill before building a comp

**Files:**
- Modify: `index.html:4977-4998` (`TE` object)
- Modify: `index.html:5476-5488` (`teCompareMode` change handler)
- Modify: `index.html:5827-5841` (`teBuildCompBtn` handler)

**Interfaces:**
- Consumes: nothing new from earlier tasks (pure frontend state).
- Produces: `TE.baseTakeId` (string|null) — the take considered "unedited default" for the current editing session. `teSegmentsWithGapsFilled()` — returns `TE.segments` plus synthetic entries covering any stretch of `[0, TE.duration]` not explicitly assigned, using `TE.baseTakeId`. `async teSetCompareMode(on)` — the single entry point for turning Compare mode on/off (sets `TE.compareMode`, the checkbox, `baseTakeId`, FX-preview mutual exclusion, and calls `refreshTrackEditor`); Task 5 and Task 6 both call this directly instead of re-implementing any of it. Task 4 and Task 5 both add entries to `TE.segments` that `teSegmentsWithGapsFilled` must handle unchanged.

- [ ] **Step 1: Add `baseTakeId` to the `TE` object**

In `index.html`, in the `TE` object literal (starts at line 4977), add a field near `segments`:

```javascript
  segments: [],   // [{takeId, start, end}] — non-destructive multi-region comp
  baseTakeId: null,   // the take considered "the default" for any stretch
                       // the user hasn't explicitly edited this session
  markers: [],    // [{time, label}]
```

- [ ] **Step 2: Extract Compare-mode entry into a shared function, and set `baseTakeId` there**

Find the `teCompareMode` change handler (`index.html:5476`):

```javascript
$('teCompareMode').addEventListener('change', async()=>{
  TE.compareMode=$('teCompareMode').checked;
  $('teCompBuildRow').classList.toggle('hidden', !TE.compareMode);
  // FX Preview and Compare mode both want to claim "the vocal lane" — the
  // former for one specific take, the latter for every take at once — so
  // they're mutually exclusive rather than trying to define what "preview"
  // means across N simultaneous lanes.
  if(TE.compareMode && TE.fxPreviewOn){
    TE.fxPreviewOn=false; $('teFxPreviewToggle').checked=false;
    if(TE.playing){ teStopPlayback(); tePlay(); }
  }
  $('teFxPreviewWrap').classList.toggle('hidden', TE.compareMode);
  await refreshTrackEditor();
});
```

Replace the whole block with a named function plus a thin listener, so Task 6's "reopen a comp" flow can enter Compare mode the exact same way instead of re-implementing a subset of it:

```javascript
// Single entry point for turning Compare mode on/off — used by the
// checkbox below AND by "Edit this comp's regions" (see teEditCompBtn),
// so both paths get the same baseTakeId/FX-preview/build-row side effects.
async function teSetCompareMode(on){
  TE.compareMode=on;
  $('teCompareMode').checked=on;
  if(on && !TE.baseTakeId) TE.baseTakeId=activeTakeId;
  if(!on) TE.baseTakeId=null;
  $('teCompBuildRow').classList.toggle('hidden', !on);
  // FX Preview and Compare mode both want to claim "the vocal lane" — the
  // former for one specific take, the latter for every take at once — so
  // they're mutually exclusive rather than trying to define what "preview"
  // means across N simultaneous lanes.
  if(on && TE.fxPreviewOn){
    TE.fxPreviewOn=false; $('teFxPreviewToggle').checked=false;
    if(TE.playing){ teStopPlayback(); tePlay(); }
  }
  $('teFxPreviewWrap').classList.toggle('hidden', on);
  await refreshTrackEditor();
}
$('teCompareMode').addEventListener('change', ()=>teSetCompareMode($('teCompareMode').checked));
```

- [ ] **Step 3: Add the gap-filling function**

Add this new function right before the `$('teBuildCompBtn')` handler (before line 5827):

```javascript
// Anything the user hasn't explicitly dragged a region onto defaults to
// TE.baseTakeId (the take that was active when Compare mode was turned on)
// — so fixing one region never requires manually covering the rest of the
// song too.
function teSegmentsWithGapsFilled(){
  const sorted=[...TE.segments].sort((a,b)=>a.start-b.start);
  const filled=[]; let cursor=0;
  sorted.forEach(seg=>{
    if(seg.start>cursor+0.01 && TE.baseTakeId){
      filled.push({takeId:TE.baseTakeId, start:cursor, end:seg.start});
    }
    filled.push(seg);
    cursor=Math.max(cursor, seg.end);
  });
  if(cursor<TE.duration-0.01 && TE.baseTakeId){
    filled.push({takeId:TE.baseTakeId, start:cursor, end:TE.duration});
  }
  return filled;
}
```

- [ ] **Step 4: Use it in the build handler**

Change `index.html:5827-5841` from:

```javascript
$('teBuildCompBtn').addEventListener('click', async()=>{
  if(!TE.segments.length){ toast('Drag at least one region on a lane first.','warn'); return; }
  $('teBuildCompBtn').disabled=true;
  try{
    const r=await fetch('/comp_multi',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({id:SID, segments:TE.segments.map(s=>({take_id:s.takeId,start:s.start,end:s.end}))})});
```

to:

```javascript
$('teBuildCompBtn').addEventListener('click', async()=>{
  if(!TE.segments.length){ toast('Drag at least one region on a lane first.','warn'); return; }
  $('teBuildCompBtn').disabled=true;
  try{
    const full=teSegmentsWithGapsFilled();
    const r=await fetch('/comp_multi',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({id:SID, segments:full.map(s=>
        s.silence ? {silence:true, start:s.start, end:s.end}
                   : {take_id:s.takeId, start:s.start, end:s.end})})});
```

(Leave the rest of the handler — `const d=await r.json(); ...` through the closing `});` — unchanged.)

- [ ] **Step 5: Manual verification**

1. Restart the server (`lsof -ti tcp:8770 | xargs -r kill && python3 studio.py &`), open the app, load a song, record and Keep one take.
2. In the Takes tab, check "Compare takes".
3. Open the browser console and run `TE.baseTakeId` — expect it to print the id of the take you just kept (not `null`).
4. Drag a **small** region (a couple seconds) on that same take's own lane.
5. In the console, run `teSegmentsWithGapsFilled()` — expect an array with **3** entries: a `baseTakeId` entry from `0` to your region's start, your dragged entry, and a `baseTakeId` entry from your region's end to `TE.duration`.
6. Click "↳ Build comp from regions" — expect success (no "need at least one segment" or other error toast), and a new take appears in the strip whose duration equals the full song, not just your dragged region's length.

- [ ] **Step 6: Commit**

```bash
git add index.html
git commit -m "$(cat <<'EOF'
Auto-fill unedited comp regions from the base take

Building a comp used to require manually dragging a segment across
every stretch of the song, even the parts you didn't touch. Track
which take was active when Compare mode was turned on and fill any
gap in the dragged regions with it before submitting to /comp_multi.
EOF
)"
```

---

### Task 4: Region-action side panel with Silence support

**Files:**
- Modify: `index.html:1444-1448` (markup, add panel after `teCompBuildRow`)
- Modify: `index.html:5227-5243` (`teDrawSegmentOverlays`)
- Modify: `index.html:5266-5269` (`teUpdateSegmentsSummary`)
- Modify: `index.html:5761-5823` (`wireTeLaneEvents`, `teAssignSegment`)
- Modify: CSS near `index.html:486` (`.te-region`)

**Interfaces:**
- Consumes: `teSegmentsWithGapsFilled` is unaffected by this task (still reads `TE.segments` directly); `TE.baseTakeId` from Task 3 used to show "currently: <default>" in the panel.
- Produces: `TE.selectedRegion` (`{start, end}|null`). `teAssignRegionEntry(entry)` — takes `{takeId, start, end}` or `{silence:true, start, end}`, does the existing overlap-clip-and-insert logic. `teOpenRegionPanel(start, end)` / `teCloseRegionPanel()`. Task 5 (Re-record button) and Task 6 (reopen-a-comp) both call `teOpenRegionPanel`/read `TE.selectedRegion`.

- [ ] **Step 1: Add the panel markup**

In `index.html`, right after the `teCompBuildRow` div (ends at line 1448), add:

```html
          <div class="railCard hidden" id="teRegionPanel" style="margin-top:10px">
            <h3>Selected region</h3>
            <div class="klabel" id="teRegionBounds">0:00 – 0:00</div>
            <div class="klabel" id="teRegionFiller" style="margin-bottom:8px">currently: —</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
              <button class="btn small" id="teRegionRerecordBtn">🎤 Re-record</button>
              <button class="btn small" id="teRegionSilenceBtn">🔇 Silence</button>
              <button class="btn small ghost" id="teRegionCloseBtn">✕ Close</button>
            </div>
            <div class="klabel">or assign an existing take</div>
            <select id="teRegionTakeSelect" class="wide"></select>
          </div>
```

- [ ] **Step 2: Refactor `teAssignSegment` into a shared entry-assignment function**

Replace `index.html:5791-5823` (the current `teAssignSegment` function) with:

```javascript
// Shared overlap-clip-and-insert logic for anything that can fill a region:
// an existing take (entry.takeId) or silence (entry.silence===true).
function teAssignRegionEntry(entry){
  const {start, end}=entry;
  const clipped=[];
  TE.segments.forEach(seg=>{
    if(seg.end<=start || seg.start>=end){ clipped.push(seg); return; }
    if(seg.start<start) clipped.push({...seg, end:start});
    if(seg.end>end) clipped.push({...seg, start:end});
  });
  clipped.push(entry);
  clipped.sort((a,b)=>a.start-b.start);
  TE.segments=clipped;
  tePushHistory();
  teDrawSegmentOverlays();
  teUpdateSegmentsSummary();
}
function teAssignSegment(takeId, start, end){
  // A take is usually shorter than the song, and the server will happily
  // "stitch" a region that lies past its end — producing a comp of pure
  // silence with no error. Clamp to what the take actually contains.
  // Falls back to the take's saved duration when its buffer hasn't been
  // decoded yet — lanes are draggable the moment they render, before
  // ensureVocalBuffer resolves, and gating the clamp on the buffer alone
  // let a region drawn in that window land anywhere on the timeline
  // (which is how a 21s take ended up with a "01:03–02:08" region and a
  // 0.06-second comp built from pure silence).
  const buf=TE.vocalBuffers[takeId];
  const t=takes.find(x=>x.id===takeId);
  const known = buf ? buf.duration : (t && t.duration) || null;
  if(known){
    end=Math.min(end, known);
    if(end-start<=0.05){ toast('That take has no audio there — it ends at '+fmt(known)+'.','warn',3500); return; }
  }
  teAssignRegionEntry({takeId, start, end});
}
function teAssignSilence(start, end){
  teAssignRegionEntry({silence:true, start, end});
}
```

- [ ] **Step 3: Open the panel after every drag**

In `wireTeLaneEvents` (`index.html:5761-5789`), the `up` handler currently ends with:

```javascript
        const a=Math.min(startT,curT), b=Math.max(startT,curT);
        if(b-a>0.05) teAssignSegment(takeId, a, b);
      };
```

Change it to also open the panel:

```javascript
        const a=Math.min(startT,curT), b=Math.max(startT,curT);
        if(b-a>0.05){ teAssignSegment(takeId, a, b); teOpenRegionPanel(a, b); }
      };
```

- [ ] **Step 4: Add the panel open/close/populate logic and wire its buttons**

Add this after the `teAssignSilence` function from Step 2:

```javascript
function teCurrentFillerLabel(){
  const mid=(TE.selectedRegion.start+TE.selectedRegion.end)/2;
  const seg=TE.segments.find(s=>s.start<=mid && s.end>mid);
  if(seg && seg.silence) return 'silence';
  const tid = seg ? seg.takeId : TE.baseTakeId;
  if(!tid) return '—';
  const take=takes.find(x=>x.id===tid);
  return take ? (TAKE_KIND_LABEL[take.kind]||take.kind||'take')+' ('+tid.slice(0,6)+')' : tid.slice(0,6);
}
function teOpenRegionPanel(start, end){
  TE.selectedRegion={start, end};
  $('teRegionBounds').textContent=fmt(start)+' – '+fmt(end);
  $('teRegionFiller').textContent='currently: '+teCurrentFillerLabel();
  const sel=$('teRegionTakeSelect');
  sel.innerHTML='<option value="">— pick a take —</option>'+
    takes.filter(t=>t.kind!=='harmony').map(t=>
      '<option value="'+t.id+'">'+(TAKE_KIND_LABEL[t.kind]||t.kind)+' ('+t.id.slice(0,6)+')</option>').join('');
  $('teRegionPanel').classList.remove('hidden');
}
function teCloseRegionPanel(){
  TE.selectedRegion=null;
  $('teRegionPanel').classList.add('hidden');
}
$('teRegionCloseBtn').addEventListener('click', teCloseRegionPanel);
$('teRegionSilenceBtn').addEventListener('click', ()=>{
  if(!TE.selectedRegion) return;
  teAssignSilence(TE.selectedRegion.start, TE.selectedRegion.end);
  $('teRegionFiller').textContent='currently: '+teCurrentFillerLabel();
});
$('teRegionTakeSelect').addEventListener('change', e=>{
  if(!TE.selectedRegion || !e.target.value) return;
  teAssignSegment(e.target.value, TE.selectedRegion.start, TE.selectedRegion.end);
  $('teRegionFiller').textContent='currently: '+teCurrentFillerLabel();
});
```

(The `teRegionRerecordBtn` listener is added in Task 5 — leave it unbound here.)

- [ ] **Step 5: Render silence segments distinctly in the overlay and summary**

In `teDrawSegmentOverlays` (`index.html:5227` onward), find where it looks up the lane by `seg.takeId`:

```javascript
function teDrawSegmentOverlays(){
  $('teVocalLanes').querySelectorAll('.te-region').forEach(el=>el.remove());
  if(!TE.compareMode) return;
  TE.segments.forEach(seg=>{
    const lane=$('teVocalLanes').querySelector('.te-lane[data-take-id="'+seg.takeId+'"]');
    if(!lane) return;
```

Change the silence case to render on every visible lane (since silence isn't tied to one take's lane) using a dedicated overlay row instead of skipping. Simplest correct fix — draw silence regions on `teLaneKaraoke`'s row area (always present) so they're visible regardless of which take lane you're looking at:

```javascript
function teDrawSegmentOverlays(){
  $('teVocalLanes').querySelectorAll('.te-region').forEach(el=>el.remove());
  $('teLaneKaraoke').querySelectorAll('.te-region').forEach(el=>el.remove());
  if(!TE.compareMode) return;
  TE.segments.forEach(seg=>{
    if(seg.silence){
      const r=document.createElement('div');
      r.className='te-region silence';
      r.style.left=teTimeToPx(seg.start)+'px';
      r.style.width=Math.max(2, teTimeToPx(seg.end)-teTimeToPx(seg.start))+'px';
      $('teLaneKaraoke').appendChild(r);
      return;
    }
    const lane=$('teVocalLanes').querySelector('.te-lane[data-take-id="'+seg.takeId+'"]');
    if(!lane) return;
```

(Leave the rest of the function — the non-silence branch's region-creation code that follows — unchanged.)

Add CSS for the silence style near `.te-region` (`index.html:486`):

```css
  .te-region.silence{background:repeating-linear-gradient(45deg,rgba(120,120,120,.35) 0 4px,rgba(80,80,80,.35) 4px 8px)}
```

Update `teUpdateSegmentsSummary` (`index.html:5266-5269`) from:

```javascript
function teUpdateSegmentsSummary(){
  if(!$('teSegmentsSummary')) return;
  $('teSegmentsSummary').textContent = TE.segments.length
    ? TE.segments.map(s=>fmt(s.start)+'–'+fmt(s.end)+' → '+s.takeId.slice(0,6)).join('  ·  ')
```

to:

```javascript
function teUpdateSegmentsSummary(){
  if(!$('teSegmentsSummary')) return;
  $('teSegmentsSummary').textContent = TE.segments.length
    ? TE.segments.map(s=>fmt(s.start)+'–'+fmt(s.end)+' → '+(s.silence?'silence':s.takeId.slice(0,6))).join('  ·  ')
```

(Leave the trailing `: 'Drag on a lane below to assign that take to a region';` unchanged.)

- [ ] **Step 6: Manual verification**

1. Restart the server, open the app, load a song, record and Keep a take.
2. Check "Compare takes", drag a short region on the take's own lane.
3. Confirm the region panel appears showing the correct time bounds and `currently: <your take>`.
4. Click "🔇 Silence" — confirm the overlay for that span switches to a hatched/gray style, and `teUpdateSegmentsSummary`'s text (visible under the build button) shows `→ silence` for that range.
5. Click "↳ Build comp from regions" — confirm success and that exporting/rendering that comp (Export tab → Render) produces audio where that span is genuinely silent (listen to it, or check the exported file's waveform around that timestamp).
6. Pick a different take from the `teRegionTakeSelect` dropdown after dragging a new region — confirm the overlay updates to that take's assignment without needing to drag again.

- [ ] **Step 7: Commit**

```bash
git add index.html
git commit -m "$(cat <<'EOF'
Add region-action side panel with Silence support

Selecting a region on any take's lane in Compare mode now opens a
side panel showing its exact bounds, what currently fills it, and
lets you silence it or pick a different existing take from a
dropdown — instead of only being able to drag a different take's
lane on top of it. Silence renders as a distinct hatched overlay.
EOF
)"
```

---

### Task 5: Bounded (auto in/out) punch-in recording

**Files:**
- Modify: `index.html:4714` (globals near `pendingContinueAt`)
- Modify: `index.html:4789` (`keepBtn` handler)
- Modify: `index.html:6287-6373` (`startRec`)
- Modify: `index.html:6393-6403` (`stopRec`)
- Modify: `index.html:6405-6429` (`finishRec`)
- Modify: `index.html:6491-6513` (`applyAlignmentAndFinish`)
- Modify: region panel from Task 4 (`teRegionRerecordBtn` listener)

**Interfaces:**
- Consumes: `TE.selectedRegion`, `teOpenRegionPanel`/`teCloseRegionPanel`, `teAssignSegment` from Task 4.
- Produces: after Keep, a new standalone take (`kind:'punch'`) is created and auto-assigned into `TE.segments` for the region that was re-recorded — Task 3's `teSegmentsWithGapsFilled`/Task 4's panel both pick this up automatically since it's just another `TE.segments` entry.

**Implementation note — pre-roll simplification vs. the spec:** the spec describes pre-roll as "mic live, MediaRecorder not yet started." This plan instead just starts the existing punch-in mechanism a couple seconds *before* the region's actual start and relies on `/comp_multi` slicing to the exact `[start,end)` region later (Task 2/3) to discard that lead-in — because every take's sample 0 already means "song position 0" (the same convention every other take already uses), extra recorded audio outside the region is simply never referenced by the comp recipe. This reuses 100% of the existing, already-hardened punch-in start path (mic priming, latency alignment) instead of adding a second "armed but not recording" state, at the cost of the new take's raw WAV containing a few extra seconds nobody will hear. If real-world testing in Task 5 Step 8 or Task 8 shows this pre-roll feels off (e.g. auto-stop firing before the singer expected, or the lead-in not being audible for some reason), fall back to the spec's literal two-phase design instead of tuning this further.

- [ ] **Step 1: Add new globals**

Near `pendingContinueAt` (`index.html:4714`):

```javascript
let pendingContinueAt=null;   // set when user picks "continue from here"
let pendingPunchOutAt=null;   // set for a BOUNDED punch-in — auto-stops recording here
let boundedPunchActive=false; // true while a bounded punch-in's auto-stop watcher is armed
let boundedPunchRegion=null;  // {start,end} the bounded punch-in is fixing, for post-Keep auto-assign
```

- [ ] **Step 2: Wire the Re-record button in the region panel**

Add to the region panel wiring block from Task 4 Step 4:

```javascript
$('teRegionRerecordBtn').addEventListener('click', ()=>{
  if(!TE.selectedRegion) return;
  const {start, end}=TE.selectedRegion;
  const preRoll=Math.min(2.5, start);   // seconds of lead-in before the region, clamped so it never goes negative
  boundedPunchRegion={start, end};
  pendingContinueAt=Math.max(0, start-preRoll);
  pendingPunchOutAt=end+0.15;   // small safety margin — /comp_multi slices to the exact region later regardless
  boundedPunchActive=true;
  teCloseRegionPanel();
  teStopPlayback();
  switchSessionTab('stageCard');
  startRec();
});
```

- [ ] **Step 3: Add the auto-stop watcher in `startRec`**

In `startRec` (`index.html:6287`), find the tail end where `recorder.start()` is called (around line 6373):

```javascript
  recorder.start();
  // A timeslice arg (periodic chunks) rather than a single final blob avoids
  // a known Chrome quirk where a WebM recorded with no timeslice can end up
  // with wrong/estimated duration metadata in the container.
  if(window._videoRecordedThisTake){
```

Add the watcher right after `recorder.start();`, before the `if(window._videoRecordedThisTake){` line:

```javascript
  recorder.start();
  if(boundedPunchActive){
    (function watchPunchOut(){
      if(!recording) return;   // stopped some other way (manual Stop, etc.)
      if(backingAudio.currentTime>=pendingPunchOutAt){ stopRec(); return; }
      requestAnimationFrame(watchPunchOut);
    })();
  }
  // A timeslice arg (periodic chunks) rather than a single final blob avoids
  // a known Chrome quirk where a WebM recorded with no timeslice can end up
  // with wrong/estimated duration metadata in the container.
  if(window._videoRecordedThisTake){
```

- [ ] **Step 4: Capture the bounded-punch flag before it gets cleared, in `stopRec`**

Change `stopRec` (`index.html:6393-6403`) from:

```javascript
function stopRec(){ if(!recording)return;
  // remember how long we actually sang (song time) so we can trim precisely
  window._recEndMusicPos = backingAudio.currentTime || 0;
  recorder.stop(); backingAudio.pause();
```

to:

```javascript
function stopRec(){ if(!recording)return;
  // remember how long we actually sang (song time) so we can trim precisely
  window._recEndMusicPos = backingAudio.currentTime || 0;
  // finishRec (recorder.onstop) runs asynchronously after recorder.stop(),
  // by which point boundedPunchActive has already been reset below — stash
  // what THIS recording actually was before clearing the live flag.
  window._recWasBoundedPunch = boundedPunchActive;
  window._recBoundedPunchRegion = boundedPunchRegion;
  boundedPunchActive=false; pendingPunchOutAt=null;
  recorder.stop(); backingAudio.pause();
```

- [ ] **Step 5: Thread the flag through `finishRec` and `applyAlignmentAndFinish`**

Change `finishRec` (`index.html:6405-6429`) — the line building `window._rawTakeMeta`:

```javascript
  window._rawTakeMeta={ intendedStart:punchAt, musicPosAtRecStart, keptAtRecord:keptBuffer };
```

to:

```javascript
  window._rawTakeMeta={ intendedStart:punchAt, musicPosAtRecStart, keptAtRecord:keptBuffer,
    boundedPunch: window._recWasBoundedPunch||false };
```

Change `applyAlignmentAndFinish` (`index.html:6491-6513`) from:

```javascript
function applyAlignmentAndFinish(ctx){
  ctx=ctx||new (window.AudioContext||window.webkitAudioContext)({sampleRate:48000});
  const raw=window._rawTakeBuffer; if(!raw) return;
  const meta=window._rawTakeMeta;
  // where the take's sample 0 sits in song time, BEFORE latency correction:
  //   = musicPosAtRecStart. We want sample 0 == intendedStart.
  // Then subtract mic input latency (vocalDelayMs) to pull the voice earlier.
  const positionOffset = meta.musicPosAtRecStart - meta.intendedStart;
  const latency = vocalDelayMs/1000;
  const effOffset = positionOffset - latency;
  let aligned=alignTake(ctx, raw, effOffset);

  let finalBuffer=aligned;
  if(meta.intendedStart>0 && meta.keptAtRecord){
    finalBuffer=stitchBuffers(ctx, meta.keptAtRecord, aligned, meta.intendedStart);
    $('renderstatus').textContent='Continued from '+fmt(meta.intendedStart)+'. Vocal timing '+vocalDelayMs+'ms.';
  } else {
    $('renderstatus').textContent='Take captured ('+fmt(aligned.duration)+'). Vocal timing '+vocalDelayMs+'ms — tweak if it drifts.';
  }
  keptBuffer=finalBuffer;
  window._lastVocalBuffer=finalBuffer;
  window._lastVocalWav=encodeWav(finalBuffer);
}
```

to:

```javascript
function applyAlignmentAndFinish(ctx){
  ctx=ctx||new (window.AudioContext||window.webkitAudioContext)({sampleRate:48000});
  const raw=window._rawTakeBuffer; if(!raw) return;
  const meta=window._rawTakeMeta;
  // where the take's sample 0 sits in song time, BEFORE latency correction:
  //   = musicPosAtRecStart. We want sample 0 == intendedStart.
  // Then subtract mic input latency (vocalDelayMs) to pull the voice earlier.
  const positionOffset = meta.musicPosAtRecStart - meta.intendedStart;
  const latency = vocalDelayMs/1000;
  const effOffset = positionOffset - latency;
  let aligned=alignTake(ctx, raw, effOffset);

  let finalBuffer=aligned;
  // A bounded punch-in's recording (pre-roll + the fixed region + a small
  // safety margin) must NOT be stitched onto the previously-kept take like
  // an open-ended punch-in — it needs to stand alone as its own take so it
  // can be assigned to just the fixed region in TE.segments (see the Keep
  // handler). /comp_multi slices to the exact region later regardless of
  // whatever extra pre-roll/margin audio is in this file.
  if(meta.intendedStart>0 && meta.keptAtRecord && !meta.boundedPunch){
    finalBuffer=stitchBuffers(ctx, meta.keptAtRecord, aligned, meta.intendedStart);
    $('renderstatus').textContent='Continued from '+fmt(meta.intendedStart)+'. Vocal timing '+vocalDelayMs+'ms.';
  } else if(meta.boundedPunch){
    $('renderstatus').textContent='Region re-recorded. Vocal timing '+vocalDelayMs+'ms.';
  } else {
    $('renderstatus').textContent='Take captured ('+fmt(aligned.duration)+'). Vocal timing '+vocalDelayMs+'ms — tweak if it drifts.';
  }
  keptBuffer=finalBuffer;
  window._lastVocalBuffer=finalBuffer;
  window._lastVocalWav=encodeWav(finalBuffer);
}
```

- [ ] **Step 6: Auto-assign the new take into the region on Keep**

Change the Keep button handler (`index.html:4789`) from:

```javascript
$('keepBtn').addEventListener('click',()=>{ $('stopChoice').style.display='none'; saveCurrentTakeToHistory(); });
```

to:

```javascript
$('keepBtn').addEventListener('click', async()=>{
  $('stopChoice').style.display='none';
  const region=window._recBoundedPunchRegion;
  await saveCurrentTakeToHistory(region ? 'punch' : undefined);
  if(region && activeTakeId){
    teAssignSegment(activeTakeId, region.start, region.end);
    window._recBoundedPunchRegion=null;
    await teSetCompareMode(true);
    teOpenRegionPanel(region.start, region.end);
    toast('Region re-recorded — build the comp when you\'re happy with it.','ok',4000);
  }
});
```

(`saveCurrentTakeToHistory` already sets `activeTakeId=d.take.id` before returning, per its existing implementation — so `activeTakeId` here is the just-created take, which is exactly what `teAssignSegment` needs. `teSetCompareMode` — added in Task 3 Step 2 — only fills `TE.baseTakeId` when it's still `null`, so if Compare mode was already active with a different base take before this punch-in, that base take is preserved rather than being reset to the fix.)

- [ ] **Step 7: Add a `punch` take-kind label**

Find `TAKE_KIND_LABEL` (`index.html:4910`):

```javascript
const TAKE_KIND_LABEL={lead:'take',comp:'comp',harmony:'harmony'};
```

Change to:

```javascript
const TAKE_KIND_LABEL={lead:'take',comp:'comp',harmony:'harmony',punch:'fix'};
```

- [ ] **Step 8: Manual verification**

1. Restart the server, open the app, load a song, record and Keep a full take.
2. Check "Compare takes", drag a ~3-second region somewhere in the middle of the take's own lane.
3. In the panel, click "🎤 Re-record". Confirm it switches to the Record tab and starts a live punch-in.
4. Sing through it — confirm playback/recording auto-stops close to the region's out-point without you touching Stop (watch the transport UI switch back to idle).
5. Click Keep in the stop-choice dialog. Confirm: a new take appears in the strip labeled "fix" (not "take"); Compare mode is on; the region panel reopens showing the same bounds with `currently:` now pointing at the new "fix" take.
6. Click "↳ Build comp from regions". Confirm success, and that the original full take is still present, untouched, elsewhere in the take strip.
7. Render/export the resulting comp and listen — confirm the fixed region actually contains your new vocal and the rest matches the original take, with no audible seam glitch.
8. If a harmony take exists on this session, confirm it's still present in the take strip, unmodified, and still plays in sync when selected alongside the new comp.

- [ ] **Step 9: Commit**

```bash
git add index.html
git commit -m "$(cat <<'EOF'
Add bounded (auto in/out) punch-in recording for region fixes

The region panel's Re-record button starts a punch-in a couple
seconds before the selected region (reusing the existing open-ended
punch mechanism as pre-roll) and now auto-stops recording once
playback passes the region's out-point. The captured clip is saved
as its own standalone take (kind "punch"/"fix") and auto-assigned
into the region rather than stitched onto the previous take, so
Building immediately produces original-head + fix + original-tail.
EOF
)"
```

---

### Task 6: Reopen an existing comp's regions for further editing

**Files:**
- Modify: `index.html:1444-1448` (markup, add button)
- Modify: `index.html:5080-5099` (`refreshTrackEditor`, toggle button visibility)

**Interfaces:**
- Consumes: `t.comp_segments` from Task 2's persisted shape; `TE.baseTakeId`, `TE.segments` from Task 3/4.
- Produces: nothing new consumed by later tasks (this is the last frontend-logic task before the take-strip visual in Task 7).

- [ ] **Step 1: Add the "Edit this comp's regions" button**

In the markup added in Task 4 Step 1, add a button inside `teRegionPanel`'s sibling area — actually simpler: add it next to `teCompBuildRow` so it's visible regardless of whether a region is currently selected. In `index.html`, change:

```html
          <div class="te-compbuild hidden" id="teCompBuildRow">
            <span id="teSegmentsSummary" class="klabel">Drag on a lane below to assign that take to a region</span>
            <button class="btn small" id="teClearSegments">Clear regions</button>
            <button class="btn" id="teBuildCompBtn">↳ Build comp from regions</button>
          </div>
```

to:

```html
          <div class="te-compbuild hidden" id="teCompBuildRow">
            <span id="teSegmentsSummary" class="klabel">Drag on a lane below to assign that take to a region</span>
            <button class="btn small" id="teClearSegments">Clear regions</button>
            <button class="btn" id="teBuildCompBtn">↳ Build comp from regions</button>
          </div>
          <button class="btn small ghost hidden" id="teEditCompBtn" style="margin-top:8px">✎ Edit this comp's regions</button>
```

- [ ] **Step 2: Wire its click handler and visibility**

Add near the other region-panel JS from Task 4 (anywhere after `teAssignRegionEntry` is defined):

```javascript
$('teEditCompBtn').addEventListener('click', async()=>{
  const t=takes.find(x=>x.id===activeTakeId);
  if(!t || !t.comp_segments) return;
  TE.baseTakeId=t.id;   // set BEFORE entering Compare mode — teSetCompareMode
                         // only fills baseTakeId when it's still null, so this
                         // takes precedence over that fallback
  TE.segments=t.comp_segments.map(s=>
    s.silence ? {silence:true, start:s.start, end:s.end}
              : {takeId:s.take_id, start:s.start, end:s.end});
  await teSetCompareMode(true);
  teDrawSegmentOverlays();
  teUpdateSegmentsSummary();
  toast('Loaded this comp\'s regions — drag to change any of them, then rebuild.','ok',3500);
});
function teUpdateEditCompBtnVisibility(){
  const t=takes.find(x=>x.id===activeTakeId);
  $('teEditCompBtn').classList.toggle('hidden', !(t && t.kind==='comp' && t.comp_segments && t.comp_segments.length));
}
```

- [ ] **Step 3: Call the visibility toggle from `refreshTrackEditor`**

In `refreshTrackEditor` (`index.html:5080`), find:

```javascript
  const tid=teLeadTakeId(activeTakeId || teDefaultTakeId());
  activeTakeId=tid;   // keep the global in sync — teNudgeVocal/teFxRestoreBtn/etc. read it directly
```

Add a call right after it:

```javascript
  const tid=teLeadTakeId(activeTakeId || teDefaultTakeId());
  activeTakeId=tid;   // keep the global in sync — teNudgeVocal/teFxRestoreBtn/etc. read it directly
  teUpdateEditCompBtnVisibility();
```

- [ ] **Step 4: Manual verification**

1. With the comp take built in Task 5's verification still present, select it in the take strip (click its row).
2. Confirm the "✎ Edit this comp's regions" button becomes visible (and is hidden when you select a plain, non-comp take instead).
3. Click it. Confirm Compare mode turns on and the region overlays match the comp's original recipe (same bounds, same source/silence per region as when it was built).
4. Drag a different existing take onto one of the regions, then click "↳ Build comp from regions" again. Confirm a **new** take is created (check its id differs from the one you edited) and the original comp you edited is still present, untouched, in the strip.

- [ ] **Step 5: Commit**

```bash
git add index.html
git commit -m "$(cat <<'EOF'
Let a comp take's regions be reopened and re-edited

Selecting an existing comp take now shows an "Edit this comp's
regions" button that loads its persisted comp_segments recipe back
into the Track Editor, so any one region's source can be swapped and
rebuilt without starting the whole comp over from scratch.
EOF
)"
```

---

### Task 7: Take-strip segment bar (provenance visual)

**Files:**
- Modify: `index.html:4938-4965` (`renderTakeStrip`)
- Modify: CSS near `index.html:445-460` (`.takerow`/`.takebadge`)

**Interfaces:**
- Consumes: `t.comp_segments`, `t.duration` from Task 2's persisted shape.
- Produces: nothing consumed by later tasks (final visual polish task).

- [ ] **Step 1: Add CSS for the segment bar**

Near the existing `.takebadge` rules (`index.html:445-460`), add:

```css
  .takeSegBar{display:flex;height:6px;border-radius:3px;overflow:hidden;flex:none;width:60px}
  .takeSegBar .seg{height:100%}
  .takeSegBar .seg.silence{background:repeating-linear-gradient(45deg,#666 0 3px,#444 3px 6px)}
```

- [ ] **Step 2: Render the bar in `renderTakeStrip`**

Change `index.html:4942-4953` from:

```javascript
  takes.forEach((t,i)=>{
    const row=document.createElement('div');
    row.className='takerow'+(t.id===activeTakeId?' active':'');
    const scoreBadge=t.pitch_score? '<span class="takeScoreBadge" title="Pitch steadiness — % of this take spent landing cleanly on a note, not a check against the song\'s actual melody">'+t.pitch_score.accuracy+'% in tune</span>' : '';
    row.innerHTML=
      '<span class="takebadge '+(t.kind||'lead')+'">'+(TAKE_KIND_LABEL[t.kind]||t.kind||'take')+' '+(i+1)+'</span>'+
      '<span class="takerow-dur">'+fmt(t.duration||0)+'</span>'+
      scoreBadge+
      '<span class="takerow-spacer"></span>'+
      '<span class="takestatus-wrap">'+takeStatusLine(t)+'</span>'+
      (t.id===activeTakeId?'<span class="takerow-active">● editing</span>':'')+
      '<button class="takex" title="Delete this take" type="button">✕</button>';
```

to:

```javascript
  takes.forEach((t,i)=>{
    const row=document.createElement('div');
    row.className='takerow'+(t.id===activeTakeId?' active':'');
    const scoreBadge=t.pitch_score? '<span class="takeScoreBadge" title="Pitch steadiness — % of this take spent landing cleanly on a note, not a check against the song\'s actual melody">'+t.pitch_score.accuracy+'% in tune</span>' : '';
    // A comp with more than one region gets a proportional segment bar so
    // you can see at a glance where it's patched, vs. a plain single-take
    // recording which shows no bar at all.
    let segBar='';
    if(t.kind==='comp' && t.comp_segments && t.comp_segments.length>1 && t.duration){
      const bits=t.comp_segments.map(s=>{
        const pct=Math.max(1, 100*(s.end-s.start)/t.duration);
        return '<span class="seg'+(s.silence?' silence':'')+'" style="width:'+pct.toFixed(2)+'%;background:'+
          (s.silence?'transparent':segColorFor(s.take_id))+'"></span>';
      }).join('');
      segBar='<span class="takeSegBar" title="'+t.comp_segments.length+' regions">'+bits+'</span>';
    }
    row.innerHTML=
      '<span class="takebadge '+(t.kind||'lead')+'">'+(TAKE_KIND_LABEL[t.kind]||t.kind||'take')+' '+(i+1)+'</span>'+
      segBar+
      '<span class="takerow-dur">'+fmt(t.duration||0)+'</span>'+
      scoreBadge+
      '<span class="takerow-spacer"></span>'+
      '<span class="takestatus-wrap">'+takeStatusLine(t)+'</span>'+
      (t.id===activeTakeId?'<span class="takerow-active">● editing</span>':'')+
      '<button class="takex" title="Delete this take" type="button">✕</button>';
```

- [ ] **Step 3: Add the `segColorFor` helper**

Add this function right before `renderTakeStrip` (`index.html:4938`):

```javascript
// Stable, deterministic color per take id, so the same source always shows
// the same color across a comp's segment bar and across renders.
const _segColorCache={};
function segColorFor(takeId){
  if(_segColorCache[takeId]) return _segColorCache[takeId];
  let h=0; for(let i=0;i<takeId.length;i++) h=(h*31+takeId.charCodeAt(i))>>>0;
  const hue=h%360;
  return _segColorCache[takeId]='hsl('+hue+',55%,45%)';
}
```

- [ ] **Step 4: Manual verification**

1. Restart the server, reload the app with the multi-region comp built in Task 6's verification (the one with a swapped-in take).
2. Confirm its row in the take strip shows a small horizontal bar with visibly distinct colored segments proportional to their durations.
3. Confirm a plain (non-comp) take's row shows no bar at all.
4. If any comp includes a silenced region, confirm that segment renders with the hatched style, not a solid color.

- [ ] **Step 5: Commit**

```bash
git add index.html
git commit -m "$(cat <<'EOF'
Show a segmented provenance bar for comp takes in the take strip

A comp built from more than one region now renders a small
proportional, color-coded bar (hatched for silenced spans) next to
its badge in the take strip, so it's visually distinguishable from a
single clean take at a glance.
EOF
)"
```

---

### Task 8: End-to-end verification pass

**Files:** none (verification only).

- [ ] **Step 1: Full happy-path walkthrough**

1. Restart the server fresh. Open the app, load a song.
2. Record a full take, Keep it.
3. Record a harmony take against it, Keep it.
4. In Compare mode, drag a region on the lead take, click Re-record, sing through the auto-stopping punch-in, Keep.
5. Build the comp.
6. Confirm in the take strip: original lead take present, harmony take present and still shows as a harmony (not orphaned), new "fix" take present, new comp take present with a segmented bar.
7. Select the comp as active, confirm the harmony still plays in sync alongside it (Track Editor, non-Compare mode).
8. Export/render the comp + harmony to WAV. Listen to the export: confirm the fixed region sounds like the re-recorded vocal, the rest matches the original, no clicks/pops at the seams, and the harmony is present and in time throughout.

- [ ] **Step 2: Silence-only path**

1. On a different take, drag a region and click Silence (no re-recording).
2. Build the comp, export, and confirm that span is genuinely silent in the vocal while the backing track keeps playing normally underneath.

- [ ] **Step 3: Reopen-and-swap path**

1. Select the comp from Step 1's walkthrough, click "Edit this comp's regions".
2. Change the fixed region to point at the original lead take instead (effectively reverting just that one region).
3. Rebuild, confirm a new take is created and the previous comp is untouched.

- [ ] **Step 4: Check server logs for errors**

```bash
tail -50 /tmp/karaoke-studio.log
```

Expected: no Python tracebacks from any of the above actions.

- [ ] **Step 5: Report results to the user**

Summarize what was tested and what (if anything) didn't behave as expected — do not claim the feature "works" without having actually driven all three paths above in a real browser session against a real recording, per this project's established verify-before-claiming standard.
