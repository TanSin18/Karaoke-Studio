#!/usr/bin/env python3
"""
Audio effects + pitch correction for the karaoke studio.

Chain (all offline, best quality):
  denoise -> de-ess -> EQ -> compression -> pitch correction -> reverb/echo

Uses ffmpeg for the DSP filter chain and librosa for musical pitch correction.
"""

import os
import subprocess
import tempfile

import numpy as np
import librosa
import soundfile as sf


# ---- Musical note helpers ---------------------------------------------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Semitone offsets (from C) that belong to each scale.
SCALES = {
    "chromatic": list(range(12)),
    "major":     [0, 2, 4, 5, 7, 9, 11],
    "minor":     [0, 2, 3, 5, 7, 8, 10],   # natural minor
}


def _key_to_pitch_classes(key_root, scale):
    """Return the set of allowed pitch classes (0-11) for key/scale."""
    try:
        root = NOTE_NAMES.index(key_root)
    except ValueError:
        root = 0
    return sorted({(root + s) % 12 for s in SCALES.get(scale, SCALES["chromatic"])})


def _snap_midi_to_scale(midi, allowed_pcs):
    """Snap a (possibly fractional) MIDI note to nearest allowed pitch class."""
    if midi is None or np.isnan(midi):
        return None
    base = int(round(midi))
    best = None
    best_dist = 99
    # search a small window around the note for the nearest allowed pitch class
    for cand in range(base - 6, base + 7):
        if (cand % 12) in allowed_pcs:
            d = abs(cand - midi)
            if d < best_dist:
                best_dist = d
                best = cand
    return best


# ---- Pitch correction -------------------------------------------------------

def pitch_correct(in_wav, out_wav, key_root="C", scale="major", strength=1.0):
    """
    Auto-Tune-style correction:
      1. detect f0 per frame (pYIN)
      2. snap each voiced frame to nearest in-key note
      3. resynthesize with a time-varying pitch shift, rendered with the
         external `rubberband` CLI which handles a per-time pitch map cleanly
         and preserves formants (natural voice).

    strength 0..1 blends between original pitch (0) and fully snapped (1).

    Implementation note: we render the correction as a sequence of constant-
    pitch segments (one per detected note region) using rubberband, then
    concatenate. This avoids the pitch smearing that per-frame overlap-add
    with librosa.pitch_shift introduces, and it snaps to the true target.
    """
    y, sr = librosa.load(in_wav, sr=None, mono=True)
    if y.size == 0:
        sf.write(out_wav, y, sr)
        return

    allowed = _key_to_pitch_classes(key_root, scale)

    frame_length = 2048
    hop = 256

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C6")),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop,
    )

    n_frames = len(f0)
    # per-frame semitone correction
    correction = np.zeros(n_frames)
    for i in range(n_frames):
        if voiced_flag[i] and f0[i] and not np.isnan(f0[i]):
            midi = librosa.hz_to_midi(f0[i])
            target = _snap_midi_to_scale(midi, allowed)
            if target is not None:
                correction[i] = (target - midi) * strength

    # Smooth so segment boundaries don't chatter.
    if n_frames >= 7:
        kernel = np.ones(7) / 7.0
        correction = np.convolve(correction, kernel, mode="same")

    # Group consecutive frames with (nearly) the same correction into segments,
    # then pitch-shift each segment by a single constant amount.
    seg_out = []
    i = 0
    win_samps = hop
    while i < n_frames:
        j = i
        c = correction[i]
        # extend segment while correction stays within 0.15 semitone of start
        while j < n_frames and abs(correction[j] - c) < 0.15:
            j += 1
        seg_semi = float(np.mean(correction[i:j]))
        start = i * hop
        end = min(len(y), j * hop)
        seg = y[start:end]
        if seg.size:
            if abs(seg_semi) > 0.02:
                seg = _rubberband_shift(seg, sr, seg_semi)
            seg_out.append(seg)
        i = j

    if seg_out:
        out = np.concatenate(seg_out)
    else:
        out = y

    # length-match to original (rubberband preserves length, but guard anyway)
    if len(out) < len(y):
        out = np.pad(out, (0, len(y) - len(out)))
    else:
        out = out[:len(y)]

    peak = np.max(np.abs(out)) or 1.0
    if peak > 0.99:
        out = out * (0.99 / peak)

    sf.write(out_wav, out.astype(np.float32), sr)


def _rubberband_shift(seg, sr, semitones):
    """Pitch-shift a mono segment by `semitones` using the rubberband CLI.

    Falls back to librosa if rubberband is unavailable. rubberband gives
    cleaner, formant-preserving shifts than the phase-vocoder path.
    """
    import shutil
    if shutil.which("rubberband") is None:
        return librosa.effects.pitch_shift(seg, sr=sr, n_steps=semitones)

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.wav")
        dst = os.path.join(d, "d.wav")
        sf.write(src, seg.astype(np.float32), sr)
        # -F preserves formants (keeps the voice natural), -c 6 = best quality
        r = subprocess.run(
            ["rubberband", "-p", f"{semitones:.4f}", "-F", "-c", "6", src, dst],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not os.path.exists(dst):
            return librosa.effects.pitch_shift(seg, sr=sr, n_steps=semitones)
        out, _ = sf.read(dst, dtype="float32")
        if out.ndim > 1:
            out = out[:, 0]
        return out


# ---- Pro vocal chain (Clean EQ -> Fast comp -> Smooth comp -> De-esser ->
#      Tone EQ -> Saturation -> Reverb/Delay AUX sends) ----------------------
#
# A fuller, adjustable studio chain modeled on a standard male-vocal signal
# path. Every stage exposes the parameters a hardware/plugin chain would, with
# defaults tuned for a mic already gain-staged to -18..-12 dBFS. Values are
# looked up from the fx["pro"] dict with these defaults as fallback, so the
# frontend only needs to send the parameters the user actually changed.
PRO_CHAIN_DEFAULTS = {
    # 1.1 clean EQ: rumble HPF + low-mid mud dip + nasal/boxiness cut
    "hpf_freq": 85,
    "dip_freq": 250, "dip_gain": -2.5, "dip_q": 1.4,
    "nasal_freq": 900, "nasal_gain": -1.5, "nasal_q": 2.0,
    # 2.2 fast peak compressor (FET/1176 style)
    "fast_ratio": 4, "fast_attack": 1, "fast_release": 80, "fast_threshold_db": -18,
    # 3.3 smooth body compressor (opto/LA-2A style)
    "smooth_ratio": 3, "smooth_attack": 25, "smooth_release": 500, "smooth_threshold_db": -20,
    # 4.4 de-esser
    "deess_freq": 6000, "deess_intensity": 40,
    # 5.5 additive tone EQ: chest warmth + presence + air shelf
    "tone1_freq": 180, "tone1_gain": 1.0, "tone1_q": 1.0,
    "tone2_freq": 3800, "tone2_gain": 2.0, "tone2_q": 1.2,
    "air_freq": 10500, "air_gain": 2.5,
    # 6.6 tube/tape saturation (parallel blend)
    "sat_drive_db": 8, "sat_mix": 12,
    # AUX 1: plate reverb send
    "verb_decay": 1.8, "verb_predelay": 35, "verb_lowcut": 200, "verb_highcut": 6000, "verb_send_db": -3,
    # AUX 2: slap/echo delay send (time is absolute ms — the app doesn't
    # detect song tempo, so this isn't beat-synced like "1/8 dotted")
    "delay_time_ms": 350, "delay_feedback": 13, "delay_lowcut": 300, "delay_highcut": 4000, "delay_send_db": -20,
}


def _pv(pro, key):
    v = pro.get(key)
    return float(v) if v is not None else float(PRO_CHAIN_DEFAULTS[key])


def _db_to_lin(db):
    return 10 ** (db / 20)


def build_pro_chain_filter_complex(pro, sample_rate=48000):
    """
    Build an ffmpeg -filter_complex graph implementing the pro vocal chain.
    Returns (filter_complex_string, output_pad_label).
    """
    g = lambda k: _pv(pro, k)

    hpf = g("hpf_freq")
    dip_f, dip_g, dip_q = g("dip_freq"), g("dip_gain"), g("dip_q")
    nas_f, nas_g, nas_q = g("nasal_freq"), g("nasal_gain"), g("nasal_q")

    fast_ratio, fast_atk, fast_rel = g("fast_ratio"), g("fast_attack"), g("fast_release")
    fast_thr = max(0.000976, min(1, _db_to_lin(g("fast_threshold_db"))))

    smooth_ratio, smooth_atk, smooth_rel = g("smooth_ratio"), g("smooth_attack"), g("smooth_release")
    smooth_thr = max(0.000976, min(1, _db_to_lin(g("smooth_threshold_db"))))

    deess_freq_ratio = max(0.01, min(1, g("deess_freq") / (sample_rate / 2)))
    deess_i = max(0, min(1, g("deess_intensity") / 100))

    tone1_f, tone1_g, tone1_q = g("tone1_freq"), g("tone1_gain"), g("tone1_q")
    tone2_f, tone2_g, tone2_q = g("tone2_freq"), g("tone2_gain"), g("tone2_q")
    air_f, air_g = g("air_freq"), g("air_gain")

    sat_drive = g("sat_drive_db")
    sat_mix = max(0, min(100, g("sat_mix"))) / 100

    verb_decay = max(0.1, g("verb_decay"))
    verb_predelay = max(0, g("verb_predelay"))
    verb_lo, verb_hi = g("verb_lowcut"), g("verb_highcut")
    verb_send = _db_to_lin(g("verb_send_db"))

    delay_ms = max(1, g("delay_time_ms"))
    delay_fb = max(0, min(0.9, g("delay_feedback") / 100))
    delay_lo, delay_hi = g("delay_lowcut"), g("delay_highcut")
    delay_send = _db_to_lin(g("delay_send_db"))

    # 1) linear pre-chain: clean EQ -> fast comp -> smooth comp -> de-esser -> tone EQ
    pre = (
        f"highpass=f={hpf:.1f}:poles=2,highpass=f={hpf:.1f}:poles=1,"
        f"equalizer=f={dip_f:.1f}:t=q:w={dip_q:.2f}:g={dip_g:.2f},"
        f"equalizer=f={nas_f:.1f}:t=q:w={nas_q:.2f}:g={nas_g:.2f},"
        f"acompressor=threshold={fast_thr:.5f}:ratio={fast_ratio:.2f}:attack={fast_atk:.2f}:"
        f"release={fast_rel:.2f}:knee=6:makeup=1.6,"
        f"acompressor=threshold={smooth_thr:.5f}:ratio={smooth_ratio:.2f}:attack={smooth_atk:.2f}:"
        f"release={smooth_rel:.2f}:knee=8:makeup=2.0,"
        f"deesser=f={deess_freq_ratio:.3f}:i={deess_i:.2f}:m=0.5,"
        f"equalizer=f={tone1_f:.1f}:t=q:w={tone1_q:.2f}:g={tone1_g:.2f},"
        f"equalizer=f={tone2_f:.1f}:t=q:w={tone2_q:.2f}:g={tone2_g:.2f},"
        f"treble=f={air_f:.1f}:g={air_g:.2f}"
    )

    # 2) saturation: parallel blend of dry + soft-clipped wet (10-15% wet by default)
    dry_w = max(0.0001, 1 - sat_mix)
    wet_w = max(0.0001, sat_mix)

    # 3) reverb AUX: predelay -> multi-tap decaying echo (stands in for a true
    #    IR reverb, no external impulse needed) -> band-limit -> send level
    rt = verb_decay / 1.8
    taps_delays = "|".join(str(x) for x in (29, 47, 71, 97))
    taps_decays = "|".join(f"{min(0.95, w * rt):.3f}" for w in (0.5, 0.4, 0.3, 0.25))
    verb_chain = (
        f"adelay={verb_predelay:.0f}|{verb_predelay:.0f},"
        f"aecho=0.85:0.9:{taps_delays}:{taps_decays},"
        f"highpass=f={verb_lo:.1f},lowpass=f={verb_hi:.1f},"
        f"volume={verb_send:.4f}"
    )

    # 4) delay AUX: slap/echo -> band-limit ("telephone" tone) -> send level
    delay_chain = (
        f"aecho=0.9:0.9:{delay_ms:.0f}:{delay_fb:.3f},"
        f"highpass=f={delay_lo:.1f},lowpass=f={delay_hi:.1f},"
        f"volume={delay_send:.4f}"
    )

    fc = (
        f"[0:a]{pre}[pre];"
        f"[pre]asplit=2[sdry][ssat];"
        f"[ssat]volume={sat_drive:.1f}dB,asoftclip=type=tanh,volume={-sat_drive * 0.7:.2f}dB[ssatw];"
        f"[sdry][ssatw]amix=inputs=2:weights={dry_w:.3f} {wet_w:.3f}:normalize=0[postsat];"
        f"[postsat]asplit=3[mdry][mverb][mdel];"
        f"[mverb]{verb_chain}[verbwet];"
        f"[mdel]{delay_chain}[delwet];"
        f"[mdry][verbwet][delwet]amix=inputs=3:weights=1 1 1:normalize=0[outfx]"
    )
    return fc, "outfx"


# ---- ffmpeg DSP chain (legacy simple chain, used if fx has no "pro" dict) ---

def build_ffmpeg_chain(fx):
    """
    Build an ffmpeg -af filter string from an fx dict.
    fx keys (all optional, all with sensible defaults):
      denoise: 0..1
      deess: bool
      eq: {low: dB, mid: dB, high: dB}
      compress: 0..1
      reverb: 0..1        (wet amount)
      echo: 0..1
      gain_db: float
    """
    parts = []

    denoise = float(fx.get("denoise", 0) or 0)
    if denoise > 0:
        # afftdn noise reduction; nr scales with strength
        nr = 6 + denoise * 24  # 6..30 dB
        parts.append(f"afftdn=nr={nr:.1f}:nf=-25")

    # --- always-on studio smoothing (gentle, makes the vocal sit better) ---
    # rumble/hp, a light de-harsh at 3 kHz, and a touch of de-ess. These run
    # regardless of toggles so the recorded vocal never sounds raw/harsh.
    parts.append("highpass=f=80")
    parts.append("equalizer=f=3000:t=q:w=1.4:g=-2.5")   # de-harsh the 3k bite
    deess_amt = 0.5 if fx.get("deess") else 0.25
    parts.append(f"deesser=i={deess_amt:.2f}")

    eq = fx.get("eq") or {}
    low = float(eq.get("low", 0) or 0)
    mid = float(eq.get("mid", 0) or 0)
    high = float(eq.get("high", 0) or 0)
    if abs(low) > 0.01:
        parts.append(f"equalizer=f=120:t=q:w=1.0:g={low:.1f}")
    if abs(mid) > 0.01:
        parts.append(f"equalizer=f=1500:t=q:w=1.2:g={mid:.1f}")
    if abs(high) > 0.01:
        parts.append(f"equalizer=f=8000:t=q:w=1.0:g={high:.1f}")

    # --- compression: smooths the "throw" (loud pushes). Always at least gentle. ---
    compress = float(fx.get("compress", 0) or 0)
    compress = max(compress, 0.35)   # floor so peaks are always evened out a bit
    ratio = 2 + compress * 4         # 2:1 .. 6:1
    thr = 0.15 - compress * 0.1      # 0.15 .. 0.05
    # softer knee + smoother attack/release for a natural, un-pumped sound
    parts.append(
        f"acompressor=threshold={thr:.3f}:ratio={ratio:.1f}:knee=6:"
        f"attack=8:release=180:makeup=2.5"
    )

    echo = float(fx.get("echo", 0) or 0)
    if echo > 0:
        decay = min(0.6, 0.2 + echo * 0.4)
        parts.append(f"aecho=0.8:0.9:{int(60+echo*180)}:{decay:.2f}")

    reverb = float(fx.get("reverb", 0) or 0)
    if reverb > 0:
        # Lightweight reverb via multi-tap echo (no external impulse needed).
        d = reverb
        taps_delays = "|".join(str(x) for x in [29, 47, 71, 97])
        taps_decays = "|".join(f"{d*w:.2f}" for w in [0.5, 0.4, 0.3, 0.25])
        parts.append(f"aecho=0.85:0.9:{taps_delays}:{taps_decays}")

    gain = float(fx.get("gain_db", 0) or 0)
    if abs(gain) > 0.01:
        parts.append(f"volume={gain:.1f}dB")

    if not parts:
        parts.append("anull")

    return ",".join(parts)


def apply_ffmpeg_fx(in_path, out_path, fx, sample_rate=48000):
    pro = fx.get("pro")
    if pro is not None and pro.get("enabled", True):
        filter_complex, out_label = build_pro_chain_filter_complex(pro, sample_rate)
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-filter_complex", filter_complex,
            "-map", f"[{out_label}]",
            "-ar", str(sample_rate),
            "-ac", "2",
            "-c:a", "pcm_s24le",  # 24-bit high quality intermediate
            out_path,
        ]
    else:
        chain = build_ffmpeg_chain(fx)
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-af", chain,
            "-ar", str(sample_rate),
            "-ac", "2",
            "-c:a", "pcm_s24le",  # 24-bit high quality intermediate
            out_path,
        ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg fx failed: " + r.stderr[-500:])


def process_vocal(in_wav, out_wav, fx, tmpdir):
    """
    Full vocal chain. Pitch correction (librosa) runs first if requested,
    then the ffmpeg DSP chain (denoise/eq/comp/reverb) on top.
    """
    pitch = fx.get("pitch") or {}
    do_pitch = bool(pitch.get("enabled"))

    stage_in = in_wav
    if do_pitch:
        pc_out = os.path.join(tmpdir, "pitched.wav")
        pitch_correct(
            in_wav, pc_out,
            key_root=pitch.get("key", "C"),
            scale=pitch.get("scale", "major"),
            strength=float(pitch.get("strength", 1.0)),
        )
        stage_in = pc_out

    apply_ffmpeg_fx(stage_in, out_wav, fx)


# ---- Mixdown ----------------------------------------------------------------

def mixdown(vocal_path, music_path, out_path,
            vocal_gain_db=0.0, music_gain_db=0.0,
            out_format="wav", loudnorm=True):
    """
    Mix processed vocal + backing music into a final high-quality file.
    """
    # Vocal bus: gain + a gentle presence lift so the voice sits ON TOP of the
    # music instead of buried in it (a common "lack of quality" cause).
    v_filter = (
        f"volume={vocal_gain_db:.1f}dB,"
        f"equalizer=f=4000:t=q:w=1.2:g=1.5,"     # air/presence
        f"equalizer=f=250:t=q:w=1.0:g=1.0"        # warmth
    )
    m_filter = f"volume={music_gain_db:.1f}dB"

    # Glue bus compressor + loudness normalise for a cohesive, "produced" master.
    glue = "acompressor=threshold=0.1:ratio=2:knee=6:attack=15:release=250:makeup=1"
    post = (glue + ",loudnorm=I=-14:TP=-1.5:LRA=11") if loudnorm else glue

    filter_complex = (
        f"[0:a]{v_filter},aresample=48000[v];"
        f"[1:a]{m_filter},aresample=48000[m];"
        f"[v][m]amix=inputs=2:duration=longest:normalize=0[mix];"
        f"[mix]{post}[out]"
    )

    codec = {
        "wav":  ["-c:a", "pcm_s24le"],
        "flac": ["-c:a", "flac", "-compression_level", "8"],
        "mp3":  ["-c:a", "libmp3lame", "-q:a", "0"],
    }.get(out_format, ["-c:a", "pcm_s24le"])

    cmd = [
        "ffmpeg", "-y",
        "-i", vocal_path,
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ar", "48000", "-ac", "2",
    ] + codec + [out_path]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("mixdown failed: " + r.stderr[-500:])


# ---- Backing-track transpose (change musical key) ---------------------------

def transpose_backing(in_wav, out_wav, semitones):
    """
    Shift the backing track up/down by `semitones` WITHOUT changing tempo,
    so the song stays the same length and speed but in a new key.

    Uses the rubberband CLI (formant-agnostic here; it's music, not voice).
    A shift of 0 just copies the file.
    """
    semitones = float(semitones)
    if abs(semitones) < 0.01:
        # no shift — straight copy so the caller always has a file to serve
        import shutil
        shutil.copyfile(in_wav, out_wav)
        return

    import shutil
    if shutil.which("rubberband") is not None:
        r = subprocess.run(
            ["rubberband", "-p", f"{semitones:.4f}", "-c", "6", in_wav, out_wav],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and os.path.exists(out_wav):
            return
        # fall through to ffmpeg if rubberband failed

    # ffmpeg fallback using its built-in rubberband filter
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", in_wav,
         "-af", f"rubberband=pitch=2^({semitones}/12)",
         "-c:a", "pcm_s24le", out_wav],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError("transpose failed: " + r.stderr[-400:])


# ---- Punch-in: stitch a kept head with a re-sung tail -----------------------

def stitch_vocals(head_wav, tail_wav, out_wav, punch_sec, crossfade_ms=40):
    """
    Build one vocal track = head[0..punch] + tail[punch..end], joined with a
    short equal-power crossfade so there's no click at the seam.

    head_wav  : the previously-kept full vocal take (we use 0..punch of it)
    tail_wav  : the newly recorded take that STARTS at `punch_sec` in song time
                (i.e. its sample 0 corresponds to song time punch_sec)
    punch_sec : where the redo begins, in seconds
    """
    head, sr = librosa.load(head_wav, sr=None, mono=True)
    tail, sr2 = librosa.load(tail_wav, sr=sr, mono=True)  # resample tail to head sr

    punch = int(punch_sec * sr)
    xf = max(1, int(crossfade_ms / 1000.0 * sr))

    punch = min(punch, len(head))
    head_part = head[:punch]

    # equal-power crossfade over the boundary
    if punch >= xf and len(tail) >= xf and len(head) >= punch:
        # overlap region: last xf of head_part vs first xf of tail
        fade_out = np.cos(np.linspace(0, np.pi / 2, xf)) ** 1
        fade_in = np.sin(np.linspace(0, np.pi / 2, xf)) ** 1
        head_tailend = head_part[-xf:] * fade_out
        tail_start = tail[:xf] * fade_in
        joined_mid = head_tailend + tail_start
        out = np.concatenate([head_part[:-xf], joined_mid, tail[xf:]])
    else:
        out = np.concatenate([head_part, tail])

    peak = np.max(np.abs(out)) or 1.0
    if peak > 0.99:
        out = out * (0.99 / peak)

    sf.write(out_wav, out.astype(np.float32), sr)
