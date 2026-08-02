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


# ---- ffmpeg DSP chain -------------------------------------------------------

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


def process_vocal(in_wav, out_wav, fx, tmpdir, progress=None):
    """
    Full vocal chain. Pitch correction (librosa) runs first if requested,
    then the ffmpeg DSP chain (denoise/eq/comp/reverb) on top.

    `progress`, if given, is called with a short stage-name string before
    each step, so a caller (e.g. the HTTP job status) can show real progress
    instead of a single "processing" state.
    """
    pitch = fx.get("pitch") or {}
    do_pitch = bool(pitch.get("enabled"))

    stage_in = in_wav
    if do_pitch:
        if progress:
            progress("correcting_pitch")
        pc_out = os.path.join(tmpdir, "pitched.wav")
        pitch_correct(
            in_wav, pc_out,
            key_root=pitch.get("key", "C"),
            scale=pitch.get("scale", "major"),
            strength=float(pitch.get("strength", 1.0)),
        )
        stage_in = pc_out

    if progress:
        progress("applying_fx")
    apply_ffmpeg_fx(stage_in, out_wav, fx)


# ---- Mixdown ----------------------------------------------------------------

def mixdown(vocal_path, music_path, out_path,
            vocal_gain_db=0.0, music_gain_db=0.0,
            harmony_path=None, harmony_gain_db=-3.0,
            out_format="wav", loudnorm=True):
    """
    Mix processed vocal + backing music (+ optional harmony vocal) into a
    final high-quality file.
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

    inputs = ["-i", vocal_path, "-i", music_path]
    if harmony_path:
        # Harmony bus: same presence treatment as the lead, sat a little
        # lower by default so it supports rather than competes with the lead.
        h_filter = (
            f"volume={harmony_gain_db:.1f}dB,"
            f"equalizer=f=4000:t=q:w=1.2:g=1.5,"
            f"equalizer=f=250:t=q:w=1.0:g=1.0"
        )
        inputs += ["-i", harmony_path]
        filter_complex = (
            f"[0:a]{v_filter},aresample=48000[v];"
            f"[1:a]{m_filter},aresample=48000[m];"
            f"[2:a]{h_filter},aresample=48000[h];"
            f"[v][h][m]amix=inputs=3:duration=longest:normalize=0[mix];"
            f"[mix]{post}[out]"
        )
    else:
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
        *inputs,
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


# ---- Phone-camera sync: chirp alignment + mux --------------------------------
#
# A phone (paired over the LAN) records its own video, remote-triggered close
# to when the app starts its own recording. Network trigger timing is sloppy
# (up to ~1s of jitter), so instead of relying on it we play a short synthetic
# "chirp" (a frequency sweep, not a flat beep — sweeps give a much sharper,
# more noise-robust matched-filter peak than a pure tone) through the computer
# speakers at the instant our own recording starts. The phone's own mic (kept
# on purely as a sync reference, its audio is never used in the final output)
# picks that chirp up. Afterward we matched-filter the phone's captured audio
# against a synthetic copy of the exact same chirp to find precisely when it
# happened in the phone's timeline, then trim + mux the phone's video against
# the app's finished audio mix.

CHIRP_F0 = 800.0          # Hz, sweep start
CHIRP_F1 = 3200.0         # Hz, sweep end
CHIRP_DUR = 0.12          # seconds
CHIRP_ANALYSIS_SR = 44100
CHIRP_CONFIDENCE_MIN = 0.15  # NCC peak in [-1,1]; empirically noise floor is ~0.06-0.08
                              # (200-trial test), real chirps score 0.2+ even when quiet


def _generate_chirp(sr=CHIRP_ANALYSIS_SR, f0=CHIRP_F0, f1=CHIRP_F1, dur=CHIRP_DUR):
    """Synthesize the reference linear-sweep chirp used as the matched-filter
    template. Windowed with raised-cosine on/off ramps so the on/off clicks
    don't smear the correlation peak."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    k = (f1 - f0) / dur
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    sig = np.sin(phase).astype(np.float64)
    ramp = max(1, int(0.1 * n))
    win = np.ones(n)
    win[:ramp] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    win[-ramp:] = win[:ramp][::-1]
    return sig * win


def _extract_audio_for_align(video_path, out_wav, sr=CHIRP_ANALYSIS_SR):
    """Pull the phone's captured audio track out of its video container as
    mono WAV, band-limited to roughly the chirp's sweep range so room noise,
    hum, and singing don't create spurious correlation peaks."""
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr),
           "-af", "highpass=f=500,lowpass=f=4000", out_wav]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("phone audio extraction failed: " + r.stderr[-500:])


def _find_chirp_offset(audio, sr):
    """FFT-based matched filter: cross-correlate `audio` against a synthetic
    reference chirp, then NORMALIZE by local signal energy at each candidate
    lag to get a proper normalized cross-correlation (NCC) in [-1, 1] — the
    standard technique for this (cf. cv2.matchTemplate TM_CCOEFF_NORMED).

    A raw-correlation peak/median ratio is NOT a valid confidence measure
    here: taking a max over hundreds of thousands of noise samples produces
    a peak that's reliably 6-8x the median from extreme-value statistics
    alone, with no real chirp present at all. NCC fixes this because it's a
    bounded, physically meaningful quantity (1.0 = perfect match) regardless
    of how large the search space is.

    Returns (offset_sec, confidence) where confidence is the NCC peak value.
    """
    ref = _generate_chirp(sr=sr)
    n_ref = len(ref)
    n_audio = len(audio)
    valid_starts = n_audio - n_ref + 1
    if valid_starts <= 0:
        return -1.0, 0.0

    n = n_audio + n_ref - 1
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.rfft(audio, nfft)
    R = np.fft.rfft(ref[::-1], nfft)  # time-reversed ref: convolution == correlation
    corr = np.fft.irfft(A * R, nfft)[:n]
    # only lags where the reference window fully overlaps the audio
    corr_valid = corr[n_ref - 1: n_ref - 1 + valid_starts]

    # sliding-window local energy at each candidate start (O(N) via cumsum)
    csum = np.concatenate(([0.0], np.cumsum(audio.astype(np.float64) ** 2)))
    local_energy = csum[n_ref:n_ref + valid_starts] - csum[:valid_starts]
    ref_norm = float(np.sqrt(np.sum(ref ** 2)))
    ncc = corr_valid / (ref_norm * np.sqrt(np.maximum(local_energy, 1e-12)) + 1e-9)

    peak_idx = int(np.argmax(ncc))
    offset_sec = peak_idx / sr
    confidence = float(ncc[peak_idx])
    return offset_sec, confidence


def align_and_mux_video(phone_video_path, final_audio_path, out_video_path,
                         chirp_music_pos_sec, tmp_dir):
    """
    1. Extract the phone's own captured audio (contains the sync chirp).
    2. Matched-filter it to find when the chirp happened in the PHONE's own
       timeline (`offset_sec`).
    3. `chirp_music_pos_sec` is where the chirp landed in the finished mix's
       timeline (song time, captured client-side when the chirp was played).
    4. Trim the phone's video to start at `offset_sec` AND trim the finished
       mix to start at `chirp_music_pos_sec`, then mux. Trimming only the
       video (and leaving the full-length mix) would leave a constant sync
       error equal to chirp_music_pos_sec — trimming both is what actually
       removes it.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    extracted = os.path.join(tmp_dir, "phone_audio_for_align.wav")
    _extract_audio_for_align(phone_video_path, extracted)

    sr = CHIRP_ANALYSIS_SR
    audio, _ = sf.read(extracted, dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    offset_sec, confidence = _find_chirp_offset(audio, sr)
    if offset_sec < 0 or confidence < CHIRP_CONFIDENCE_MIN:
        raise RuntimeError(
            f"Couldn't confidently locate the sync chirp in the phone's audio "
            f"(offset={offset_sec:.3f}s, confidence={confidence:.1f}). "
            f"The phone video may be too quiet/noisy to sync automatically."
        )

    music_pos = max(0.0, float(chirp_music_pos_sec))
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{offset_sec:.3f}", "-i", phone_video_path,
        "-ss", f"{music_pos:.3f}", "-i", final_audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_video_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("video mux failed: " + r.stderr[-500:])
