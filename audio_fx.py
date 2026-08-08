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
    # 0.0 breath control: a soft downward expander sitting BEFORE the
    # compressors — the compressors otherwise do the opposite of what a
    # listener wants with breaths: they turn the loud parts down, which
    # RAISES every inhale/exhale relative to the singing. Deliberately an
    # expander, not a hard gate: a gate that fully mutes between words
    # audibly "chops" tails of phrases and pumps. gate_range_db caps how
    # far breaths get pushed down (-14dB reads as natural quiet, not a
    # hole); threshold sits well under sung level (~-18..-12 dBFS) so soft
    # phrase endings pass untouched.
    # Tuned against real takes (see session d56a078b95f6): -28dB/peak
    # detection cuts breaths ~8-10dB while leaving sung frames at 0.0dB and
    # phrase tails at ~0.3dB of reduction; RMS detection at -37 only
    # managed ~5dB on breaths because their short peaks ride over an RMS
    # average.
    "gate_threshold_db": -28, "gate_ratio": 3.0, "gate_range_db": -14,
    "gate_attack": 4, "gate_release": 260,
    # 1.1 clean EQ: rumble HPF + low-mid mud dip + nasal/boxiness cut
    "hpf_freq": 85,
    "dip_freq": 250, "dip_gain": -2.5, "dip_q": 1.4,
    "nasal_freq": 900, "nasal_gain": -1.5, "nasal_q": 2.0,
    # 2.2 fast peak compressor (FET/1176 style). Thresholds sit a bit above
    # where a well-gain-staged vocal actually lives (-18..-12 dBFS per the
    # module docstring) on purpose — at -18/-20 both compressors were almost
    # always in some amount of gain reduction, constantly shaving off the
    # exact micro-dynamics (breath, consonant transients) that read as a
    # voice's natural "grain"; catching only real peaks preserves that while
    # still providing a safety net.
    "fast_ratio": 4, "fast_attack": 1, "fast_release": 80, "fast_threshold_db": -15,
    # 3.3 smooth body compressor (opto/LA-2A style). Threshold sits lower
    # than the fast comp's on purpose: this stage is the LEVELER — it's what
    # brings soft low phrases up toward the belted ones (the "low notes get
    # lost, high notes blast" complaint is a dynamic-range problem, and this
    # plus the bus limiter is the pair that closes that range from both
    # ends).
    "smooth_ratio": 3.2, "smooth_attack": 25, "smooth_release": 450, "smooth_threshold_db": -23,
    "smooth_makeup": 2.6,
    # 4.4 de-esser
    "deess_freq": 6000, "deess_intensity": 40,
    # 4.5 dynamic harshness tamer: a bell cut around 3.4kHz that only
    # engages when that band gets HOT (belted/shouted notes) — loud high
    # notes lose their icepick edge while soft singing keeps full presence.
    # A static EQ cut here would dull the whole vocal; dynamic is the point.
    "harsh_freq": 3400, "harsh_threshold": 0.10, "harsh_ratio": 2.5,
    "harsh_range_db": 8, "harsh_attack": 6, "harsh_release": 140,
    # 5.5 additive tone EQ: chest warmth + presence + air shelf
    "tone1_freq": 180, "tone1_gain": 1.0, "tone1_q": 1.0,
    "tone2_freq": 3800, "tone2_gain": 2.0, "tone2_q": 1.2,
    "air_freq": 10500, "air_gain": 2.5,
    # 6.6 tube/tape saturation (parallel blend)
    "sat_drive_db": 8, "sat_mix": 12,
    # AUX 1: plate reverb send. -3dB (~71% linear) was too hot for a default —
    # it washed vocals out rather than just adding polish; -9dB (~35%) is a
    # much safer "gold standard" amount of ambience out of the box.
    "verb_decay": 1.8, "verb_predelay": 35, "verb_lowcut": 200, "verb_highcut": 6000, "verb_send_db": -9,
    # AUX 2: slap/echo delay send (time is absolute ms — the app doesn't
    # detect song tempo, so this isn't beat-synced like "1/8 dotted")
    "delay_time_ms": 350, "delay_feedback": 13, "delay_lowcut": 300, "delay_highcut": 4000, "delay_send_db": -20,
    # AUX 3: vocal doubler/thickener — a short, slow, shallow chorus blended
    # in underneath the dry vocal (the classic ADT trick: makes one take
    # sound like two voices in unison). Kept short/slow/shallow on purpose —
    # push these too far and it turns into a swirly chorus effect rather
    # than a subtle "double".
    "double_delay_ms": 18, "double_speed": 0.15, "double_depth": 1.2, "double_send_db": -8,
    # 7.7 parallel "punch" compression: a heavily squashed copy blended back
    # UNDER the main vocal for extra perceived loudness/density without
    # flattening the main path's own dynamics (the "New York compression"
    # trick — a separate ADD-only layer, not a replacement for the fast/
    # smooth compressors above).
    "punch_ratio": 8, "punch_attack": 3, "punch_release": 150, "punch_threshold_db": -30,
    "punch_makeup": 6.0, "punch_mix": 20,
    # 8.8 vocal-bus limiter, the chain's very last stage: catches belted/
    # shouted notes that sail past both compressors (a real shout can hit
    # the file at 0dBFS; 4:1 at -15 still leaves it ~7dB above the rest of
    # the performance AND audibly hard/brittle). A fast lookahead-style
    # limiter at -2dBFS turns "blast" into "loud", which is what a produced
    # vocal does. Applied after the aux mix so reverb/delay swells can't
    # spike past it either.
    "limit_ceiling_db": -2.0, "limit_attack": 2, "limit_release": 90,
}

# Genre/production-style presets: each is a set of overrides applied ON TOP
# OF PRO_CHAIN_DEFAULTS (any key not listed here keeps its default value) —
# selecting one is meant to be a full, reproducible reset to that style, not
# a nudge from wherever the sliders currently sit.
PRO_CHAIN_PRESETS = {
    "clean": {
        "label": "Clean & Natural",
        "desc": "Just enough polish to sound intentional. Closest to your own voice.",
        "values": {},
    },
    "pop_radio": {
        "label": "Pop / Radio",
        "desc": "Bright, upfront, and loud — tighter compression, more presence/air, "
                "noticeable doubling and punch.",
        "values": {
            "dip_gain": -3.5, "nasal_gain": -2.0,
            "fast_ratio": 5, "fast_threshold_db": -20,
            "smooth_ratio": 4, "smooth_threshold_db": -22,
            "deess_intensity": 55,
            "tone2_gain": 3.0, "air_gain": 3.5,
            "sat_drive_db": 10, "sat_mix": 18,
            "double_send_db": -6, "punch_mix": 35,
            "verb_send_db": -10, "delay_send_db": -22,
        },
    },
    "warm_intimate": {
        "label": "Warm & Intimate",
        "desc": "Close and rich — gentle compression, extra low-mid warmth, a longer "
                "smoother reverb tail. Acoustic ballad, not arena.",
        "values": {
            "tone1_gain": 2.5, "tone2_gain": 0.5, "air_gain": 1.0,
            "sat_drive_db": 6, "sat_mix": 20,
            "smooth_ratio": 2.5, "smooth_attack": 40, "smooth_release": 700,
            "verb_decay": 2.4, "verb_send_db": -7, "delay_send_db": -30,
            "double_send_db": -14, "punch_mix": 10,
        },
    },
    "rock_powerful": {
        "label": "Rock / Powerful",
        "desc": "Aggressive and in-your-face — fast, hard-hitting compression, heavier "
                "saturation, and a strong punch layer for a belted, arena-scale vocal.",
        "values": {
            "fast_ratio": 6, "fast_attack": 0.5, "fast_threshold_db": -16,
            "smooth_ratio": 4, "smooth_threshold_db": -18,
            "tone2_gain": 3.5, "air_gain": 2.0,
            "sat_drive_db": 13, "sat_mix": 25,
            "punch_mix": 45, "punch_ratio": 10,
            "double_send_db": -7,
            "verb_send_db": -11, "delay_send_db": -24,
        },
    },
    "acoustic_natural": {
        "label": "Acoustic / Unplugged",
        "desc": "Light-touch processing that stays true to the performance — gentle "
                "compression, subtle everything, a short natural room reverb.",
        "values": {
            "dip_gain": -1.5, "nasal_gain": -1.0,
            "fast_ratio": 2.5, "smooth_ratio": 2,
            "sat_mix": 6, "sat_drive_db": 4,
            "air_gain": 1.5,
            "verb_send_db": -12, "delay_send_db": -34,
            "double_send_db": -18, "punch_mix": 8,
        },
    },
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
    smooth_makeup = max(1.0, min(8.0, g("smooth_makeup")))

    harsh_f = g("harsh_freq")
    harsh_thr = max(0.0, min(100.0, g("harsh_threshold")))
    harsh_ratio = max(1.0, min(30.0, g("harsh_ratio")))
    harsh_range = max(1.0, min(30.0, g("harsh_range_db")))
    harsh_atk = max(0.01, g("harsh_attack"))
    harsh_rel = max(0.01, g("harsh_release"))

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

    double_delay = max(1, g("double_delay_ms"))
    double_speed = max(0.01, g("double_speed"))
    double_depth = max(0.01, g("double_depth"))
    double_send = _db_to_lin(g("double_send_db"))

    punch_ratio, punch_atk, punch_rel = g("punch_ratio"), g("punch_attack"), g("punch_release")
    punch_thr = max(0.000976, min(1, _db_to_lin(g("punch_threshold_db"))))
    punch_makeup = max(1, min(64, g("punch_makeup")))
    punch_mix = max(0, min(100, g("punch_mix"))) / 100

    gate_thr = max(0.0001, min(1, _db_to_lin(g("gate_threshold_db"))))
    gate_ratio = max(1.0, min(9000, g("gate_ratio")))
    gate_range = max(0.0001, min(1, _db_to_lin(g("gate_range_db"))))
    gate_atk = max(0.01, g("gate_attack"))
    gate_rel = max(1, g("gate_release"))

    lim_ceiling = max(0.0625, min(1, _db_to_lin(g("limit_ceiling_db"))))
    lim_atk = max(0.1, g("limit_attack"))
    lim_rel = max(1, g("limit_release"))

    # 1) linear pre-chain: HPF -> breath expander -> clean EQ -> fast comp
    #    -> smooth comp -> de-esser -> tone EQ. The expander sits right
    #    after the HPF (so rumble can't hold it open) and BEFORE both
    #    compressors (which would otherwise bring every breath UP).
    pre = (
        f"highpass=f={hpf:.1f}:poles=2,highpass=f={hpf:.1f}:poles=1,"
        f"agate=threshold={gate_thr:.5f}:ratio={gate_ratio:.2f}:range={gate_range:.4f}:"
        f"attack={gate_atk:.1f}:release={gate_rel:.0f}:knee=3:detection=peak,"
        f"equalizer=f={dip_f:.1f}:t=q:w={dip_q:.2f}:g={dip_g:.2f},"
        f"equalizer=f={nas_f:.1f}:t=q:w={nas_q:.2f}:g={nas_g:.2f},"
        f"acompressor=threshold={fast_thr:.5f}:ratio={fast_ratio:.2f}:attack={fast_atk:.2f}:"
        f"release={fast_rel:.2f}:knee=6:makeup=1.6,"
        f"acompressor=threshold={smooth_thr:.5f}:ratio={smooth_ratio:.2f}:attack={smooth_atk:.2f}:"
        f"release={smooth_rel:.2f}:knee=8:makeup={smooth_makeup:.2f},"
        f"deesser=f={deess_freq_ratio:.3f}:i={deess_i:.2f}:m=0.5,"
        f"adynamicequalizer=mode=cutabove:threshold={harsh_thr:.4f}:"
        f"dfrequency={harsh_f:.0f}:dqfactor=1.2:tfrequency={harsh_f:.0f}:tqfactor=1.1:"
        f"ratio={harsh_ratio:.2f}:range={harsh_range:.1f}:"
        f"attack={harsh_atk:.1f}:release={harsh_rel:.1f},"
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

    # 5) doubler AUX: a short, slow, shallow 2-voice chorus stands in for a
    # second take sung in unison (the classic ADT trick) -> send level. Kept
    # short/slow/shallow so it reads as "doubled", not a swirly chorus.
    double_chain = (
        f"chorus=0.5:0.5:{double_delay:.1f}|{double_delay * 1.25:.1f}:"
        f"0.4|0.32:{double_speed:.3f}|{double_speed * 1.4:.3f}:"
        f"{double_depth:.2f}|{double_depth * 0.75:.2f},"
        f"volume={double_send:.4f}"
    )

    # 6) parallel "punch" bus: a heavily squashed, made-up copy blended back
    # UNDER the dry+saturated signal — the "New York compression" trick for
    # extra perceived density/loudness without flattening the main path's
    # own dynamics. This is an ADD-only layer (not a dry/wet complement like
    # saturation above), so punch_mix doesn't need to sum to 1 with anything.
    punch_chain = (
        f"acompressor=threshold={punch_thr:.5f}:ratio={punch_ratio:.2f}:"
        f"attack={punch_atk:.2f}:release={punch_rel:.2f}:knee=2:makeup={punch_makeup:.2f},"
        f"volume={punch_mix:.3f}"
    )

    fc = (
        f"[0:a]{pre}[pre];"
        f"[pre]asplit=2[sdry][ssat];"
        f"[ssat]volume={sat_drive:.1f}dB,asoftclip=type=tanh,volume={-sat_drive * 0.7:.2f}dB[ssatw];"
        f"[sdry][ssatw]amix=inputs=2:weights={dry_w:.3f} {wet_w:.3f}:normalize=0[postsat0];"
        f"[postsat0]asplit=2[postsat_main][spunch];"
        f"[spunch]{punch_chain}[spunchw];"
        f"[postsat_main][spunchw]amix=inputs=2:weights=1 1:normalize=0[postsat];"
        f"[postsat]asplit=4[mdry][mverb][mdel][mdbl];"
        # dry vocal stays centered — only the AUX sends get stereo-widened,
        # so the voice itself never loses focus/clarity.
        f"[mdry]pan=stereo|c0=c0|c1=c0[drystereo];"
        f"[mverb]{verb_chain},haas[verbwet];"
        f"[mdel]{delay_chain},haas[delwet];"
        f"[mdbl]{double_chain},haas[dblwet];"
        f"[drystereo][verbwet][delwet][dblwet]amix=inputs=4:weights=1 1 1 1:normalize=0[fxsum];"
        # vocal-bus limiter LAST, over the full sum (dry + all sends) — a
        # belted note that both compressors let through gets rounded off
        # at the ceiling instead of blasting/cracking over the music
        f"[fxsum]alimiter=limit={lim_ceiling:.4f}:attack={lim_atk:.1f}:"
        f"release={lim_rel:.0f}:level=false[outfx]"
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
            harmonies=None,
            out_format="wav", loudnorm=True,
            trim_start=None, trim_end=None,
            master_wav_path=None, music_eq=None,
            vocal_align_ms=0.0):
    """
    Mix processed vocal + backing music (+ any number of harmony vocals)
    into a final high-quality file.

    harmonies: optional list of {"path": <wav>, "gain_db": <float>} — as
    many simultaneous extra vocal layers as the caller wants (multiple
    stacked harmonies, not just one), each already FX-processed exactly
    like the lead vocal (process_vocal) by the caller. Empty/None mixes
    just lead + music, same as before this could be more than one.

    master_wav_path: if given (and out_format isn't already "wav"), also
    write a second, lossless copy of the SAME mix to this path, from the
    SAME filter graph run (ffmpeg supports mapping one filter output to
    multiple encoded outputs in one pass — no second decode/process needed).
    This exists for the phone-camera video mux, which used to pull its
    audio from whatever format the singer picked for their own download —
    if that was MP3, the video's audio was a lossy re-encode of an already-
    lossy file, audibly worse than it needed to be. Muxing always prefers
    this master instead, regardless of the download format chosen.

    music_eq: optional {"low_gain","mid_gain","high_gain"} (dB) — the
    karaoke/backing track used to only have a single gain slider, unlike
    the vocal's own tone-shaping EQ; this gives it the same simple 3-band
    control (rumble-range shelf, midrange bell, presence shelf).

    vocal_align_ms: non-destructive vocal-vs-karaoke timing nudge from the
    Track Editor (separate from whatever one-shot mic-latency correction
    already happened at capture time) — positive pushes the vocal later
    (silence-pad the front via adelay), negative pulls it earlier (cut the
    front off via atrim). Small nudges only; this is a drift fix, not a
    general-purpose trim.
    """
    # Vocal bus: gain + a gentle presence lift so the voice sits ON TOP of the
    # music instead of buried in it (a common "lack of quality" cause).
    v_align = ""
    if vocal_align_ms and abs(vocal_align_ms) >= 1:
        if vocal_align_ms > 0:
            v_align = f"adelay={vocal_align_ms:.0f}|{vocal_align_ms:.0f},"
        else:
            v_align = f"atrim=start={abs(vocal_align_ms) / 1000:.3f},asetpts=PTS-STARTPTS,"
    v_filter = (
        f"{v_align}"
        f"volume={vocal_gain_db:.1f}dB,"
        f"equalizer=f=4000:t=q:w=1.2:g=1.5,"     # air/presence
        f"equalizer=f=250:t=q:w=1.0:g=1.0"        # warmth
    )
    music_eq = music_eq or {}
    m_low = float(music_eq.get("low_gain") or 0)
    m_mid = float(music_eq.get("mid_gain") or 0)
    m_high = float(music_eq.get("high_gain") or 0)
    m_filter = f"volume={music_gain_db:.1f}dB"
    if abs(m_low) > 0.01:
        m_filter += f",bass=f=150:g={m_low:.2f}"
    if abs(m_mid) > 0.01:
        m_filter += f",equalizer=f=1000:t=q:w=1.0:g={m_mid:.2f}"
    if abs(m_high) > 0.01:
        m_filter += f",treble=f=6000:g={m_high:.2f}"

    # Glue bus compressor + loudness normalise for a cohesive, "produced" master.
    glue = "acompressor=threshold=0.1:ratio=2:knee=6:attack=15:release=250:makeup=1"
    post = (glue + ",loudnorm=I=-14:TP=-1.5:LRA=11") if loudnorm else glue

    # Optional trim, applied last (after mixing/mastering) so it's sample-
    # accurate against the final mix rather than each input separately.
    # asetpts resets the timestamp base to 0 after cutting the front off —
    # without it the trimmed file would still carry the original start
    # offset and downstream tools (players, the video mux step) would see
    # a file that "starts" at a nonzero timestamp.
    trim = ""
    if trim_start is not None or trim_end is not None:
        start = max(0.0, float(trim_start or 0))
        bound = f"start={start:.3f}"
        if trim_end is not None:
            bound += f":end={max(start, float(trim_end)):.3f}"
        trim = f",atrim={bound},asetpts=PTS-STARTPTS"

    harmonies = harmonies or []
    inputs = ["-i", vocal_path, "-i", music_path]
    parts = [f"[0:a]{v_filter},aresample=48000[v]", f"[1:a]{m_filter},aresample=48000[m]"]
    mix_labels = ["[v]"]
    for i, h in enumerate(harmonies):
        # Harmony bus: same presence treatment as the lead, sat a little
        # lower by default so it supports rather than competes with the lead.
        h_gain = float(h.get("gain_db", -3.0))
        h_filter = (
            f"volume={h_gain:.1f}dB,"
            f"equalizer=f=4000:t=q:w=1.2:g=1.5,"
            f"equalizer=f=250:t=q:w=1.0:g=1.0"
        )
        inputs += ["-i", h["path"]]
        input_idx = 2 + i
        label = f"h{i}"
        parts.append(f"[{input_idx}:a]{h_filter},aresample=48000[{label}]")
        mix_labels.append(f"[{label}]")
    mix_labels.append("[m]")
    n_inputs = len(mix_labels)
    filter_complex = (
        ";".join(parts) + ";" +
        "".join(mix_labels) + f"amix=inputs={n_inputs}:duration=longest:normalize=0[mix];"
        f"[mix]{post}{trim}[out]"
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

    if master_wav_path and out_format != "wav":
        cmd += ["-map", "[out]", "-ar", "48000", "-ac", "2",
                "-c:a", "pcm_s24le", master_wav_path]

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


# ---- Punch-in: comp an ordered list of take/region segments into one track -

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
    """
    if not segments:
        raise ValueError("no segments to comp")

    sr = None
    parts = []
    for seg in segments:
        y, this_sr = librosa.load(seg["path"], sr=sr, mono=True)
        if sr is None:
            sr = this_sr
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


# ---- Phone-camera video: mux -------------------------------------------------
#
# The desktop records the phone's live WebRTC video stream itself, started in
# the same tick as the vocal recording — both are on one clock from t=0, so
# there's no separate alignment step needed here, just muxing the video
# against the finished audio mix. (An earlier design had the phone record
# and upload independently, using a synthetic audio "chirp" the phone's own
# mic would pick up through a real speaker to find the offset after the
# fact — dropped because it silently failed for anyone monitoring through
# headphones, and the necessary full-length upload from the phone was
# unreliable at real recording sizes.)

def mux_video(phone_video_path, final_audio_path, out_video_path, trim_start=None):
    # Frame-accurate output benefits from re-encoding rather than a stream
    # copy. This runs once, after recording, in a background thread — no
    # real-time constraint — so "slow"/crf 18 is worth it over faster,
    # lower-quality presets.
    #
    # trim_start: if the render trimmed the front off the final mix,
    # final_audio_path's timeline now starts at 0 where the untrimmed
    # phone video still starts at the original t=0 — seek the video input
    # forward by the same amount so the two stay aligned. The end doesn't
    # need separate handling: -shortest already cuts to whichever stream
    # (now the trimmed, shorter audio) ends first.
    video_input = ["-i", phone_video_path]
    if trim_start:
        video_input = ["-ss", f"{max(0.0, float(trim_start)):.3f}"] + video_input
    cmd = [
        "ffmpeg", "-y",
        *video_input,
        "-i", final_audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-ar", "48000",
        "-c:a", "aac", "-b:a", "320k",
        "-shortest",
        out_video_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("video mux failed: " + r.stderr[-500:])
