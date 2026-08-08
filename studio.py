#!/usr/bin/env python3
"""
MicDrop — local karaoke studio web app.

Flow:
  1. Paste a YouTube URL. Backend downloads the pristine backing audio
     (yt-dlp) and returns the video id so the browser can show the video
     (for lyrics) muted, in sync.
  2. You sing. The browser records your mic (WebAudio, high sample rate).
  3. Browser uploads the vocal take. Backend applies the FX chain
     (denoise / de-ess / EQ / compression / pitch-correction / reverb)
     and mixes it with the pristine backing track.
  4. Download the final high-quality song (WAV / FLAC / MP3).

Backend: pure-stdlib http.server + yt-dlp + ffmpeg + librosa (audio_fx.py).
Run:  python3 studio.py   ->  http://127.0.0.1:8770
"""

import base64
import colorsys
import json
import os
import queue as pyqueue
import random
import re
import shutil
import sys
import socket
import ssl
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import audio_fx
import govee_lights

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "index.html")
THEME_PATH = os.path.join(HERE, "theme.css")


def _read_index():
    with open(INDEX_PATH, "rb") as fh:
        return fh.read()


def _read_theme():
    with open(THEME_PATH, "rb") as fh:
        return fh.read()

def _sessions_dir():
    """Sessions go next to the code for a git-clone/dev run, but inside a macOS
    .app bundle that folder is often read-only, so fall back to a user-writable
    Application Support location. Env override wins for advanced users."""
    env = os.environ.get("KARAOKE_SESSIONS")
    if env:
        return env
    # detect running from inside a .app bundle (…/Contents/Resources)
    in_app = ("/Contents/Resources" in HERE) or ("/Contents/MacOS" in HERE)
    local = os.path.join(HERE, "sessions")
    if not in_app:
        try:
            os.makedirs(local, exist_ok=True)
            testf = os.path.join(local, ".write_test")
            with open(testf, "w") as fh:
                fh.write("ok")
            os.remove(testf)
            return local
        except OSError:
            pass
    support = os.path.expanduser("~/Library/Application Support/Karaoke Studio/sessions")
    os.makedirs(support, exist_ok=True)
    return support

SESS_DIR = _sessions_dir()
os.makedirs(SESS_DIR, exist_ok=True)

# NOTE: "localhost" is what the DESKTOP browser opens (YouTube's embedded
# player rejects a raw-IP origin — Error 153 / "Video unavailable" — but
# accepts "localhost"). The server SOCKET binds to 0.0.0.0 below so a phone
# on the same WiFi can also reach it for the phone-camera-sync feature;
# binding 0.0.0.0 still happily accepts localhost/127.0.0.1 connections too,
# so the desktop flow above is unaffected. KARAOKE_PORT lets multiple dev
# checkouts run side by side without fighting over the same port.
HOST, PORT = "localhost", int(os.environ.get("KARAOKE_PORT", 8770))
BIND_HOST = "0.0.0.0"

# Phone-camera-sync needs the phone's browser to grant camera/mic access,
# and browsers only expose getUserMedia on a "secure context" — https:, or
# http: to localhost specifically. A phone hitting the LAN IP over plain
# http (which is all the main server above offers) gets `undefined` for
# navigator.mediaDevices and a cryptic TypeError. So /phone is served from
# a second, HTTPS-wrapped listener on PHONE_PORT instead; PHONE_HTTPS_READY
# reports whether that listener actually came up (needs `openssl` on PATH).
PHONE_PORT = PORT + 1
PHONE_HTTPS_READY = False


def _lan_ip():
    """Best-effort local-network IP, for showing the phone a URL it can reach.
    Opens a UDP socket to a public IP purely to see which local interface the
    OS would route through — no packet is actually sent."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _ensure_phone_cert(lan_ip):
    """Generate a fresh self-signed TLS cert (via the `openssl` CLI, same
    optional-dependency pattern as qrencode/yt-dlp elsewhere here) for the
    phone-camera HTTPS listener. Regenerated on every start rather than
    cached to disk, since the LAN IP baked into its subjectAltName can
    change between runs (DHCP) — a stale IP would fail hostname
    verification even though the cert is otherwise valid. Returns
    (cert_path, key_path) in a throwaway temp dir, or (None, None) if
    openssl isn't available."""
    if not lan_ip:
        return None, None
    cert_dir = os.path.join(os.path.dirname(SESS_DIR), ".certs")
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "phone.crt")
    key_path = os.path.join(cert_dir, "phone.key")
    try:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048",
             "-keyout", key_path, "-out", cert_path,
             "-days", "365", "-nodes",
             "-subj", "/CN=karaoke-studio-phone",
             "-addext", f"subjectAltName=IP:{lan_ip}"],
            check=True, capture_output=True, timeout=15,
        )
        return cert_path, key_path
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None, None


JOBS = {}
LOCK = threading.Lock()

YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


# ---- Stage lighting (Govee LAN Control) -------------------------------------
# Which discovered lights currently react to the singer — a physical-room
# setting, not tied to any one karaoke session, so it's simple global state
# rather than something stored per-jid like everything else here.
LIGHTS_SELECTED = set()  # device IPs
# IPs that /lights/discover actually saw answer on the LAN. /lights/select,
# /lights/test, and /lights/off all used to take an ip/ips straight from the
# request body with no check against this — meaning any client that can
# reach this server (including, notably, anyone with a Tailscale Funnel
# link once one's turned on) could make it fire arbitrary UDP packets at
# arbitrary IPs on the local network, repeatedly, for as long as singing
# continued. Every route that accepts a caller-supplied IP now validates
# against this allowlist first.
LIGHTS_DISCOVERED = set()  # device IPs ever seen from a real discover() scan
LIGHTS_LOCK = threading.Lock()


# Adaptive per-performance pitch range for brightness: tracks the lowest
# and highest notes actually sung in the current unbroken phrase, so "full
# bright" always means THIS performance's climax note, whether that's a
# bass singer's B3 or a soprano's A5 — a single fixed vocal range would
# make one voice type look dramatic and another look permanently dim.
# Resets after a gap (silence between takes/songs) so a new take gets the
# full range back instead of inheriting wherever the last one topped out.
_LIGHTS_RANGE_RESET_GAP = 2.5  # seconds of silence before resetting
_lights_range = {"low": None, "high": None, "last_ts": 0.0}


def _pitch_to_rgb(midi, cents, level, mode="multi", mono_color=None):
    """Map live tuner data to a color + brightness for the lights.
    midi:  detected note as a MIDI number, or None while silent/no pitch.
    cents: how far off the nearest note you are (-50..+50ish).
    level: mic input level, roughly 0..1.
    mode:  "multi" (color follows the note, the original behavior) or
           "mono" (one fixed color from mono_color={r,g,b}; only brightness
           reacts to your singing).
    Height within your current range — not raw mic volume — drives
    brightness in both modes: volume alone sat in a narrow "loud enough to
    sing" band and made every note look about the same, which is exactly
    what felt flat; a big high note should read as the moment, a low note
    as comparatively moody.

    Intensity is baked directly into the RGB magnitude sent to the light
    (not left to the separate Govee "brightness" command alone) — these
    devices don't reliably apply a brightness command that arrives right
    after a color command to the same light, so a dim note needs to
    actually BE a dim color, not just "full-brightness red, allegedly at
    22%". The brightness % is still sent too (harmless, helps if the device
    does honor it), but the RGB scaling is what's actually guaranteed to
    show up on the light."""
    now = time.monotonic()
    gap = now - _lights_range["last_ts"]
    _lights_range["last_ts"] = now

    if midi is None:
        return (0, 0, 0), 0

    if gap > _LIGHTS_RANGE_RESET_GAP or _lights_range["low"] is None:
        # fresh phrase: seed a modest span around this note rather than a
        # single point, so the very first note doesn't instantly read as
        # both the floor AND the ceiling (which would force it to 100%).
        _lights_range["low"] = midi - 4
        _lights_range["high"] = midi + 4
    else:
        _lights_range["low"] = min(_lights_range["low"], midi)
        _lights_range["high"] = max(_lights_range["high"], midi)

    span = max(1, _lights_range["high"] - _lights_range["low"])
    pitch_norm = max(0.0, min(1.0, (midi - _lights_range["low"]) / span))
    in_tune = abs(cents or 0) <= 15

    # Volume still matters, but as a secondary modifier on top of pitch
    # height rather than the primary signal — a loud low note shouldn't
    # outshine a soft high one.
    level_norm = max(0.0, min(1.0, (level or 0) * 3))
    floor = 22
    brightness = (floor + pitch_norm * (100 - floor)) * (0.75 + 0.25 * level_norm)
    brightness = max(15, min(100, round(brightness)))
    scale = brightness / 100.0

    if mode == "mono" and mono_color:
        # Single fixed hue, exactly as picked — but scaled to the computed
        # intensity so a low note is genuinely a dim version of your color,
        # not the same full-strength color with an unreliable brightness
        # command tacked on.
        r = max(0, min(255, int(mono_color.get("r", 255))))
        g = max(0, min(255, int(mono_color.get("g", 255))))
        b = max(0, min(255, int(mono_color.get("b", 255))))
        return (round(r * scale), round(g * scale), round(b * scale)), brightness

    # multi mode: hue cycles once per octave (each of the 12 pitch classes
    # gets its own color around the wheel) so the lights visibly track the
    # melody; saturation dips when you're off-pitch. Value uses the SAME
    # brightness fraction as above (not a separate narrow ramp) so the two
    # numbers can't disagree with each other.
    hue = (midi % 12) / 12.0
    sat = 1.0 if in_tune else 0.55
    r, g, b = colorsys.hsv_to_rgb(hue, sat, scale)
    return (round(r * 255), round(g * 255), round(b * 255)), brightness


def set_job(jid, **kw):
    with LOCK:
        JOBS.setdefault(jid, {}).update(kw)


def get_job(jid):
    with LOCK:
        return dict(JOBS.get(jid, {}))


LYRICS_CLEAN_RE = re.compile(
    r"(?i)\b(karaoke|instrumental|with lyrics|lyrics|lyrical|full song|hd|4k|"
    r"official|video|audio|version|cover|track|hq)\b|[\|\[\]\(\)~]")

def fetch_synced_lyrics(jid):
    """Fetch time-synced lyrics for this session's song from LRCLIB (free,
    keyless) and cache them in meta.json as [{t: seconds, text: line}].

    The search query is the YouTube title scrubbed of karaoke-video cruft
    ('karaoke', 'with lyrics', channel decorations) — those words are why a
    naive title search misses: LRCLIB indexes the ORIGINAL recordings.
    Returns the parsed list, an empty list when nothing matched (cached as
    such so we don't hammer the API), or None when lookup wasn't possible.
    """
    meta = _read_meta(jid)
    if meta.get("lyrics") is not None:
        return meta["lyrics"]
    title = (meta.get("title") or "").strip()
    if not title:
        return None
    # YouTube titles are segment soup: "Song Name Karaoke | Movie | Artist |
    # ChannelName". LRCLIB indexes ORIGINAL recordings and its search is
    # AND-ish, so one junk segment (the channel) zeroes every result. Build
    # cleaned segments and retry with progressively fewer of them — measured
    # on real titles: the full string got 0 hits while "song + movie" or
    # "song + artist" found the exact track with synced lyrics.
    segs = []
    for raw in re.split(r"[|/]+|\s-\s|\|\|", title):
        c = LYRICS_CLEAN_RE.sub(" ", raw)
        c = re.sub(r"\s+", " ", c).strip()
        if c:
            segs.append(c)
    if not segs:
        return None
    import urllib.request, urllib.parse
    synced = None
    for k in range(len(segs), 0, -1):
        q = " ".join(segs[:k])[:120]
        try:
            u = "https://lrclib.net/api/search?q=" + urllib.parse.quote(q)
            req = urllib.request.Request(u, headers={"User-Agent": "MicDrop karaoke (local app)"})
            with urllib.request.urlopen(req, timeout=10) as r:
                hits = json.load(r)
        except Exception:
            return None                  # network trouble: retry next time
        synced = next((h.get("syncedLyrics") for h in hits if h.get("syncedLyrics")), None)
        if synced:
            break
    lines = []
    if synced:
        for m in re.finditer(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", synced):
            t = int(m.group(1)) * 60 + float(m.group(2))
            text = m.group(3).strip()
            if text:
                lines.append({"t": round(t, 2), "text": text[:200]})
        lines.sort(key=lambda x: x["t"])
    _write_meta(jid, lyrics=lines)       # empty list = "looked, none found"
    return lines


def detect_song_key(jid):
    """Detect the song's musical key from the backing track (cached in meta).

    This exists because the singing score was unanchored: it judged distance
    to the nearest CHROMATIC note, so cleanly-sung wrong notes scored as well
    as right ones — a score that "doesn't make sense" for karaoke. Scoring
    against the song's actual scale needs the key, and the backing track we
    already have is the ground truth for it.

    Standard Krumhansl-Schmuckler: fold the spectrum of the first ~2 minutes
    into a 12-bin chroma vector, correlate against the major/minor key
    profiles at all 12 rotations, take the best. Returns
    {"root": 0-11 (C=0), "scale": "major"|"minor", "confidence": 0..1}.
    """
    meta = _read_meta(jid)
    if meta.get("song_key"):
        return meta["song_key"]
    path = os.path.join(SESS_DIR, jid, "backing.wav")
    if not os.path.exists(path):
        return None
    try:
        import numpy as np
        import soundfile as sf
        with sf.SoundFile(path) as fh:
            sr = fh.samplerate
            n = min(len(fh), sr * 120)
            y = fh.read(n, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        N, hop = 8192, 8192          # coarse frames are plenty for chroma
        win = np.hanning(N)
        freqs = np.fft.rfftfreq(N, 1 / sr)
        # map each FFT bin to a pitch class, ignoring rumble and hiss
        valid = (freqs > 60) & (freqs < 4000)
        pcs = (np.round(12 * np.log2(freqs[valid] / 440.0)) + 69).astype(int) % 12
        chroma = np.zeros(12)
        for i in range(0, len(y) - N, hop):
            mag = np.abs(np.fft.rfft(y[i:i + N] * win))[valid] ** 2
            np.add.at(chroma, pcs, mag)
        if chroma.sum() <= 0:
            return None
        chroma = chroma / chroma.sum()
        MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        best = None
        for root in range(12):
            for scale, prof in (("major", MAJ), ("minor", MIN)):
                p = np.roll(prof, root)
                r = float(np.corrcoef(chroma, p)[0, 1])
                if best is None or r > best[0]:
                    best = (r, root, scale)
        key = {"root": best[1], "scale": best[2], "confidence": round(max(0.0, best[0]), 3)}
        _write_meta(jid, song_key=key)
        return key
    except Exception:
        return None


def _read_meta(jid):
    try:
        with open(os.path.join(SESS_DIR, jid, "meta.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _write_meta(jid, **kw):
    sdir = os.path.join(SESS_DIR, jid)
    os.makedirs(sdir, exist_ok=True)
    meta = _read_meta(jid)
    meta.update(kw)
    try:
        with open(os.path.join(sdir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    except OSError:
        pass


# ---- Profiles ----------------------------------------------------------------
# A profile is just a display name, not an account — no login, matching how
# party mode already treats identity (free-text names, no auth). It scopes
# (a) which sessions belong to whom, via meta.json's "profile" field, and
# (b) a saved library of shortlisted songs, in its own JSON file.

def _profiles_dir():
    # sibling to SESS_DIR, reusing its already-resolved writable location
    # (handles the .app-bundle case for free — see _sessions_dir()).
    d = os.path.join(os.path.dirname(SESS_DIR), "profiles")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_profile_name(name):
    """Sanitize a user-typed profile name into something safe to use as both
    a display name and (lowercased) a filename — this comes straight from
    user input, so it must be defused against path traversal."""
    name = re.sub(r"[^A-Za-z0-9 _-]", "", (name or "")).strip()
    return name[:40]


def _profile_slug(name):
    return re.sub(r"\s+", "_", name.lower())


def _read_profile(name):
    slug = _profile_slug(name)
    try:
        with open(os.path.join(_profiles_dir(), slug + ".json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"display_name": name, "library": []}


def _write_profile(name, **kw):
    slug = _profile_slug(name)
    prof = _read_profile(name)
    prof["display_name"] = name
    prof.update(kw)
    with open(os.path.join(_profiles_dir(), slug + ".json"), "w", encoding="utf-8") as fh:
        json.dump(prof, fh)
    return prof


def list_profiles():
    out = []
    try:
        entries = os.listdir(_profiles_dir())
    except OSError:
        return out
    for fn in entries:
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(_profiles_dir(), fn), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        out.append({"name": d.get("display_name") or fn[:-5], "slug": fn[:-5],
                     "email": d.get("email") or ""})
    out.sort(key=lambda p: p["name"].lower())
    return out


def _delete_profile(name):
    slug = _profile_slug(name)
    try:
        os.remove(os.path.join(_profiles_dir(), slug + ".json"))
        return True
    except OSError:
        return False


def _rename_profile(old_name, new_name):
    old_slug = _profile_slug(old_name)
    new_slug = _profile_slug(new_name)
    prof = _read_profile(old_name)
    prof["display_name"] = new_name
    with open(os.path.join(_profiles_dir(), new_slug + ".json"), "w", encoding="utf-8") as fh:
        json.dump(prof, fh)
    if old_slug == new_slug:
        return prof
    try:
        os.remove(os.path.join(_profiles_dir(), old_slug + ".json"))
    except OSError:
        pass
    # re-tag this account's sessions so "resume a recent song" still finds
    # them under the new name instead of orphaning them under the old slug.
    try:
        entries = os.listdir(SESS_DIR)
    except OSError:
        entries = []
    for jid in entries:
        meta = _read_meta(jid)
        if meta.get("profile") == old_slug:
            _write_meta(jid, profile=new_slug)
    return prof


def _session_outputs(jid):
    """Which finished render files actually exist on disk for a session.
    Render completion ("status": "done") only lives in the in-memory JOBS
    dict (see set_job/get_job) and is lost on a server restart — but the
    output files themselves are permanent, so this checks the filesystem
    directly instead, the same way list_recent_sessions already treats
    backing.wav as the source of truth rather than trusting a status flag."""
    sdir = os.path.join(SESS_DIR, jid)
    audio = None
    for fmt in ("wav", "flac", "mp3"):
        cand = os.path.join(sdir, f"final.{fmt}")
        if os.path.isfile(cand):
            audio = f"final.{fmt}"
            break
    video = "final_video.mp4" if os.path.isfile(os.path.join(sdir, "final_video.mp4")) else None
    return {"audio": audio, "video": video}


def _dir_size(path):
    """Total bytes under a directory — used to show each History entry's
    real disk cost, which used to be completely invisible anywhere in the
    app despite sessions/ being the single biggest disk consumer this app
    has (7.9GB across 74 sessions with no cleanup, before the automatic
    cleanup added alongside this)."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def list_recent_sessions(profile=None, limit=8):
    """Sessions with a downloaded backing track, newest first, for the
    'resume a recent song' panel and the History page. Survives server
    restarts (reads meta.json + checks disk for finished outputs directly).
    Optionally filtered to sessions tagged with `profile` (untagged/older
    sessions just won't match a filter — no migration needed)."""
    out = []
    try:
        entries = os.listdir(SESS_DIR)
    except OSError:
        return out
    want_slug = _profile_slug(profile) if profile else None
    for jid in entries:
        sdir = os.path.join(SESS_DIR, jid)
        backing = os.path.join(sdir, "backing.wav")
        if not os.path.isfile(backing):
            continue
        meta = _read_meta(jid)
        if want_slug is not None and meta.get("profile") != want_slug:
            continue
        created_at = meta.get("created_at")
        if created_at is None:
            try:
                created_at = os.path.getmtime(backing)
            except OSError:
                created_at = 0
        vid = meta.get("video_id")
        outputs = _session_outputs(jid)
        take_count = len([t for t in (meta.get("takes") or []) if not t.get("deleted")])
        out.append({
            "id": jid,
            "title": meta.get("title") or "Karaoke",
            "video_id": vid,
            "duration": meta.get("duration"),
            "thumb": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg" if vid else None,
            "created_at": created_at,
            "has_audio": bool(outputs["audio"]),
            "has_video": bool(outputs["video"]),
            "take_count": take_count,
            "size_bytes": _dir_size(sdir),
            # backing track downloaded, never actually sung into — pure
            # disk waste with nothing of yours to lose by clearing it out.
            "abandoned": take_count == 0 and not outputs["audio"] and not outputs["video"],
        })
    out.sort(key=lambda s: s["created_at"] or 0, reverse=True)
    return out[:limit]


def _probe_duration(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                            "format=duration", "-of", "csv=p=0", path],
                           capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return None


# ---- Take history (multi-take recording + comping) -------------------------
# Stored as a `takes` list inside the session's meta.json (via _read_meta/
# _write_meta above), one WAV file per take. Deletes are soft (a `deleted`
# flag) so undo can just un-hide the entry — the file is never removed.

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
        # the capture-latency compensation (ms) that was baked into this
        # take's WAV at save time — pure diagnostics, so "vocal lags" reports
        # can be checked against what the client actually corrected for
        "latency_ms": latency_ms,
        # non-destructive vocal-vs-karaoke timing nudge, editable any time
        # from the Track Editor (unlike vocalDelayMs, which is a one-shot
        # capture-time correction baked into the WAV before this take ever
        # existed) — 0 = no nudge, +N = vocal plays N ms later, -N = earlier.
        "align_ms": 0,
    }
    takes.append(take)
    _write_meta(jid, takes=takes)
    return take


def _get_take(jid, tid):
    for t in _read_meta(jid).get("takes") or []:
        if t.get("id") == tid:
            return t
    return None


def _update_take(jid, tid, **fields):
    """Merge fields (e.g. render/video progress) into one take's own record,
    keyed by that take's id — not the session-wide JOBS dict. Render/video
    generation used to live in one global per-session slot, so recording a
    retake while the previous take's video was still combining (or had
    already finished) silently clobbered or no-op'd against that old take's
    state instead of tracking its own. Scoping status to the take itself is
    what lets each take show its own "generating…" progress and end up with
    its own result, independent of whatever else is happening in the
    session."""
    meta = _read_meta(jid)
    takes = meta.get("takes") or []
    found = False
    for t in takes:
        if t.get("id") == tid:
            for k, v in fields.items():
                if isinstance(v, dict) and isinstance(t.get(k), dict):
                    t[k].update(v)
                else:
                    t[k] = v
            found = True
    if found:
        _write_meta(jid, takes=takes)
    return found


def _set_take_deleted(jid, tid, deleted):
    meta = _read_meta(jid)
    takes = meta.get("takes") or []
    found = False
    for t in takes:
        if t.get("id") == tid:
            t["deleted"] = deleted
            found = True
    if found:
        _write_meta(jid, takes=takes)
    return found


def _del_take(jid, tid):
    return _set_take_deleted(jid, tid, True)


def _restore_take(jid, tid):
    return _set_take_deleted(jid, tid, False)


def list_takes(jid):
    return [t for t in (_read_meta(jid).get("takes") or []) if not t.get("deleted")]


def job_or_disk(jid):
    """Return the in-memory job, or reconstruct a minimal one from disk if the
    server was restarted but the session folder (with a backing track) survives.
    This keeps a downloaded track usable across restarts."""
    j = get_job(jid)
    if j:
        return j
    if not jid or not re.fullmatch(r"[A-Za-z0-9]{6,32}", jid or ""):
        return {}
    backing = os.path.join(SESS_DIR, jid, "backing.wav")
    if os.path.exists(backing):
        meta = _read_meta(jid)
        dur = meta.get("duration")
        if dur is None:
            dur = _probe_duration(backing)
        set_job(jid, status="ready", stage="ready", pct=100,
                title=meta.get("title") or "Karaoke", video_id=meta.get("video_id"),
                duration=dur, recovered=True)
        return get_job(jid)
    return {}


def extract_video_id(url):
    m = YT_ID_RE.search(url)
    return m.group(1) if m else None


def search_youtube(query, n=12):
    """Search YouTube via yt-dlp (no API key). Returns a list of result dicts:
    {id, title, duration, thumb, channel}."""
    query = (query or "").strip()
    if not query:
        return []
    # ytsearchN: runs a search and returns up to N flat results (fast, no per-video fetch)
    cmd = [
        "yt-dlp", f"ytsearch{int(n)}:{query}",
        "--flat-playlist", "--dump-json", "--no-warnings",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    results = []
    for line in r.stdout.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = d.get("id")
        if not vid or len(vid) != 11:
            continue
        thumbs = d.get("thumbnails") or []
        thumb = (thumbs[0].get("url") if thumbs else None) or d.get("thumbnail") \
            or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
        results.append({
            "id": vid,
            "title": d.get("title") or "(untitled)",
            "duration": d.get("duration"),
            "thumb": thumb,
            "channel": d.get("channel") or d.get("uploader") or "",
        })
    return results


# ---- Backing-track download -------------------------------------------------

def separate_vocals(jid, backing_path):
    """Split the downloaded track into instrumental + vocals with Demucs.

    This is what turns ANY song into a karaoke track: the instrumental
    replaces backing.wav (the original full mix is kept as original.wav),
    and the vocal stem is saved as vocals.wav — which doubles as the
    melody reference for note-accurate scoring. Runs on CPU in roughly
    1/4 of song length on this machine (measured: 30s of audio in ~8s,
    stems correlating 0.98 with ground truth).
    """
    sdir = os.path.join(SESS_DIR, jid)
    outdir = os.path.join(sdir, "sep")
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals",
           "-o", outdir, backing_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("separation failed: " + (r.stderr or "")[-300:])
    base = os.path.splitext(os.path.basename(backing_path))[0]
    stem_dir = os.path.join(outdir, "htdemucs", base)
    inst = os.path.join(stem_dir, "no_vocals.wav")
    voc = os.path.join(stem_dir, "vocals.wav")
    if not (os.path.exists(inst) and os.path.exists(voc)):
        raise RuntimeError("separation produced no stems")
    shutil.move(backing_path, os.path.join(sdir, "original.wav"))
    shutil.move(inst, os.path.join(sdir, "backing.wav"))
    shutil.move(voc, os.path.join(sdir, "vocals.wav"))
    shutil.rmtree(outdir, ignore_errors=True)


def do_prepare(jid, url, profile=None, strip_vocals=False):
    set_job(jid, status="running", stage="downloading", pct=0, error=None)
    vid = extract_video_id(url)
    sdir = os.path.join(SESS_DIR, jid)
    os.makedirs(sdir, exist_ok=True)
    out = os.path.join(sdir, "backing.%(ext)s")

    cmd = [
        "yt-dlp", "--no-playlist", "-f", "bestaudio/best",
        "--extract-audio", "--audio-format", "wav",   # decode to PCM for clean mixing
        "--newline", "-o", out, url,
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    except FileNotFoundError:
        set_job(jid, status="error", error="yt-dlp not found on PATH.")
        return

    for line in proc.stdout:
        m = re.search(r"\[download\]\s+([\d.]+)%", line)
        if m:
            try:
                set_job(jid, pct=float(m.group(1)))
            except ValueError:
                pass
        if "Extracting audio" in line or "[ExtractAudio]" in line:
            set_job(jid, stage="preparing", pct=99)
    proc.wait()

    if proc.returncode != 0:
        set_job(jid, status="error",
                error="Download failed. If this keeps happening run: brew upgrade yt-dlp")
        return

    backing = os.path.join(sdir, "backing.wav")
    if not os.path.exists(backing):
        set_job(jid, status="error", error="Backing track not produced.")
        return

    title = None
    try:
        r = subprocess.run(["yt-dlp", "--no-playlist", "--skip-download",
                            "--print", "%(title)s", url],
                           capture_output=True, text=True, timeout=30)
        title = (r.stdout.strip().splitlines() or [None])[0]
    except Exception:
        pass

    separated = False
    if strip_vocals:
        set_job(jid, stage="separating vocals (any song → karaoke)", pct=99)
        try:
            separate_vocals(jid, backing)
            separated = True
        except Exception as e:
            # a failed separation shouldn't kill the session — the full mix
            # still works as a sing-along track; surface it in the title bar
            set_job(jid, stage="preparing", separation_error=str(e)[:200])

    dur = _probe_duration(backing)
    set_job(jid, status="ready", stage="ready", pct=100,
            video_id=vid, title=title or "Karaoke", duration=dur,
            separated=separated)
    meta_kw = {"video_id": vid, "title": title or "Karaoke", "duration": dur,
               "url": url, "created_at": time.time(), "separated": separated}
    if profile:
        meta_kw["profile"] = _profile_slug(profile)
    _write_meta(jid, **meta_kw)


# ---- Local file import (no YouTube) -----------------------------------------

MAX_IMPORT_BYTES = 500 * 1024 * 1024  # 500MB


def do_import(jid, upload_path, filename):
    """Convert an uploaded local audio/video file into the session's backing
    track, the same role yt-dlp's download plays for a YouTube session. No
    video_id is set, so the client falls into the existing lyrics-fallback
    (synced clock) UI automatically — there's no video to show."""
    set_job(jid, status="running", stage="preparing", pct=50, error=None)
    sdir = os.path.join(SESS_DIR, jid)
    backing = os.path.join(sdir, "backing.wav")
    cmd = ["ffmpeg", "-y", "-i", upload_path, "-vn",
           "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", backing]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        set_job(jid, status="error", error="ffmpeg not found on PATH.")
        return
    try:
        os.remove(upload_path)
    except OSError:
        pass
    if r.returncode != 0 or not os.path.exists(backing):
        set_job(jid, status="error",
                error="Could not read that file as audio/video. Try a different file.")
        return

    title = os.path.splitext(os.path.basename(filename or ""))[0].strip() or "Karaoke"
    set_job(jid, status="ready", stage="ready", pct=100,
            video_id=None, title=title, duration=_probe_duration(backing))


# ---- Backing transpose (change key), rendered on demand + cached ------------

_KEY_LOCKS = {}

def backing_for_key(jid, semitones):
    """Return path to the backing track transposed by `semitones`, rendering
    and caching it on first request. semitones==0 returns the original."""
    sdir = os.path.join(SESS_DIR, jid)
    base = os.path.join(sdir, "backing.wav")
    if not os.path.exists(base):
        return None
    if semitones == 0:
        return base
    out = os.path.join(sdir, f"backing_key{semitones:+d}.wav")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    # serialize concurrent renders of the same key
    with LOCK:
        lk = _KEY_LOCKS.setdefault((jid, semitones), threading.Lock())
    with lk:
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
        try:
            audio_fx.transpose_backing(base, out, semitones)
        except Exception:
            return base  # fall back to original key rather than 404
    return out


# ---- Vocal processing + mixdown --------------------------------------------

# One lock per session, so a second /render for the same jid while one is
# already running can't race on vocal_fx.wav/final.<fmt> (same pattern as
# _KEY_LOCKS above).
_RENDER_LOCKS = {}


def _render_lock(jid):
    with LOCK:
        return _RENDER_LOCKS.setdefault(jid, threading.Lock())


def do_render(jid, vocal_path, fx, mix_opts, scope_id=None):
    has_take = bool(scope_id and _get_take(jid, scope_id))
    def progress(stage=None, **kw):
        fields = dict(kw)
        if stage is not None:
            fields["stage"] = stage
        set_job(jid, **fields)
        if has_take:
            _update_take(jid, scope_id, render=fields)

    progress(status="rendering", stage="starting", error=None)
    sdir = os.path.join(SESS_DIR, jid)
    # mix against whatever key the singer settled on (0 = original)
    try:
        music_key = int(mix_opts.get("music_key", 0) or 0)
    except (TypeError, ValueError):
        music_key = 0
    music_key = max(-6, min(6, music_key))
    backing = backing_for_key(jid, music_key) or os.path.join(sdir, "backing.wav")
    if not backing or not os.path.exists(backing):
        progress(status="error", error="Backing track missing; prepare again.")
        return

    tmp = os.path.join(sdir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    out_fmt = mix_opts.get("format", "wav")

    trim_start = mix_opts.get("trim_start")
    trim_end = mix_opts.get("trim_end")
    try:
        trim_start = float(trim_start) if trim_start not in (None, "") else None
    except (TypeError, ValueError):
        trim_start = None
    try:
        trim_end = float(trim_end) if trim_end not in (None, "") else None
    except (TypeError, ValueError):
        trim_end = None

    # Every output/intermediate file from here on is named with `scope_id`
    # (when there is one) so a retake's render can run without touching, or
    # racing against, whatever an earlier take's still-in-flight render or
    # video combine is reading/writing under its OWN scoped filenames.
    suffix = f"_{scope_id}" if scope_id else ""
    try:
        processed_vocal = os.path.join(sdir, f"vocal_fx{suffix}.wav")
        audio_fx.process_vocal(
            vocal_path, processed_vocal, fx, tmp,
            progress=lambda stage: progress(stage=stage),
        )

        # Any number of simultaneous harmony layers now, not just one —
        # each entry is {take_id, gain_db, fx?}, built client-side from
        # whichever harmony takes are linked to the active lead. fx is that
        # take's OWN saved fx_snapshot (converted client-side to this same
        # server fx shape) — sent only when one exists, so a harmony with
        # nothing saved still mixes in raw rather than guessing settings.
        harmonies = []
        for i, hspec in enumerate(mix_opts.get("harmonies") or []):
            h_take_id = hspec.get("take_id")
            if not h_take_id:
                continue
            h_take = _get_take(jid, h_take_id)
            if not h_take:
                continue
            h_path = os.path.join(sdir, h_take["file"])
            h_fx = hspec.get("fx")
            if h_fx:
                processed_h = os.path.join(sdir, f"harmony{i}_fx{suffix}.wav")
                audio_fx.process_vocal(h_path, processed_h, h_fx, tmp)
                h_path = processed_h
            try:
                h_gain = float(hspec.get("gain_db", -3.0))
            except (TypeError, ValueError):
                h_gain = -3.0
            harmonies.append({"path": h_path, "gain_db": h_gain})

        progress(stage="mixing", format=out_fmt)
        final = os.path.join(sdir, f"final{suffix}.{out_fmt}")
        # Kept as the video-mux audio source regardless of the download
        # format chosen — if that's a lossy format (MP3), the video's audio
        # would otherwise be a lossy re-encode of an already-lossy file.
        master_wav = os.path.join(sdir, f"final_master{suffix}.wav") if out_fmt != "wav" else None
        try:
            vocal_align_ms = float(mix_opts.get("vocal_align_ms", 0) or 0)
        except (TypeError, ValueError):
            vocal_align_ms = 0.0
        audio_fx.mixdown(
            processed_vocal, backing, final,
            vocal_gain_db=float(mix_opts.get("vocal_gain_db", 0)),
            music_gain_db=float(mix_opts.get("music_gain_db", -3)),
            harmonies=harmonies,
            out_format=out_fmt,
            loudnorm=bool(mix_opts.get("loudnorm", True)),
            trim_start=trim_start,
            trim_end=trim_end,
            master_wav_path=master_wav,
            music_eq=mix_opts.get("music_eq"),
            vocal_align_ms=vocal_align_ms,
        )
    except Exception as e:
        progress(status="error", error=str(e)[:400])
        return

    title = get_job(jid).get("title", "song")
    safe = re.sub(r'[\\/:*?"<>|]+', "_", title)[:100].strip() or "song"
    # trim_start is kept around (not just used above) so do_combine_video can
    # seek the phone video by the same amount later — the render and the
    # video combine happen in separate requests/steps, so this is the only
    # way that later step knows the front of the mix got trimmed.
    progress(status="done", stage="done",
             final_file=os.path.basename(final),
             audio_master_file=os.path.basename(master_wav) if master_wav else os.path.basename(final),
             display_name=f"{safe} (karaoke).{out_fmt}",
             render_trim_start=trim_start)

    # tmp/ is scratch space for pitch_correct's per-segment rubberband
    # files — dead weight the instant a render finishes, and one real
    # contributor to sessions/ growing unbounded with no cleanup anywhere
    # in this codebase. Best-effort: a render having finished successfully
    # is worth more than a leftover temp file blocking on it.
    shutil.rmtree(tmp, ignore_errors=True)


def do_combine_video(jid, scope_id, phone_video_file, final_mix_file):
    """Mux the phone's video with the finished audio mix into one final
    video file. No alignment step needed: the desktop starts recording the
    live WebRTC stream from the phone in the same tick it starts the vocal
    recording, so both are already on the same clock from t=0 — unlike the
    old design, which had the phone record+upload independently and relied
    on a sync chirp (audible only through a real speaker, not headphones)
    to find the offset after the fact.

    scope_id ties output to one take (see /render) — without it, a retake's
    combine used to overwrite a single shared "final_video.mp4" regardless
    of which take actually finished last, and a guard meant to stop a
    combine from double-running would also silently swallow a NEW take's
    combine request if the previous take had already reached "done"."""
    sdir = os.path.join(SESS_DIR, jid)
    j = get_job(jid)
    has_take = bool(scope_id and _get_take(jid, scope_id))

    def set_video(**fields):
        set_job(jid, **fields)
        if has_take:
            _update_take(jid, scope_id, video=fields)

    phone_video = os.path.join(sdir, phone_video_file)
    final_mix = os.path.join(sdir, final_mix_file)
    out_name = f"final_video_{scope_id}.mp4" if scope_id else "final_video.mp4"
    out_video = os.path.join(sdir, out_name)
    try:
        audio_fx.mux_video(phone_video, final_mix, out_video,
                            trim_start=j.get("render_trim_start"))
    except Exception as e:
        set_video(status="error", error=str(e)[:400])
        return
    set_video(status="done", error=None, final_video_file=out_name)

    # The raw phone-camera recording (often the single biggest file in a
    # session now that 1080p video is the default) is fully consumed the
    # moment it's muxed into final_video — /combine_video's own guard above
    # never lets this scope_id combine again, so nothing will ever read it
    # a second time. This is the other big contributor to sessions/ growing
    # unbounded with no cleanup anywhere in this codebase.
    try:
        os.remove(phone_video)
    except OSError:
        pass


def _run_render_locked(jid, vocal_path, fx, mix_opts, lock, scope_id=None):
    """Thread target for /render: holds `lock` for the whole render so a
    second request for the same session can't start until this one is done
    (the lock itself was already acquired by the HTTP handler)."""
    try:
        do_render(jid, vocal_path, fx, mix_opts, scope_id)
    finally:
        lock.release()


# ---- Party mode: room + queue ------------------------------------------------
# One party room per running server (a party is one physical event, so there's
# no need for multi-room isolation). The host's TV screen and every guest phone
# share this state; ROOM_LOCK guards it, and SUBSCRIBERS holds an SSE queue per
# connected browser tab so everyone sees queue changes live. Songs themselves
# still go through the existing JOBS/do_prepare pipeline — a queue entry just
# points at a jid once that job's status is "ready".

ROOM = {"code": None, "queue": [], "now_playing": None, "challenges": [], "guests": [],
        "scores": {}, "live_score": None, "scoring_enabled": True}
ROOM_LOCK = threading.Lock()
SUBSCRIBERS = []
SUB_LOCK = threading.Lock()


def room_state():
    with ROOM_LOCK:
        lan = _lan_ip()
        secure = f"https://{lan}:{PHONE_PORT}" if (lan and PHONE_HTTPS_READY) else None
        return {"code": ROOM["code"], "queue": list(ROOM["queue"]),
                "now_playing": ROOM["now_playing"], "challenges": list(ROOM["challenges"]),
                "guests": list(ROOM["guests"]),
                "scores": dict(ROOM["scores"]), "live_score": ROOM["live_score"],
                "secure_join": secure,
                "scoring_enabled": ROOM.get("scoring_enabled", True)}


def room_join(name, device_id):
    """Register a guest as present in the party (shown on the TV screen),
    independent of whether they've queued a song yet — just opening /join
    and entering a name counts. Each name is bound to the device_id that
    first claims it in this room (a random token the guest's browser
    generates and persists) — this is what lets challenge_accept later
    verify "is this really the person being challenged," not just "someone
    typed their name," so nobody can be put on the spot to sing by someone
    else impersonating them and accepting on their behalf."""
    name = (name or "").strip()[:40]
    device_id = (device_id or "").strip()[:64]
    if not name or not device_id:
        return {"ok": False, "error": "Missing name."}
    with ROOM_LOCK:
        if ROOM["code"] is None:
            return {"ok": False, "error": "No open party room."}
        existing = next((g for g in ROOM["guests"] if g["name"] == name), None)
        if existing and existing["device_id"] != device_id:
            return {"ok": False, "error": "That name is already taken by someone else at this party — try another."}
        if not existing:
            ROOM["guests"].append({"name": name, "device_id": device_id, "joined_at": time.time()})
    broadcast_room()
    return {"ok": True}


def broadcast_room():
    payload = json.dumps(room_state())
    with SUB_LOCK:
        subs = list(SUBSCRIBERS)
    for q in subs:
        q.put(payload)


def start_room():
    code = "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4))
    with ROOM_LOCK:
        ROOM["code"] = code
        ROOM["queue"] = []
        ROOM["now_playing"] = None
        ROOM["challenges"] = []
        ROOM["guests"] = []
        ROOM["scores"] = {}
        ROOM["live_score"] = None
        ROOM["scoring_enabled"] = True
    broadcast_room()
    return code


def _make_queue_entry(jid, singers, semitones, from_singer=None):
    j = get_job(jid)
    if not j or j.get("status") != "ready":
        return None
    return {
        "entry_id": uuid.uuid4().hex[:10],
        "jid": jid,
        "title": j.get("title") or "Karaoke",
        "singers": singers,
        "semitones": semitones,
        "status": "queued",
        "from_singer": from_singer,  # set only for entries born from an accepted challenge
    }


def queue_add(jid, singers, semitones):
    entry = _make_queue_entry(jid, singers, semitones)
    if not entry:
        return None
    with ROOM_LOCK:
        if ROOM["code"] is None:
            return None
        ROOM["queue"].append(entry)
    broadcast_room()
    return entry


def _finalize_live_score_locked():
    """Fold the current live score into the party scoreboard (caller holds
    ROOM_LOCK). Runs when a turn ends — either the singer's phone said
    'done' or the host hit Next mid-song."""
    ls = ROOM.get("live_score")
    if not ls or not ls.get("points"):
        ROOM["live_score"] = None
        return
    for name in ls.get("singers") or [ls.get("name")]:
        if not name:
            continue
        rec = ROOM["scores"].setdefault(name, {"total": 0, "best": 0, "songs": 0, "best_rank": "D"})
        pts = int(ls["points"])
        rec["total"] += pts
        rec["songs"] += 1
        if pts > rec["best"]:
            rec["best"] = pts
            rec["best_rank"] = ls.get("rank") or "D"
    ROOM["live_score"] = None


def queue_advance():
    """Mark the current now-playing entry done (if any) and promote the next
    queued entry."""
    with ROOM_LOCK:
        _finalize_live_score_locked()
        q = ROOM["queue"]
        for e in q:
            if e["status"] == "now_playing":
                e["status"] = "done"
        nxt = next((e for e in q if e["status"] == "queued"), None)
        if nxt:
            nxt["status"] = "now_playing"
            ROOM["now_playing"] = nxt["entry_id"]
        else:
            ROOM["now_playing"] = None
    broadcast_room()


def queue_remove(entry_id):
    with ROOM_LOCK:
        ROOM["queue"] = [e for e in ROOM["queue"] if e["entry_id"] != entry_id]
        if ROOM["now_playing"] == entry_id:
            ROOM["now_playing"] = None
    broadcast_room()


def queue_move(entry_id, direction):
    """Swap entry_id with its neighbor among still-queued entries (done /
    now-playing entries are skipped over, not swapped with)."""
    with ROOM_LOCK:
        q = ROOM["queue"]
        queued_idx = [i for i, e in enumerate(q) if e["status"] == "queued"]
        pos = next((k for k, i in enumerate(queued_idx) if q[i]["entry_id"] == entry_id), None)
        if pos is None:
            return
        target = pos - 1 if direction == "up" else pos + 1
        if target < 0 or target >= len(queued_idx):
            return
        i, j = queued_idx[pos], queued_idx[target]
        q[i], q[j] = q[j], q[i]
    broadcast_room()


def queue_reorder(order):
    """Reorder the still-queued entries to match `order` (a list of entry_ids
    from a drag-and-drop drop on the TV screen). done/now_playing entries
    keep their existing positions; only the queued ones get rearranged."""
    with ROOM_LOCK:
        q = ROOM["queue"]
        by_id = {e["entry_id"]: e for e in q if e["status"] == "queued"}
        if set(order) != set(by_id):
            return False  # not a straight permutation of what's actually queued — ignore
        queued_idx = [i for i, e in enumerate(q) if e["status"] == "queued"]
        for i, entry_id in zip(queued_idx, order):
            q[i] = by_id[entry_id]
    broadcast_room()
    return True


def challenge_add(jid, from_singer, to_singer, semitones):
    j = get_job(jid)
    if not j or j.get("status") != "ready":
        return None
    challenge = {
        "challenge_id": uuid.uuid4().hex[:10],
        "jid": jid,
        "title": j.get("title") or "Karaoke",
        "from_singer": from_singer,
        "to_singer": to_singer,
        "semitones": semitones,
        "status": "pending",
    }
    with ROOM_LOCK:
        if ROOM["code"] is None:
            return None
        ROOM["challenges"].append(challenge)
    broadcast_room()
    return challenge


def _guest_owns_name(name, device_id):
    """True if device_id is the device that claimed `name` via room_join, or
    if nobody has claimed that name yet (can't verify either way — err
    toward allowing rather than locking out a party where people jump
    straight to queueing without ever hitting the join step)."""
    g = next((g for g in ROOM["guests"] if g["name"] == name), None)
    return g is None or g["device_id"] == device_id


def challenge_accept(challenge_id, device_id=None):
    """Move an accepted challenge into the real queue as an entry sung by
    the challenged person, tagged with who challenged them.

    device_id=None means this came from the TV/host screen, not a guest's
    phone — the host is a trusted physical presence who can get a verbal
    "yeah I'm cool with that" before clicking, so host accepts are always
    allowed. A guest-originated accept (device_id set) is only allowed if
    it's coming from the device that actually claimed the challenged
    person's name — otherwise anyone could accept "as" someone else and
    put them on the spot to sing without real consent."""
    with ROOM_LOCK:
        ch = next((c for c in ROOM["challenges"] if c["challenge_id"] == challenge_id), None)
        if not ch or ch["status"] != "pending":
            return None, "unknown or already-resolved challenge"
        if device_id is not None and not _guest_owns_name(ch["to_singer"], device_id):
            return None, "only " + ch["to_singer"] + " (or the host) can accept this challenge"
        entry = _make_queue_entry(ch["jid"], [ch["to_singer"]], ch["semitones"],
                                   from_singer=ch["from_singer"])
        if not entry:
            return None, "song isn't ready"
        ROOM["queue"].append(entry)
        ch["status"] = "accepted"
    broadcast_room()
    return entry, None


def challenge_decline(challenge_id, device_id=None):
    with ROOM_LOCK:
        ch = next((c for c in ROOM["challenges"] if c["challenge_id"] == challenge_id), None)
        if not ch:
            return True  # already gone — fine
        if device_id is not None and not _guest_owns_name(ch["to_singer"], device_id):
            return False
        ROOM["challenges"] = [c for c in ROOM["challenges"] if c["challenge_id"] != challenge_id]
    broadcast_room()
    return True


def make_qr_png(data):
    """Render `data` as a QR code PNG via the optional `qrencode` CLI (same
    "optional but recommended" pattern as rubberband/sox). Returns None if
    qrencode isn't installed or fails — callers should fall back to showing
    the join URL as plain text."""
    try:
        r = subprocess.run(["qrencode", "-o", "-", "-t", "PNG", "-s", "6", "-m", "2", data],
                            capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or not r.stdout:
        return None
    return r.stdout


# ---- HTTP -------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _base_url(self):
        """Base URL guests should actually use to reach this server — derived
        from the Host header the browser sent, not from a hardcoded LAN IP.
        On a plain LAN request that's still e.g. http://10.0.0.6:8770 (same
        as before), but it also comes out right automatically when the
        request arrived through a tunnel (Tailscale Funnel, Cloudflare
        Tunnel, etc.), which terminates HTTPS at some public hostname and
        forwards here — those set X-Forwarded-Proto so we know to say
        https:// instead of http:// even though this process only ever
        speaks plain HTTP itself. Falls back to LAN-IP detection if there's
        no Host header, OR if Host is localhost/127.0.0.1 — the host opens
        the TV screen at exactly that address (it's what the YouTube embed
        needs), but "localhost" in a join link means something different on
        every device that opens it, so it's useless to a guest's phone."""
        host = self.headers.get("Host")
        hostname = (host or "").split(":")[0]
        if not host or hostname in ("localhost", "127.0.0.1"):
            lan = _lan_ip()
            return f"http://{lan}:{PORT}" if lan else None
        proto = self.headers.get("X-Forwarded-Proto", "http")
        return f"{proto}://{host}"

    def do_GET(self):
        p = urlparse(self.path)
        if p.path in ("/", "/index.html"):
            try:
                b = _read_index()
            except OSError:
                self._json(500, {"error": "index.html is missing next to studio.py."})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            # Without this the response has NO caching headers at all, and
            # browsers heuristically cache such responses — so after the
            # code on disk changes, a normal reload can silently serve the
            # OLD app from cache. That looks exactly like "the bug I just
            # fixed is still happening", indefinitely, on the user's open
            # browser. The app shell must always come from disk.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
            return

        if p.path == "/theme.css":
            try:
                b = _read_theme()
            except OSError:
                self._json(500, {"error": "theme.css is missing next to studio.py."})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/css; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")   # same reason as index.html
            self.end_headers()
            self.wfile.write(b)
            return

        if p.path == "/search":
            q = (parse_qs(p.query).get("q") or [""])[0]
            if not q.strip():
                self._json(400, {"error": "Type something to search for."})
                return
            try:
                results = search_youtube(q, n=12)
            except Exception:
                results = []
            self._json(200, {"results": results})
            return

        if p.path == "/pro_chain_defaults":
            self._json(200, audio_fx.PRO_CHAIN_DEFAULTS)
            return

        if p.path == "/pro_chain_presets":
            self._json(200, audio_fx.PRO_CHAIN_PRESETS)
            return

        if p.path == "/sessions":
            q = parse_qs(p.query)
            profile = (q.get("profile") or [""])[0]
            try:
                limit = max(1, min(200, int((q.get("limit") or ["8"])[0])))
            except ValueError:
                limit = 8
            self._json(200, {"sessions": list_recent_sessions(profile or None, limit=limit)})
            return

        if p.path == "/profiles":
            self._json(200, {"profiles": list_profiles()})
            return

        if p.path == "/profile/library":
            name = _safe_profile_name((parse_qs(p.query).get("name") or [""])[0])
            if not name:
                self._json(400, {"error": "Missing profile name."})
                return
            prof = _read_profile(name)
            self._json(200, {"display_name": prof.get("display_name") or name,
                              "library": prof.get("library") or []})
            return

        if p.path == "/status":
            q = parse_qs(p.query)
            jid = (q.get("id") or [""])[0]
            # job_or_disk (not get_job) so a resumed session — from the
            # "resume a recent song" panel, or a bookmarked ?sid= link —
            # still works after a server restart wiped the in-memory JOBS
            # dict, as long as the session's files are still on disk.
            j = job_or_disk(jid)
            if not j:
                self._json(404, {"error": "unknown job"})
                return
            # A whole-session render overwrites the job-level status with
            # "done" (or "error" on a failed render) — see do_render's final
            # progress() call — so the status field alone can't tell a
            # client whether the SESSION is usable, only what the last
            # render did. has_backing is the ground truth "this session can
            # be resumed and played": the downloaded backing track exists.
            j = dict(j)
            j["has_backing"] = os.path.exists(
                os.path.join(SESS_DIR, jid, "backing.wav"))
            # &take=: overlay THIS take's own render/video progress on top of
            # the session-level status, so polling for a specific take never
            # reads whatever a different (earlier or later) take's render
            # happens to be doing.
            take_id = (q.get("take") or [""])[0]
            take = _get_take(jid, take_id) if take_id else None
            if take:
                j = dict(j)
                j.update(take.get("render") or {})
                video = take.get("video")
                if video:
                    j["video_status"] = video.get("status")
                    j["video_error"] = video.get("error")
                    j["final_video_file"] = video.get("final_video_file")
                else:
                    j["video_status"] = None
            self._json(200, j)
            return

        if p.path == "/lyrics":
            jid = (parse_qs(p.query).get("id") or [""])[0]
            if not job_or_disk(jid):
                self._json(404, {"error": "unknown session"})
                return
            self._json(200, {"lyrics": fetch_synced_lyrics(jid)})
            return

        if p.path == "/songkey":
            jid = (parse_qs(p.query).get("id") or [""])[0]
            if not job_or_disk(jid):
                self._json(404, {"error": "unknown session"})
                return
            key = detect_song_key(jid)
            self._json(200, {"key": key})
            return

        if p.path == "/backing":
            # serve the backing track so the browser can play it in sync.
            # ?key=N renders (and caches) a version transposed by N semitones,
            # so you can rehearse and record in your own key.
            q = parse_qs(p.query)
            jid = (q.get("id") or [""])[0]
            try:
                key = int(float((q.get("key") or ["0"])[0]))
            except ValueError:
                key = 0
            key = max(-6, min(6, key))
            f = backing_for_key(jid, key)
            if not f:
                self._json(404, {"error": "backing not ready"})
                return
            self._send_file(f, "audio/wav")
            return

        if p.path == "/final":
            q = parse_qs(p.query)
            jid = (q.get("id") or [""])[0]
            j = get_job(jid)
            # &take=: this specific take's own rendered result, so retakes
            # never fetch whatever the session's single most-recent render
            # happened to be.
            take = _get_take(jid, (q.get("take") or [""])[0])
            render = (take.get("render") or {}) if take else {}
            final_file = render.get("final_file") if take else j.get("final_file")
            display_name = render.get("display_name") if take else j.get("display_name")
            done = (render.get("status") == "done") if take else (j.get("status") == "done")
            if not done or not final_file:
                # in-memory job status doesn't survive a server restart —
                # fall back to whatever finished output actually exists on
                # disk (see _session_outputs) so old History entries stay
                # downloadable across restarts, not just within one run.
                final_file = _session_outputs(jid)["audio"]
                if not final_file:
                    self._json(404, {"error": "not ready"})
                    return
                meta = _read_meta(jid)
                safe_title = re.sub(r'[\\/:*?"<>|\r\n]+', "_", meta.get("title") or "song")[:100].strip() or "song"
                display_name = f"{safe_title} (karaoke).{final_file.rsplit('.', 1)[-1]}"
            f = os.path.join(SESS_DIR, jid, final_file)
            self._send_file(f, "application/octet-stream", display_name)
            return

        if p.path == "/final_vocal":
            # The fully processed vocal-only stem (denoise/EQ/compression/
            # pitch-correction already applied — the same audio that went
            # into the mix) is written to disk as vocal_fx.wav during render
            # and never cleaned up, but was never exposed as a download.
            # Serving it as-is (uncompressed WAV) rather than re-encoding it
            # to anything lossy keeps it at the same quality it was mixed at.
            jid = (parse_qs(p.query).get("id") or [""])[0]
            j = get_job(jid)
            f = os.path.join(SESS_DIR, jid, "vocal_fx.wav")
            if not os.path.isfile(f):
                self._json(404, {"error": "vocal stem not found"})
                return
            display_name = j.get("display_name")
            if not display_name:
                meta = _read_meta(jid)
                safe_title = re.sub(r'[\\/:*?"<>|\r\n]+', "_", meta.get("title") or "song")[:100].strip() or "song"
                display_name = safe_title
            name = display_name.rsplit(".", 1)[0] + " (vocal only).wav"
            self._send_file(f, "audio/wav", name)
            return

        if p.path == "/final_video":
            q = parse_qs(p.query)
            jid = (q.get("id") or [""])[0]
            j = get_job(jid)
            take = _get_take(jid, (q.get("take") or [""])[0])
            video = (take.get("video") or {}) if take else None
            if take:
                video_file = video.get("final_video_file") if video.get("status") == "done" else None
            else:
                video_file = j.get("final_video_file") if j.get("video_status") == "done" else None
            if not video_file:
                video_file = _session_outputs(jid)["video"]
                if not video_file:
                    self._json(404, {"error": "not ready"})
                    return
            display_name = (take.get("render") or {}).get("display_name") if take else j.get("display_name")
            if not display_name:
                meta = _read_meta(jid)
                display_name = re.sub(r'[\\/:*?"<>|\r\n]+', "_", meta.get("title") or "karaoke")[:100].strip() or "karaoke"
            f = os.path.join(SESS_DIR, jid, video_file)
            name = display_name.rsplit(".", 1)[0] + ".mp4"
            self._send_file(f, "video/mp4", name)
            return

        if p.path == "/phone":
            b = PHONE_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
            return

        if p.path == "/phone_sync_info":
            jid = (parse_qs(p.query).get("id") or [""])[0]
            lan = _lan_ip()
            # Phone browsers refuse camera/mic access on plain http to a LAN
            # IP (not a "secure context"), so this only offers a phone_url
            # once the HTTPS listener on PHONE_PORT is actually up.
            phone_url = f"https://{lan}:{PHONE_PORT}/phone?id={jid}" if (lan and PHONE_HTTPS_READY) else None
            self._json(200, {
                "lan_ip": lan,
                "port": PORT,
                "phone_url": phone_url,
                "https_unavailable": bool(lan) and not PHONE_HTTPS_READY,
            })
            return

        if p.path == "/party":
            b = PARTY_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
            return

        if p.path == "/join":
            b = JOIN_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
            return

        if p.path == "/host":
            b = HOST_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
            return

        if p.path == "/room/state":
            self._json(200, room_state())
            return

        if p.path == "/room/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = pyqueue.Queue()
            with SUB_LOCK:
                SUBSCRIBERS.append(q)
            try:
                self.wfile.write(f"data: {json.dumps(room_state())}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        payload = q.get(timeout=15)
                        self.wfile.write(f"data: {payload}\n\n".encode())
                    except pyqueue.Empty:
                        self.wfile.write(b": keep-alive\n\n")  # comment ping
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with SUB_LOCK:
                    if q in SUBSCRIBERS:
                        SUBSCRIBERS.remove(q)
            return

        if p.path == "/qr.png":
            data = (parse_qs(p.query).get("data") or [""])[0]
            png = make_qr_png(data) if data else None
            if not png:
                self._json(404, {"error": "qrencode not available"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
            return

        # ---- WebRTC signaling for the live phone-camera self-view -----------
        # The server only relays these small SDP/ICE messages (poll-based) —
        # the actual video never touches it, it flows peer-to-peer once the
        # connection is up. The phone (sender, has the camera) creates the
        # offer; the desktop (receiver)
        # answers. webrtc_gen lets either side detect a fresh phone
        # connection and rebuild its peer connection instead of reusing one
        # tied to a camera stream that no longer exists.
        if p.path == "/webrtc/offer":
            jid = (parse_qs(p.query).get("id") or [""])[0]
            j = get_job(jid)
            if not j:
                self._json(404, {"error": "unknown session"})
                return
            self._json(200, {"gen": j.get("webrtc_gen") or 0, "offer": j.get("webrtc_offer")})
            return

        if p.path == "/webrtc/answer":
            jid = (parse_qs(p.query).get("id") or [""])[0]
            j = get_job(jid)
            if not j:
                self._json(404, {"error": "unknown session"})
                return
            self._json(200, {"gen": j.get("webrtc_gen") or 0, "answer": j.get("webrtc_answer")})
            return

        if p.path == "/webrtc/ice":
            q = parse_qs(p.query)
            jid = (q.get("id") or [""])[0]
            src = (q.get("from") or [""])[0]
            since = int((q.get("since") or ["0"])[0] or 0)
            j = get_job(jid)
            if not j or src not in ("phone", "desktop"):
                self._json(404, {"error": "unknown session"})
                return
            candidates = j.get("webrtc_ice_" + src) or []
            self._json(200, {"gen": j.get("webrtc_gen") or 0, "candidates": candidates[since:], "total": len(candidates)})
            return

        if p.path == "/takes":
            jid = (parse_qs(p.query).get("id") or [""])[0]
            self._json(200, {"takes": list_takes(jid)})
            return

        if p.path == "/take_audio":
            q = parse_qs(p.query)
            jid = (q.get("id") or [""])[0]
            tid = (q.get("take") or [""])[0]
            take = _get_take(jid, tid)
            if not take:
                self._json(404, {"error": "unknown take"})
                return
            self._send_file(os.path.join(SESS_DIR, jid, take["file"]), "audio/wav")
            return

        if p.path == "/lights/discover":
            devices = govee_lights.discover()
            with LIGHTS_LOCK:
                LIGHTS_DISCOVERED.update(d["ip"] for d in devices)
                selected = set(LIGHTS_SELECTED)
            for d in devices:
                d["selected"] = d["ip"] in selected
            self._json(200, {"devices": devices})
            return

        self._json(404, {"error": "not found"})

    def _send_file(self, path, ctype, dl_name=None):
        if not os.path.isfile(path):
            self._json(404, {"error": "gone"})
            return
        size = os.path.getsize(path)

        # Honor HTTP Range requests so <audio> can seek within the file.
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        is_partial = False
        if range_header and range_header.startswith("bytes="):
            try:
                rng = range_header.split("=", 1)[1].split(",")[0].strip()
                s, _, e = rng.partition("-")
                if s.strip():
                    start = int(s)
                    end = int(e) if e.strip() else size - 1
                else:
                    # suffix range: bytes=-N  -> last N bytes
                    n = int(e)
                    start = max(0, size - n)
                    end = size - 1
                if start > end or start >= size:
                    # unsatisfiable
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                end = min(end, size - 1)
                is_partial = True
            except (ValueError, IndexError):
                start, end, is_partial = 0, size - 1, False

        length = end - start + 1
        self.send_response(206 if is_partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if is_partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if dl_name:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{dl_name}"')
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break  # client seeked away / closed — normal for media
                remaining -= len(chunk)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw or b"{}")

    def do_POST(self):
        p = urlparse(self.path)

        if p.path == "/prepare":
            body = self._read_body()
            url = (body.get("url") or "").strip()
            if not url.lower().startswith("http"):
                self._json(400, {"error": "Paste a valid YouTube URL."})
                return
            profile = _safe_profile_name(body.get("profile")) or None
            strip_vocals = bool(body.get("strip_vocals"))
            jid = uuid.uuid4().hex[:12]
            set_job(jid, status="queued")
            threading.Thread(target=do_prepare, args=(jid, url, profile, strip_vocals), daemon=True).start()
            self._json(200, {"id": jid})
            return

        if p.path == "/profile/library/add":
            body = self._read_body()
            name = _safe_profile_name(body.get("name"))
            video_id = (body.get("video_id") or "").strip()
            if not name or not video_id:
                self._json(400, {"error": "Missing profile name or song."})
                return
            prof = _read_profile(name)
            lib = prof.get("library") or []
            if not any(e.get("video_id") == video_id for e in lib):
                lib.append({
                    "video_id": video_id,
                    "title": (body.get("title") or "").strip() or "Karaoke",
                    "thumb": body.get("thumb") or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                    "channel": (body.get("channel") or "").strip(),
                    "added_at": time.time(),
                })
            prof = _write_profile(name, library=lib)
            self._json(200, prof)
            return

        if p.path == "/profile/library/remove":
            body = self._read_body()
            name = _safe_profile_name(body.get("name"))
            video_id = (body.get("video_id") or "").strip()
            if not name:
                self._json(400, {"error": "Missing profile name."})
                return
            prof = _read_profile(name)
            lib = [e for e in (prof.get("library") or []) if e.get("video_id") != video_id]
            prof = _write_profile(name, library=lib)
            self._json(200, prof)
            return

        if p.path == "/profile/ensure":
            body = self._read_body()
            name = _safe_profile_name(body.get("name"))
            if not name:
                self._json(400, {"error": "Missing account name."})
                return
            # optional, local-only — just stored for display in the account
            # menu (e.g. "signed in as alice@example.com"). No verification,
            # no OAuth, no outbound network call of any kind.
            email = (body.get("email") or "").strip()[:120]
            if "email" in body:
                prof = _write_profile(name, email=email)
            else:
                prof = _write_profile(name)
            self._json(200, prof)
            return

        if p.path == "/profile/rename":
            body = self._read_body()
            old_name = _safe_profile_name(body.get("old_name"))
            new_name = _safe_profile_name(body.get("new_name"))
            if not old_name or not new_name:
                self._json(400, {"error": "Missing account name."})
                return
            prof = _rename_profile(old_name, new_name)
            self._json(200, prof)
            return

        if p.path == "/profile/delete":
            body = self._read_body()
            name = _safe_profile_name(body.get("name"))
            if not name:
                self._json(400, {"error": "Missing account name."})
                return
            _delete_profile(name)
            self._json(200, {"ok": True})
            return

        if p.path == "/profile/library/edit":
            body = self._read_body()
            name = _safe_profile_name(body.get("name"))
            video_id = (body.get("video_id") or "").strip()
            title = (body.get("title") or "").strip()
            if not name or not video_id or not title:
                self._json(400, {"error": "Missing profile name, song, or title."})
                return
            prof = _read_profile(name)
            lib = prof.get("library") or []
            found = False
            for e in lib:
                if e.get("video_id") == video_id:
                    e["title"] = title[:200]
                    found = True
            if not found:
                self._json(404, {"error": "Song not in library."})
                return
            prof = _write_profile(name, library=lib)
            self._json(200, prof)
            return

        if p.path == "/import":
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = 0
            if length <= 0:
                self._json(400, {"error": "No file received."})
                return
            if length > MAX_IMPORT_BYTES:
                self._json(413, {"error": "That file is too large (500MB max)."})
                return
            filename = unquote(self.headers.get("X-Filename", "") or "")
            ext = os.path.splitext(filename)[1][:10] or ".upload"
            ext = re.sub(r"[^A-Za-z0-9.]", "", ext) or ".upload"
            jid = uuid.uuid4().hex[:12]
            sdir = os.path.join(SESS_DIR, jid)
            os.makedirs(sdir, exist_ok=True)
            upload_path = os.path.join(sdir, "upload" + ext)
            try:
                with open(upload_path, "wb") as fh:
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(65536, remaining))
                        if not chunk:
                            break
                        fh.write(chunk)
                        remaining -= len(chunk)
            except OSError:
                self._json(500, {"error": "Could not save the uploaded file."})
                return
            set_job(jid, status="queued")
            threading.Thread(target=do_import, args=(jid, upload_path, filename),
                             daemon=True).start()
            self._json(200, {"id": jid})
            return

        if p.path == "/render":
            body = self._read_body()
            jid = body.get("id")
            j = job_or_disk(jid)
            if not j:
                self._json(404, {"error": "unknown session"})
                return
            sdir = os.path.join(SESS_DIR, jid)
            os.makedirs(sdir, exist_ok=True)

            # vocal comes either as an inline base64 recording, or by
            # referencing an already-saved take (from the take-history strip)
            take_id = body.get("take_id")
            if take_id:
                take = _get_take(jid, take_id)
                if not take:
                    self._json(404, {"error": "unknown take"})
                    return
                vocal_path = os.path.join(sdir, take["file"])
            else:
                b64 = body.get("vocal_wav_b64", "")
                if not b64:
                    self._json(400, {"error": "No vocal recording received."})
                    return
                vocal_path = os.path.join(sdir, "vocal_raw.wav")
                try:
                    with open(vocal_path, "wb") as fh:
                        fh.write(base64.b64decode(b64))
                except Exception:
                    self._json(400, {"error": "Could not decode recording."})
                    return

            # scope_id ties this render's OUTPUT files (and, if a take
            # record exists for it, that take's own progress/result) to one
            # specific take instead of a single session-wide "final.wav"
            # slot — otherwise recording a retake while an earlier render or
            # video combine is still in flight clobbers or gets confused
            # with that earlier one. take_id covers an already-saved take;
            # rec_id (the id generated client-side at record-start) covers
            # rendering directly from a fresh, not-yet-saved recording.
            rec_id = body.get("rec_id")
            rec_id = rec_id if re.fullmatch(r"[A-Za-z0-9]{4,32}", rec_id or "") else None
            scope_id = take_id if (take_id and re.fullmatch(r"[A-Za-z0-9]{4,32}", take_id)) else rec_id

            lock = _render_lock(jid)
            if not lock.acquire(blocking=False):
                self._json(409, {"error": "A render is already running for this session."})
                return
            fx = body.get("fx", {})
            mix_opts = body.get("mix", {})
            set_job(jid, status="rendering")
            if scope_id and _get_take(jid, scope_id):
                _update_take(jid, scope_id, render={"status": "rendering", "stage": "starting", "error": None})
            threading.Thread(target=_run_render_locked,
                             args=(jid, vocal_path, fx, mix_opts, lock, scope_id),
                             daemon=True).start()
            self._json(200, {"ok": True, "scope_id": scope_id})
            return

        if p.path == "/phone/register":
            body = self._read_body()
            jid = body.get("id")
            j = job_or_disk(jid)
            if not j:
                self._json(404, {"error": "unknown session"})
                return
            # Bump webrtc_gen so the desktop's live self-view knows this is a
            # fresh phone connection (new camera stream, or a reload/re-scan)
            # and tears down + rebuilds its peer connection instead of trying
            # to reuse one tied to a camera that no longer exists.
            gen = (j.get("webrtc_gen") or 0) + 1
            set_job(jid, phone_paired=True, phone_status="paired",
                    phone_last_seen=time.time(),
                    webrtc_gen=gen, webrtc_offer=None, webrtc_answer=None,
                    webrtc_ice_phone=[], webrtc_ice_desktop=[])
            self._json(200, {"ok": True, "webrtc_gen": gen})
            return

        if p.path == "/phone/disconnect":
            # Lets the desktop end the pairing on demand ("Disconnect phone"
            # button) without needing to touch the phone — the phone's own
            # poll loop notices phone_paired flipping to False and tears down
            # its camera/peer connection on its own next tick.
            body = self._read_body()
            jid = body.get("id")
            if not job_or_disk(jid):
                self._json(404, {"error": "unknown session"})
                return
            set_job(jid, phone_paired=False, phone_status="disconnected")
            self._json(200, {"ok": True})
            return

        if p.path == "/phone/upload":
            # The phone posts the raw video blob as the request body (id/mime
            # as query params, not JSON) — a base64-in-JSON body used to be
            # required here, but that inflates a multi-minute 1080p recording
            # (~150-250MB) by another ~33% and forces both the phone's
            # FileReader and this handler to hold a full extra copy in
            # memory, which reliably failed on real devices. Stream straight
            # to disk in chunks instead of reading the whole body at once.
            q = parse_qs(p.query)
            jid = (q.get("id") or [""])[0]
            mime = (q.get("mime") or ["video/webm"])[0]
            # `rec`: the client-generated id for THIS recording (set at
            # startRec() time, before either MediaRecorder starts) — saving
            # each take's phone footage under its own filename, keyed by
            # this id, is what lets a retake's video combine independently
            # instead of racing/clobbering whatever the previous take's
            # combine was doing with a single shared "phone_raw" file.
            rec = (q.get("rec") or [""])[0]
            rec = rec if re.fullmatch(r"[A-Za-z0-9]{4,32}", rec or "") else None
            j = job_or_disk(jid)
            if not j:
                self._json(404, {"error": "unknown session"})
                return
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                self._json(400, {"error": "No video received."})
                return
            ext = ".mp4" if "mp4" in mime else ".webm"
            sdir = os.path.join(SESS_DIR, jid)
            os.makedirs(sdir, exist_ok=True)
            fname = f"phone_{rec}{ext}" if rec else f"phone_raw{ext}"
            video_path = os.path.join(sdir, fname)
            try:
                remaining = length
                with open(video_path, "wb") as fh:
                    while remaining > 0:
                        chunk = self.rfile.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        fh.write(chunk)
                        remaining -= len(chunk)
            except Exception:
                self._json(400, {"error": "Could not save video."})
                return
            phone_videos = dict(j.get("phone_videos") or {})
            if rec:
                phone_videos[rec] = os.path.basename(video_path)
            set_job(jid, phone_status="uploaded",
                    phone_video_file=os.path.basename(video_path),  # legacy/global fallback
                    phone_videos=phone_videos)
            self._json(200, {"ok": True})
            return

        if p.path == "/webrtc/offer":
            body = self._read_body()
            jid = body.get("id")
            j = get_job(jid)
            if not j:
                self._json(404, {"error": "unknown session"})
                return
            set_job(jid, webrtc_offer=body.get("sdp"))
            self._json(200, {"ok": True})
            return

        if p.path == "/webrtc/answer":
            body = self._read_body()
            jid = body.get("id")
            j = get_job(jid)
            if not j:
                self._json(404, {"error": "unknown session"})
                return
            set_job(jid, webrtc_answer=body.get("sdp"))
            self._json(200, {"ok": True})
            return

        if p.path == "/webrtc/ice":
            body = self._read_body()
            jid = body.get("id")
            src = body.get("from")
            j = get_job(jid)
            if not j or src not in ("phone", "desktop"):
                self._json(404, {"error": "unknown session"})
                return
            key = "webrtc_ice_" + src
            candidates = (j.get(key) or []) + [body.get("candidate")]
            set_job(jid, **{key: candidates})
            self._json(200, {"ok": True})
            return

        if p.path == "/combine_video":
            body = self._read_body()
            jid = body.get("id")
            j = get_job(jid)
            if not j:
                self._json(409, {"error": "not ready"})
                return
            take_id = body.get("take_id")
            rec_id = body.get("rec_id")
            scope_id = next((s for s in (take_id, rec_id)
                              if s and re.fullmatch(r"[A-Za-z0-9]{4,32}", s)), None)
            take = _get_take(jid, scope_id) if scope_id else None

            # Resolve THIS take's own render result and phone footage rather
            # than the single session-wide slot — otherwise a retake here
            # would either combine against a stale take's finished mix, or
            # (via the old status guard below) get silently skipped entirely
            # because some OTHER take had already reached "done"/"aligning".
            if take:
                render = take.get("render") or {}
                render_done = render.get("status") == "done"
                final_file = render.get("audio_master_file") or render.get("final_file")
                video_state = take.get("video") or {}
            else:
                render_done = j.get("status") == "done"
                final_file = j.get("audio_master_file") or j.get("final_file")
                video_state = j  # legacy: global video_status/video_error fields

            phone_video_file = (j.get("phone_videos") or {}).get(scope_id) or j.get("phone_video_file")

            if not render_done or not final_file or not phone_video_file:
                self._json(409, {"error": "not ready"})
                return
            if video_state.get("video_status" if take is None else "status") in ("aligning", "done"):
                self._json(200, {"ok": True})  # already running / done for THIS take, avoid double-combine
                return

            if take:
                _update_take(jid, scope_id, video={"status": "aligning", "error": None})
            else:
                set_job(jid, video_status="aligning", video_error=None)
            threading.Thread(target=do_combine_video,
                             args=(jid, scope_id, phone_video_file, final_file),
                             daemon=True).start()
            self._json(200, {"ok": True})
            return

        if p.path == "/take":
            body = self._read_body()
            jid = body.get("id")
            j = job_or_disk(jid)
            if not j:
                self._json(404, {"error": "unknown session"})
                return
            b64 = body.get("vocal_wav_b64", "")
            if not b64:
                self._json(400, {"error": "No recording received."})
                return
            sdir = os.path.join(SESS_DIR, jid)
            os.makedirs(sdir, exist_ok=True)
            # rec_id: the same client-generated id used to tag this take's
            # phone video upload (see /phone/upload). Reusing it as the
            # take's own id means a take and "its" phone footage share one
            # identifier from the moment recording started, with no separate
            # linking step needed later.
            rec_id = body.get("rec_id")
            rec_id = rec_id if re.fullmatch(r"[A-Za-z0-9]{4,32}", rec_id or "") else None
            tid = rec_id if (rec_id and not _get_take(jid, rec_id)) else uuid.uuid4().hex[:8]
            fname = f"take_{tid}.wav"
            fpath = os.path.join(sdir, fname)
            try:
                with open(fpath, "wb") as fh:
                    fh.write(base64.b64decode(b64))
            except Exception:
                self._json(400, {"error": "Could not decode recording."})
                return
            take = _add_take(
                jid, file=fname, duration=_probe_duration(fpath),
                kind=body.get("kind", "lead"),
                fx_snapshot=body.get("fx_snapshot"),
                punch_sec=body.get("punch_sec"),
                source_take_ids=body.get("source_take_ids"),
                tid=tid,
                pitch_score=body.get("pitch_score"),
                latency_ms=body.get("latency_ms"),
            )
            self._json(200, {"take": take})
            return

        if p.path == "/take_delete":
            body = self._read_body()
            jid = body.get("id")
            tid = body.get("take")
            ok = _del_take(jid, tid)
            if not ok:
                self._json(404, {"error": "unknown take"})
                return
            self._json(200, {"ok": True})
            return

        if p.path == "/take_restore":
            body = self._read_body()
            jid = body.get("id")
            tid = body.get("take")
            ok = _restore_take(jid, tid)
            if not ok:
                self._json(404, {"error": "unknown take"})
                return
            self._json(200, {"take": _get_take(jid, tid)})
            return

        if p.path == "/take_update":
            # Generic small-field patch for a saved take — today just the
            # alignment nudge and a re-saved FX snapshot, both editable from
            # the Track Editor after the take already exists on disk (unlike
            # the WAV itself, which is written once and never touched again).
            body = self._read_body()
            jid = body.get("id")
            tid = body.get("take")
            fields = {}
            if "align_ms" in body:
                try:
                    fields["align_ms"] = max(-2000.0, min(2000.0, float(body.get("align_ms") or 0)))
                except (TypeError, ValueError):
                    fields["align_ms"] = 0.0
            if "fx_snapshot" in body:
                fields["fx_snapshot"] = body.get("fx_snapshot")
            if not fields:
                self._json(400, {"error": "nothing to update"})
                return
            ok = _update_take(jid, tid, **fields)
            if not ok:
                self._json(404, {"error": "unknown take"})
                return
            self._json(200, {"take": _get_take(jid, tid)})
            return

        if p.path == "/session_delete":
            # There was no way to reclaim disk space at all before this —
            # every take, raw phone video, and render just accumulated
            # forever (sessions/ reached 7.8GB across 74 sessions with zero
            # cleanup anywhere). This is the manual side of fixing that;
            # do_render/do_combine_video separately clean up their own
            # provably-dead intermediates automatically.
            body = self._read_body()
            jid = body.get("id")
            if not jid or not re.fullmatch(r"[A-Za-z0-9]{6,32}", jid):
                self._json(400, {"error": "invalid session id"})
                return
            sdir = os.path.join(SESS_DIR, jid)
            if not os.path.isdir(sdir):
                self._json(404, {"error": "unknown session"})
                return
            shutil.rmtree(sdir, ignore_errors=True)
            with LOCK:
                JOBS.pop(jid, None)
            self._json(200, {"ok": True})
            return

        if p.path == "/sessions_delete":
            # Bulk form of /session_delete — one confirm, one request,
            # instead of the History page firing N sequential deletes for
            # a multiselect action. Best-effort per id: one bad/missing id
            # in the batch doesn't abort the rest.
            body = self._read_body()
            ids = [i for i in (body.get("ids") or []) if isinstance(i, str)]
            deleted, failed = [], []
            for jid in ids:
                if not re.fullmatch(r"[A-Za-z0-9]{6,32}", jid):
                    failed.append(jid)
                    continue
                sdir = os.path.join(SESS_DIR, jid)
                if not os.path.isdir(sdir):
                    failed.append(jid)
                    continue
                shutil.rmtree(sdir, ignore_errors=True)
                with LOCK:
                    JOBS.pop(jid, None)
                deleted.append(jid)
            self._json(200, {"ok": True, "deleted": deleted, "failed": failed})
            return

        if p.path == "/comp_multi":
            # Generalizes /comp from exactly 2 takes at 1 punch point to an
            # arbitrary ordered list of (take, region) segments — pick a
            # different take for each stretch of the song instead of one
            # head + one tail take.
            body = self._read_body()
            jid = body.get("id")
            raw_segments = body.get("segments") or []
            if len(raw_segments) < 1:
                self._json(400, {"error": "need at least one segment"})
                return
            sdir = os.path.join(SESS_DIR, jid)
            segments = []
            source_ids = []
            for s in raw_segments:
                t = _get_take(jid, s.get("take_id"))
                if not t:
                    self._json(404, {"error": f"unknown take {s.get('take_id')}"})
                    return
                try:
                    start = float(s.get("start", 0))
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
                source_take_ids=source_ids, tid=tid,
            )
            self._json(200, {"take": take})
            return

        if p.path == "/room/start":
            code = start_room()
            base = self._base_url()
            join_url = f"{base}/join?room={code}" if base else None
            lan = _lan_ip()
            if join_url and join_url.startswith("http://") and lan and PHONE_HTTPS_READY:
                # phone browsers refuse the microphone on plain http to a LAN
                # IP - guests MUST join over the HTTPS listener or scoring
                # dies with "mic blocked" the moment their turn starts
                join_url = "https://" + lan + ":" + str(PHONE_PORT) + "/join?room=" + code
            host_url = f"{base}/host" if base else None
            self._json(200, {"code": code, "join_url": join_url, "host_url": host_url})
            return

        if p.path == "/room/join":
            body = self._read_body()
            result = room_join(body.get("name"), body.get("device_id"))
            if not result["ok"]:
                self._json(400, {"error": result["error"]})
                return
            self._json(200, room_state())
            return

        if p.path == "/queue/add":
            body = self._read_body()
            jid = (body.get("jid") or "").strip()
            singers = [s.strip() for s in (body.get("singers") or []) if s.strip()][:6]
            try:
                semitones = max(-6, min(6, int(body.get("semitones") or 0)))
            except (TypeError, ValueError):
                semitones = 0
            if not jid or not singers:
                self._json(400, {"error": "Need at least a song and a singer name."})
                return
            entry = queue_add(jid, singers, semitones)
            if not entry:
                self._json(400, {"error": "No open party room, or song isn't ready yet."})
                return
            self._json(200, {"entry": entry})
            return

        if p.path == "/party/scoring":
            # host toggle: score mode on/off for this party. When turned off
            # mid-song, any in-flight live score is finalized first so a
            # singer's points aren't silently dropped.
            body = self._read_body()
            enabled = bool(body.get("enabled"))
            with ROOM_LOCK:
                if not enabled:
                    _finalize_live_score_locked()
                ROOM["scoring_enabled"] = enabled
            broadcast_room()
            self._json(200, room_state())
            return

        if p.path == "/party/score":
            body = self._read_body()
            device_id = body.get("device_id") or ""
            entry_id = body.get("entry_id") or ""
            with ROOM_LOCK:
                if not ROOM.get("scoring_enabled", True):
                    self._json(409, {"error": "scoring is off for this party"})
                    return
                guest = next((g for g in ROOM["guests"] if g["device_id"] == device_id), None)
                now = next((e for e in ROOM["queue"] if e["entry_id"] == ROOM.get("now_playing")), None)
                # only the phone of someone actually SINGING the current song
                # can post scores for it
                if not guest or not now or now["entry_id"] != entry_id or guest["name"] not in now["singers"]:
                    self._json(403, {"error": "not your turn"})
                    return
                ROOM["live_score"] = {
                    "name": guest["name"], "singers": list(now["singers"]),
                    "entry_id": entry_id,
                    "points": max(0, int(body.get("points") or 0)),
                    "form": max(0, min(100, int(body.get("form") or 0))),
                    "mult": max(1, min(4, int(body.get("mult") or 1))),
                    "accuracy": max(0, min(100, int(body.get("accuracy") or 0))),
                    "rank": (body.get("rank") or "D")[:1],
                }
                done = bool(body.get("done"))
                if done:
                    _finalize_live_score_locked()
            broadcast_room()
            self._json(200, {"ok": True})
            return

        if p.path == "/queue/next":
            queue_advance()
            self._json(200, room_state())
            return

        if p.path == "/queue/remove":
            body = self._read_body()
            queue_remove(body.get("entry_id"))
            self._json(200, room_state())
            return

        if p.path == "/queue/move":
            body = self._read_body()
            direction = body.get("direction")
            if direction not in ("up", "down"):
                self._json(400, {"error": "direction must be 'up' or 'down'"})
                return
            queue_move(body.get("entry_id"), direction)
            self._json(200, room_state())
            return

        if p.path == "/queue/reorder":
            body = self._read_body()
            order = body.get("order")
            if not isinstance(order, list) or not order:
                self._json(400, {"error": "order must be a non-empty list of entry_ids"})
                return
            if not queue_reorder(order):
                self._json(409, {"error": "queue changed — reload and try again"})
                return
            self._json(200, room_state())
            return

        if p.path == "/challenge/add":
            body = self._read_body()
            jid = (body.get("jid") or "").strip()
            from_singer = (body.get("from_singer") or "").strip()
            to_singer = (body.get("to_singer") or "").strip()
            try:
                semitones = max(-6, min(6, int(body.get("semitones") or 0)))
            except (TypeError, ValueError):
                semitones = 0
            if not jid or not from_singer or not to_singer:
                self._json(400, {"error": "Need a song, your name, and who you're challenging."})
                return
            challenge = challenge_add(jid, from_singer, to_singer, semitones)
            if not challenge:
                self._json(400, {"error": "No open party room, or song isn't ready yet."})
                return
            self._json(200, {"challenge": challenge})
            return

        if p.path == "/challenge/accept":
            body = self._read_body()
            entry, err = challenge_accept(body.get("challenge_id"), body.get("device_id"))
            if not entry:
                self._json(403 if "only" in (err or "") else 404, {"error": err})
                return
            self._json(200, room_state())
            return

        if p.path == "/challenge/decline":
            body = self._read_body()
            if not challenge_decline(body.get("challenge_id"), body.get("device_id")):
                self._json(403, {"error": "only the challenged person (or the host) can decline this"})
                return
            self._json(200, room_state())
            return

        if p.path == "/lights/select":
            body = self._read_body()
            requested = [ip for ip in (body.get("ips") or []) if isinstance(ip, str)]
            with LIGHTS_LOCK:
                ips = [ip for ip in requested if ip in LIGHTS_DISCOVERED]
                LIGHTS_SELECTED.clear()
                LIGHTS_SELECTED.update(ips)
            self._json(200, {"ok": True, "selected": ips})
            return

        if p.path == "/lights/update":
            body = self._read_body()
            with LIGHTS_LOCK:
                targets = list(LIGHTS_SELECTED)
            if not targets:
                self._json(200, {"ok": True, "targets": 0})
                return
            midi = body.get("midi")
            cents = body.get("cents") or 0
            level = body.get("level") or 0
            mode = body.get("mode") if body.get("mode") in ("multi", "mono") else "multi"
            mono_color = body.get("color") if mode == "mono" else None
            (r, g, b), brightness = _pitch_to_rgb(midi, cents, level, mode, mono_color)
            for ip in targets:
                if brightness <= 0:
                    continue
                govee_lights.set_color(ip, r, g, b)
                govee_lights.set_brightness(ip, brightness)
            self._json(200, {"ok": True, "targets": len(targets), "rgb": [r, g, b], "brightness": brightness})
            return

        if p.path == "/lights/test":
            body = self._read_body()
            ip = body.get("ip")
            if not ip:
                self._json(400, {"error": "missing ip"})
                return
            with LIGHTS_LOCK:
                known = ip in LIGHTS_DISCOVERED
            if not known:
                self._json(403, {"error": "unknown device — run discover first"})
                return
            govee_lights.turn(ip, True)
            govee_lights.set_color(ip, 0, 255, 120)
            govee_lights.set_brightness(ip, 80)
            self._json(200, {"ok": True})
            return

        if p.path == "/lights/off":
            body = self._read_body()
            with LIGHTS_LOCK:
                targets = list(LIGHTS_SELECTED)
                requested_ips = body.get("ips")
                if requested_ips:
                    targets = [ip for ip in requested_ips if isinstance(ip, str) and ip in LIGHTS_DISCOVERED]
            for ip in targets:
                govee_lights.turn(ip, False)
            self._json(200, {"ok": True})
            return

        self._json(404, {"error": "not found"})


PHONE_HTML = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>MicDrop — Phone Camera</title>
<style>
  html,body{height:100%;margin:0;background:#0b0b10;color:#eee;
    font:15px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;}
  #wrap{display:flex;flex-direction:column;height:100%;}
  #preview{flex:1;width:100%;height:100%;object-fit:cover;background:#000;}
  #zoomRow{display:flex;align-items:center;gap:10px;}
  #zoomRow input[type=range]{flex:1;accent-color:#f5a623;height:28px;}
  #zoomVal{min-width:44px;text-align:right;font-variant-numeric:tabular-nums;font-size:14px;}
  #bar{padding:14px 16px calc(14px + env(safe-area-inset-bottom));
    background:#14141c;display:flex;flex-direction:column;gap:8px;}
  #status{font-weight:600;font-size:16px;}
  #hint{opacity:.7;font-size:13px;}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;
    background:#555;margin-right:8px;vertical-align:middle;}
  .dot.rec{background:#ff4d4f;box-shadow:0 0 8px #ff4d4f;}
  .dot.ok{background:#4ade80;}
  .dot.warn{background:#f5a623;}
  #flip{position:absolute;top:calc(14px + env(safe-area-inset-top));right:14px;
    background:rgba(0,0,0,.5);color:#fff;border:1px solid #444;border-radius:20px;
    padding:8px 14px;font-size:14px;}
  #err{color:#ff8080;font-size:13px;}
</style>
<div id="wrap">
  <div style="position:relative;flex:1;overflow:hidden;">
    <!-- camVideo is the raw camera sink, never shown directly — it's only a
         source for the zoom canvas below. Both the on-screen preview and
         the WebRTC stream sent to the desktop are the CANVAS's output, so
         zoom is a real crop+scale of every frame, not a cosmetic CSS effect
         layered on top of an unzoomed recording. -->
    <video id="camVideo" autoplay muted playsinline style="display:none"></video>
    <canvas id="preview"></canvas>
    <button id="flip" onclick="flipCamera()">🔄 flip</button>
  </div>
  <div id="bar">
    <div id="status"><span class="dot" id="dot"></span><span id="statusText">Starting camera…</span></div>
    <div id="zoomRow">
      <span>🔍</span>
      <input type="range" id="zoomSlider" min="1" max="8" step="0.1" value="1">
      <span id="zoomVal">1.0x</span>
    </div>
    <div id="hint">Keep this tab open while you record — if your screen locks between takes,
      this page reconnects the camera on its own once you unlock it, no rescanning needed.</div>
    <button id="reconnectBtn" onclick="manualReconnect()" style="display:none">🔄 Reconnect</button>
    <div id="err"></div>
  </div>
</div>
<script>
const params = new URLSearchParams(location.search);
const id = params.get('id');
const statusText = document.getElementById('statusText');
const dot = document.getElementById('dot');
const errEl = document.getElementById('err');
const camVideo = document.getElementById('camVideo');
const canvas = document.getElementById('preview');
const ctx = canvas.getContext('2d');
let stream = null, facing = 'environment';
let wakeLock = null;
let zoomLevel = 1;

// The canvas is both what the singer sees and what actually gets sent over
// WebRTC (see startWebRTC below) — capturing at a fixed 30fps regardless of
// how often the draw loop repaints keeps the outgoing track's frame rate
// stable even if a slow phone occasionally skips a rAF tick.
const canvasStream = canvas.captureStream(30);

function drawFrame(){
  const vw = camVideo.videoWidth, vh = camVideo.videoHeight;
  if(vw && vh){
    if(canvas.width !== vw || canvas.height !== vh){ canvas.width = vw; canvas.height = vh; }
    const cropW = vw / zoomLevel, cropH = vh / zoomLevel;
    const sx = (vw - cropW) / 2, sy = (vh - cropH) / 2;
    ctx.drawImage(camVideo, sx, sy, cropW, cropH, 0, 0, vw, vh);
  }
  requestAnimationFrame(drawFrame);
}
requestAnimationFrame(drawFrame);

const zoomSlider = document.getElementById('zoomSlider');
const zoomVal = document.getElementById('zoomVal');
zoomSlider.addEventListener('input', ()=>{
  zoomLevel = parseFloat(zoomSlider.value) || 1;
  zoomVal.textContent = zoomLevel.toFixed(1) + 'x';
});

function setStatus(text, cls){
  statusText.textContent = text;
  dot.className = 'dot' + (cls ? ' ' + cls : '');
}

function showErr(msg){ errEl.textContent = msg || ''; }

if(!id){
  setStatus('No session id in link.', 'warn');
  showErr('Open this link from the pairing panel in the desktop app.');
} else {
  startCamera();
}

// The phone's only job now is to be a live camera source over WebRTC — the
// desktop records the incoming stream itself (see startRec()/stopRec() in
// the main app), started in the same tick as the vocal recording so the two
// share one clock instead of needing a separate phone-side recording +
// upload + audio-chirp alignment pass after the fact. That old design
// required the sync chirp to be audible to the phone's own mic through a
// real speaker — silently broken for anyone monitoring on headphones — and
// a full-length video upload from the phone at the end, which is exactly
// the "video not even getting recorded" failure this replaces.
async function startCamera(){
  try{
    await acquireCameraAndConnect();
    pollLoop();   // started exactly once, for the lifetime of this page
  }catch(e){
    setStatus('Camera access failed', 'warn');
    showErr((e && e.message) || String(e));
  }
}

// The actual get-camera + register + WebRTC-connect steps, split out from
// startCamera() so the same sequence can re-run later without starting a
// second concurrent pollLoop() (register()/startWebRTC() already close any
// previous peer connection, so this is safe to call again at any time).
async function acquireCameraAndConnect(){
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: facing, width: {ideal:1920}, height: {ideal:1080} },
    audio: false,   // no phone-side audio needed anymore — nothing here
                     // ever gets used for sync or for the final mix.
  });
  camVideo.srcObject = stream;
  showErr('');
  $('reconnectBtn').style.display='none';
  await register();
  requestWakeLock();
  setStatus('Connected — streaming to desktop', 'ok');
}
function $(id){ return document.getElementById(id); }
async function manualReconnect(){
  setStatus('Reconnecting…', 'warn');
  try{ await acquireCameraAndConnect(); }
  catch(e){ setStatus('Camera access failed', 'warn'); showErr((e && e.message) || String(e)); }
}

// Swap the camera in place (new getUserMedia, pointed at the same hidden
// camVideo sink the zoom canvas already reads from) rather than tearing
// down and re-registering — a full re-register would bump webrtc_gen and
// force the desktop to rebuild its peer connection, causing a visible
// glitch in the live self-view every time you just want to flip front/back
// camera. No sender.replaceTrack needed either: the WebRTC sender's track
// is the canvas's own captureStream track (see startWebRTC), which keeps
// drawing whatever camVideo shows regardless of which physical camera feeds
// it — swapping camVideo.srcObject is the entire flip.
async function flipCamera(){
  facing = facing === 'environment' ? 'user' : 'environment';
  let newStream;
  try{
    newStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: {ideal:1920}, height: {ideal:1080} }, audio: false });
  }catch(e){ showErr((e && e.message) || String(e)); return; }
  const oldStream = stream;
  stream = newStream;
  camVideo.srcObject = stream;
  if(oldStream) oldStream.getTracks().forEach(t=>t.stop());
}

async function register(){
  try{
    const r = await fetch('/phone/register', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({id})});
    if(r.ok){
      const j = await r.json();
      startWebRTC(j.webrtc_gen || 1);
    }
  }catch(_){}
}

// ---- WebRTC live self-view: phone is the sender (it has the camera), the
// desktop is the receiver. The server only relays the small SDP/ICE
// signaling messages below (poll-based, same pattern as the start/stop
// commands) — actual video goes peer-to-peer once connected, never through
// the server. No STUN/TURN configured: both sides are on the same LAN, so
// direct host candidates are expected to work.
let pc = null, answerApplied = false, iceRecvCount = 0;
async function startWebRTC(gen){
  if(pc){ try{ pc.close(); }catch(_){} pc = null; }
  answerApplied = false; iceRecvCount = 0;
  pc = new RTCPeerConnection({iceServers: []});
  // Sending the CANVAS's track (post-zoom), not the raw camera track — the
  // desktop should receive exactly the cropped/scaled frame the singer sees
  // in the phone preview, since that's what gets recorded as the final
  // video too.
  canvasStream.getVideoTracks().forEach(t=>pc.addTrack(t, canvasStream));
  pc.onicecandidate = (e)=>{ if(e.candidate) postIce(e.candidate.toJSON()); };
  try{
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await fetch('/webrtc/offer', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({id, sdp:{type:pc.localDescription.type, sdp:pc.localDescription.sdp}})});
  }catch(_){}
}
async function postIce(candidate){
  try{
    await fetch('/webrtc/ice', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({id, from:'phone', candidate})});
  }catch(_){}
}
async function pollWebRTC(){
  if(!pc) return;
  try{
    if(!answerApplied){
      const r = await fetch('/webrtc/answer?id='+encodeURIComponent(id));
      if(r.ok){
        const j = await r.json();
        if(j.answer && pc.signalingState==='have-local-offer'){
          await pc.setRemoteDescription(j.answer);
          answerApplied = true;
        }
      }
    }
    const r2 = await fetch('/webrtc/ice?id='+encodeURIComponent(id)+'&from=desktop&since='+iceRecvCount);
    if(r2.ok){
      const j2 = await r2.json();
      for(const c of (j2.candidates||[])){ if(c){ try{ await pc.addIceCandidate(c); }catch(_){} } }
      iceRecvCount = j2.total;
    }
  }catch(_){}
}

async function requestWakeLock(){
  try{
    if('wakeLock' in navigator) wakeLock = await navigator.wakeLock.request('screen');
  }catch(_){}  // best-effort; not supported on iOS Safari — user keeps screen on manually
}
// Locking the screen (or switching apps) backgrounds this tab, and mobile
// browsers — iOS Safari especially, where the Wake Lock API above doesn't
// even exist — routinely kill the camera track while backgrounded rather
// than just pausing it. Without this, that meant walking back to the phone
// and re-scanning the QR code after every take. Instead: the instant the
// page becomes visible again, check whether the video track actually
// survived, and if not, silently re-run the exact same get-camera/register/
// reconnect sequence startCamera() used the first time — no rescan needed,
// the desktop picks up the new webrtc_gen automatically.
document.addEventListener('visibilitychange', async ()=>{
  if(document.visibilityState!=='visible' || !id) return;
  requestWakeLock();
  const track = stream && stream.getVideoTracks()[0];
  if(!track || track.readyState==='ended'){
    setStatus('Reconnecting camera…', 'warn');
    try{ await acquireCameraAndConnect(); }
    catch(e){ setStatus('Camera access failed', 'warn'); showErr((e && e.message) || String(e)); }
  }
});

// Lets the desktop end the pairing remotely (a "Disconnect phone" button
// there) without needing to touch the phone at all — checked at a slower
// cadence than the WebRTC signaling poll since it only needs to catch a
// one-time flip, not track something continuously changing.
let lastPairCheck=0;
async function checkStillPaired(){
  if(!id || Date.now()-lastPairCheck<2000) return;
  lastPairCheck=Date.now();
  try{
    const r=await fetch('/status?id='+encodeURIComponent(id));
    if(!r.ok) return;
    const j=await r.json();
    if(j.phone_paired===false && pc) disconnectCamera();
  }catch(_){}
}
function disconnectCamera(){
  if(pc){ try{ pc.close(); }catch(_){} pc=null; }
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
  setStatus('Disconnected from desktop', 'warn');
  showErr('Tap Reconnect below when you\'re ready to pair again.');
  $('reconnectBtn').style.display='';
}

async function pollLoop(){
  while(true){
    await pollWebRTC();
    await checkStillPaired();
    await new Promise(r=>setTimeout(r, 300));
  }
}
</script>
"""


# ---- Party mode: TV screen ---------------------------------------------------
# Open this on the host machine as http://localhost:8770/party (must be
# "localhost", same YouTube-embed-origin rule as personal mode).
PARTY_HTML = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Karaoke Party</title>
<link rel="stylesheet" href="/theme.css">
<script>
// applied before first paint to avoid a flash of the wrong theme — same
// localStorage key as the main studio page, so a preference set there
// carries over here too instead of party mode only ever following the
// OS/browser's own light/dark setting with no override at all.
try{ const t=localStorage.getItem('kstudio.theme'); if(t) document.documentElement.dataset.theme=t; }catch(e){}
</script>
<style>
  #themeToggle{position:fixed;top:16px;right:16px;z-index:50}
  body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);
    font:16px/1.4 -apple-system,system-ui,sans-serif;padding:24px;}
  h1{font-size:24px;margin:0 0 4px;color:var(--amber);text-shadow:var(--glow-amber);
    display:flex;align-items:center;gap:12px}
  .sub{color:var(--muted);margin:0 0 20px}

  /* pre-start "attract mode": this screen gets projected on a TV before
     anyone's joined, so a small left-aligned block over a mostly-empty page
     reads as unfinished at room scale. Center it full-viewport instead, with
     the back-link/theme-toggle pulled out to fixed corners so they don't
     constrain the centering, and size text for across-the-room legibility. */
  .backLink{position:fixed;top:16px;left:16px;color:var(--muted);font-size:13px;
    text-decoration:none;z-index:50}
  #start{min-height:calc(100vh - 48px);display:flex;flex-direction:column;
    align-items:center;justify-content:center;text-align:center;gap:22px}
  #start h1{font-size:48px;text-shadow:var(--glow-amber)}
  #start .sub{font-size:18px;margin:0;max-width:520px}
  #start .eq-bars{font-size:44px}

  /* room code + QR: the single biggest "moment" on this screen — give it a
     proper banner treatment instead of a plain inline row, and size it for
     TV viewing distance rather than desktop reading distance. */
  .codeRow{display:flex;align-items:center;justify-content:center;gap:32px;
    padding:22px 30px;border-radius:20px;margin-bottom:20px;
    background:var(--glass-fill);backdrop-filter:blur(22px) saturate(160%);-webkit-backdrop-filter:blur(22px) saturate(160%);
    border:1px solid var(--glass-border);box-shadow:0 10px 40px rgba(0,0,0,.5)}
  .code.marquee-sign span{width:64px;height:88px;font-size:64px}
  .join{color:var(--muted);font-size:15px}
  .join b{color:var(--ink)}
  .grid{display:grid;grid-template-columns:2fr 1fr;gap:20px}
  #stage{aspect-ratio:16/9;background:#000;border-radius:10px;overflow:hidden;position:relative}
  #stage iframe,#ytplayer{width:100%;height:100%}
  /* idle "nobody queued yet" visual: a plain black box reads as broken on a
     TV screen — give it the same equalizer-bar attract motion as the
     pre-start screen so the empty state still looks alive. */
  #stage .idleStage{position:absolute;inset:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;gap:16px;color:var(--dim)}
  #stage .idleStage .eq-bars{font-size:40px}
  .nowSingers{font-size:28px;font-weight:700;margin:14px 0 2px}
  .nowTitle{color:var(--muted)}
  .upnext{list-style:none;margin:0;padding:0}
  .upnext li{padding:10px 34px 10px 22px;border-bottom:1px solid var(--edge);position:relative;
    cursor:grab;transition:background .1s}
  .upnext li:hover{background:rgba(255,255,255,.03)}
  .upnext li.dragging{opacity:.35}
  .upnext li::before{content:"⠿";position:absolute;left:0;top:10px;color:var(--dim);font-size:16px}
  .upnext .who{font-weight:600}
  .upnext .what{color:var(--muted);font-size:14px}
  .rowBtns{position:absolute;right:0;top:8px;display:flex;gap:6px}
  .empty{color:var(--dim);padding:20px 0}
  #qrImg{width:140px;height:140px;background:#fff;border-radius:10px;padding:6px;
    border:2px solid var(--pink2);box-shadow:var(--glow-pink)}
  .challenges{margin-top:20px}
  .challenges h3{display:flex;align-items:center;gap:8px;margin-top:0}
  .chal{display:flex;align-items:center;gap:10px;padding:10px;border:1px solid var(--pink2);
    border-radius:10px;margin-bottom:8px;background:rgba(255,61,129,.07);box-shadow:var(--glow-pink)}
  .chal .txt{flex:1;font-size:14px}
  .chal .txt b{color:var(--pink)}
  .guestList{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  .guestChip{background:var(--panel);border:1px solid var(--edge);border-radius:999px;
    padding:5px 12px;font-size:13px;color:var(--ink)}

/* synced lyrics under the stage */
.tvLyrics{margin-top:10px;text-align:center;padding:10px 14px;border-radius:12px;
  background:var(--well);border:1px solid var(--edge)}
.tvLyrNow{font-family:var(--display);font-weight:700;font-size:26px;line-height:1.3;color:var(--amber2)}
.tvLyrNext{font-size:14px;color:var(--dim);margin-top:4px}
/* full-stage announcement / celebration overlay */
.partyOverlay{position:absolute;inset:0;z-index:30;display:flex;align-items:center;justify-content:center;
  background:rgba(6,8,7,.86);backdrop-filter:blur(6px);border-radius:inherit}
.poInner{text-align:center;padding:30px;max-width:80%;animation:poIn .55s cubic-bezier(.2,1.8,.35,1)}
@keyframes poIn{0%{transform:scale(.55);opacity:0}100%{transform:scale(1);opacity:1}}
.poKicker{font-family:var(--mono);font-size:14px;letter-spacing:.22em;text-transform:uppercase;color:var(--amber2);margin-bottom:10px}
.poBig{font-family:var(--display);font-weight:800;font-size:44px;line-height:1.15;color:var(--ink)}
.poSub{margin-top:12px;font-size:18px;color:var(--muted)}
.poRank{font-family:var(--display);font-weight:800;font-size:96px;line-height:1;color:var(--amber2);
  text-shadow:0 0 34px rgba(30,215,96,.5);margin:6px 0}
.poPts{font-family:var(--mono);font-size:26px;font-weight:700;color:var(--ink)}
/* live score HUD above the stage — the singer's phone streams this */
.grid .card:first-child{position:relative}
.liveHud{display:flex;align-items:center;gap:14px;margin-bottom:10px;padding:9px 16px;
  border-radius:14px;background:var(--well);border:1px solid var(--edge);
  font-family:var(--mono);transition:box-shadow .2s,border-color .2s}
.liveHud.hot{border-color:var(--amber2);box-shadow:0 0 0 1px rgba(30,215,96,.5),0 4px 28px rgba(30,215,96,.35)}
.lhName{font-weight:700;color:var(--ink);font-size:15px}
.lhForm{font-size:22px;font-weight:700;color:var(--amber2)}
.lhMult{font-weight:800;font-size:16px;padding:3px 10px;border-radius:10px;border:1px solid var(--edge);color:var(--dim)}
.lhMult.m2{color:var(--ink)}
.lhMult.m3{color:var(--amber2);border-color:var(--amber2)}
.lhMult.m4{color:#000;background:var(--amber2);border-color:var(--amber2)}
.lhMult.pop{animation:lhPop .45s cubic-bezier(.2,2.2,.4,1)}
@keyframes lhPop{0%{transform:scale(1)}40%{transform:scale(1.6)}100%{transform:scale(1)}}
.lhPts{margin-left:auto;font-size:22px;font-weight:700;color:var(--ink)}
/* scoreboard */
.scoreboard{list-style:none;margin:0 0 14px;padding:0;display:flex;flex-direction:column;gap:6px}
.scoreboard li{display:flex;align-items:center;gap:10px;background:var(--well);border:1px solid var(--edge);
  border-radius:10px;padding:8px 12px;font-family:var(--mono)}
.scoreboard li.singingNow{border-color:var(--amber2);box-shadow:0 0 0 1px rgba(30,215,96,.4)}
.scoreboard li.empty{color:var(--dim);border-style:dashed;font-size:12px}
.sbPos{width:26px;text-align:center;font-size:15px}
.sbName{font-weight:700;color:var(--ink);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sbBest{font-size:10px;color:var(--dim)}
.sbTotal{font-size:16px;font-weight:800;color:var(--amber2)}
</style>

<svg style="display:none" aria-hidden="true">
  <symbol id="i-mic" viewBox="0 0 24 24"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></symbol>
  <symbol id="i-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></symbol>
  <symbol id="i-moon" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></symbol>
  <symbol id="i-phone" viewBox="0 0 24 24"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></symbol>
  <symbol id="i-users" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></symbol>
  <symbol id="i-zap" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></symbol>
</svg>

<button class="btn ghost small" id="themeToggle" type="button" title="Toggle light/dark theme"><svg class="icon"><use href="#i-moon"/></svg></button>

<a href="/" class="backLink">← Back to Studio</a>

<div id="start">
  <h1><svg class="icon"><use href="#i-mic"/></svg> Karaoke Party <span class="marquee"><i class="bulb"></i><i class="bulb"></i><i class="bulb"></i><i class="bulb"></i></span></h1>
  <p class="sub">Start a room, then have guests join from their phones on the same WiFi.</p>
  <span class="eq-bars"><i></i><i></i><i></i><i></i></span>
  <button class="btn amber" id="startBtn">Start Party</button>
</div>

<div id="live" class="hidden">
  <div class="codeRow">
    <div>
      <div class="code marquee-sign" id="roomCode"></div>
      <div class="join">Guests: join at <b id="joinUrl"></b></div>
      <div class="join"><svg class="icon"><use href="#i-phone"/></svg> Run this from your phone: <b id="hostUrl"></b></div>
    </div>
    <img id="qrImg" class="hidden" alt="Scan to join">
  </div>
  <div class="grid">
    <div class="card">
      <div class="liveHud hidden" id="liveHud">
        <span class="lhName" id="lhName"></span>
        <span class="lhForm" id="lhForm">--%</span>
        <span class="lhMult m1" id="lhMult">×1</span>
        <span class="lhPts" id="lhPts">0</span>
      </div>
      <div class="partyOverlay hidden" id="partyOverlay"><div class="poInner" id="poInner"></div></div>
      <div id="stage"><div class="idleStage"><span class="eq-bars"><i></i><i></i><i></i><i></i></span>Nobody queued yet — waiting for guests to join and add songs.</div></div>
      <div class="tvLyrics hidden" id="tvLyrics">
        <div class="tvLyrNow" id="tvLyrNow">&nbsp;</div>
        <div class="tvLyrNext" id="tvLyrNext">&nbsp;</div>
      </div>
      <div class="nowSingers" id="singers"></div>
      <div class="nowTitle" id="songTitle"></div>
      <div style="margin-top:14px;display:flex;gap:10px;align-items:center">
        <button class="btn pink" id="nextBtn">Next ▶</button>
        <button class="btn ghost" id="scoringToggle" title="Turn singing scores on/off for this party">🎯 Scoring: ON</button>
      </div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Up next <span style="font-size:11px;color:var(--dim);font-weight:400">— drag to reorder</span></h3>
      <ul class="upnext" id="upnext"><li class="empty">Queue is empty.</li></ul>
      <h3>🏆 Scoreboard</h3>
      <ol class="scoreboard" id="scoreboard"><li class="empty">No scores yet — sing with your phone in hand!</li></ol>
      <h3><svg class="icon"><use href="#i-users"/></svg> Who's here <span id="guestCount" style="color:var(--dim);font-weight:400"></span></h3>
      <div class="guestList" id="guestList"><span class="empty">Waiting for guests to join…</span></div>
    </div>
  </div>
  <div class="card challenges hidden" id="challengesCard">
    <h3><svg class="icon"><use href="#i-zap"/></svg> Challenges <span class="marquee"><i class="bulb"></i><i class="bulb"></i></span></h3>
    <div id="challengeList"></div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let state={code:null,queue:[],now_playing:null};
let currentEntryId=null, ytPlayer=null, ytReady=false, backingAudio=null, syncTimer=null;

// ---- light/dark theme toggle (same pattern/localStorage key as the main
// studio page) -----------------------------------------------------------
function currentTheme(){
  return document.documentElement.dataset.theme ||
    (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
}
function applyThemeIcon(){ $('themeToggle').innerHTML = '<svg class="icon"><use href="#i-'+(currentTheme()==='light'?'sun':'moon')+'"/></svg>'; }
applyThemeIcon();
$('themeToggle').addEventListener('click',()=>{
  const next = currentTheme()==='light' ? 'dark' : 'light';
  document.documentElement.dataset.theme=next;
  try{ localStorage.setItem('kstudio.theme', next); }catch(e){}
  applyThemeIcon();
});

$('startBtn').addEventListener('click',async()=>{
  const r=await fetch('/room/start',{method:'POST'}); const j=await r.json();
  $('start').classList.add('hidden'); $('live').classList.remove('hidden');
  $('roomCode').innerHTML=[...j.code].map((c,i)=>`<span style="--i:${i}">${esc(c)}</span>`).join('');
  $('joinUrl').textContent=j.join_url||'(no LAN IP found — check WiFi)';
  $('hostUrl').textContent=j.host_url||'(no LAN IP found — check WiFi)';
  if(j.join_url){
    const qr=$('qrImg');
    qr.onload=()=>qr.classList.remove('hidden');
    qr.onerror=()=>qr.classList.add('hidden');  // qrencode not installed — text URL above still works
    qr.src='/qr.png?data='+encodeURIComponent(j.join_url);
  }
  subscribe();
});

$('scoringToggle').addEventListener('click',()=>{
  const currentlyOn=state.scoring_enabled!==false;
  fetch('/party/scoring',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({enabled:!currentlyOn})});
});
$('nextBtn').addEventListener('click',()=>{
  // mount synchronously inside the click so audio.play() rides this user gesture
  const next=state.queue.find(e=>e.status==='queued');
  if(next) mountEntry(next);
  fetch('/queue/next',{method:'POST'});
});

function subscribe(){
  const es=new EventSource('/room/events');
  es.onmessage=(ev)=>{ state=JSON.parse(ev.data); render(); };
}

function render(){
  if(dragging) return;  // a live drag rebuilds its own DOM order — don't fight it with an SSE-triggered re-render
  const upcoming=state.queue.filter(e=>e.status==='queued');
  $('upnext').innerHTML = upcoming.length
    ? upcoming.map((e,i)=>{
        const tag=e.from_singer?`<div class="what">⚡ challenged by ${esc(e.from_singer)}</div>`:'';
        return `<li draggable="true" data-entry-id="${e.entry_id}">
        <div class="who">${esc(e.singers.join(' & '))}</div>
        <div class="what">${esc(e.title)}</div>
        ${tag}
        <div class="rowBtns">
          <button class="btn ghost small" data-move="up" data-id="${e.entry_id}" ${i===0?'disabled':''}>▲</button>
          <button class="btn ghost small" data-move="down" data-id="${e.entry_id}" ${i===upcoming.length-1?'disabled':''}>▼</button>
          <button class="btn ghost small" data-remove="${e.entry_id}">✕</button>
        </div>
      </li>`; }).join('')
    : '<li class="empty">Queue is empty.</li>';
  const now=state.queue.find(e=>e.entry_id===state.now_playing);
  if(now){ $('singers').textContent=now.singers.join(' & '); $('songTitle').textContent=now.title; }
  else { $('singers').textContent=''; $('songTitle').textContent=''; }

  const pending=(state.challenges||[]).filter(c=>c.status==='pending');
  $('challengesCard').classList.toggle('hidden', pending.length===0);
  $('challengeList').innerHTML=pending.map(c=>`
    <div class="chal">
      <div class="txt"><b>${esc(c.from_singer)}</b> challenges <b>${esc(c.to_singer)}</b> to sing "${esc(c.title)}"</div>
      <button class="btn amber small" data-accept="${c.challenge_id}">Accept</button>
      <button class="btn ghost small" data-decline="${c.challenge_id}">Decline</button>
    </div>`).join('');

  const guests=state.guests||[];
  $('guestCount').textContent=guests.length?'('+guests.length+')':'';
  $('guestList').innerHTML = guests.length
    ? guests.map(g=>'<span class="guestChip">'+esc(g.name)+'</span>').join('')
    : '<span class="empty">Waiting for guests to join…</span>';

  // scoring toggle + everything it gates
  const scoringOn = state.scoring_enabled!==false;
  const tg=$('scoringToggle');
  if(tg){ tg.textContent='🎯 Scoring: '+(scoringOn?'ON':'OFF'); tg.classList.toggle('ghost', scoringOn); }
  // live score HUD: whoever is singing right now, straight off their phone
  const ls=scoringOn ? state.live_score : null;
  $('liveHud').classList.toggle('hidden', !ls);
  if(ls){
    $('lhName').textContent=ls.name;
    $('lhForm').textContent=(ls.form!=null?ls.form:'--')+'%';
    const lm=$('lhMult');
    const prev=+((lm.dataset.m)||1);
    lm.textContent='×'+ls.mult; lm.className='lhMult m'+ls.mult; lm.dataset.m=ls.mult;
    if(ls.mult>prev){ lm.classList.remove('pop'); void lm.offsetWidth; lm.classList.add('pop'); }
    $('lhPts').textContent=(ls.points||0).toLocaleString();
    $('liveHud').classList.toggle('hot', ls.mult>=3);
  }

  // scoreboard: party standings by total points, medals on the podium.
  // Hidden entirely while scoring is off — a frozen leaderboard reads as
  // broken, and some parties just don't want the competition.
  document.getElementById('scoreboard').style.display=scoringOn?'':'none';
  document.querySelectorAll('h3').forEach(h=>{ if(h.textContent.includes('Scoreboard')) h.style.display=scoringOn?'':'none'; });
  const scores=Object.entries(state.scores||{}).sort((a,b)=>b[1].total-a[1].total);
  const medals=['🥇','🥈','🥉'];
  $('scoreboard').innerHTML = scores.length
    ? scores.map(([name,rec],i)=>`
       <li class="${ls&&ls.singers&&ls.singers.includes(name)?'singingNow':''}">
         <span class="sbPos">${medals[i]||(i+1)}</span>
         <span class="sbName">${esc(name)}</span>
         <span class="sbBest">best ${rec.best.toLocaleString()} (${esc(rec.best_rank||'')})</span>
         <span class="sbTotal">${rec.total.toLocaleString()}</span>
       </li>`).join('')
    : '<li class="empty">No scores yet — sing with your phone in hand!</li>';
  mcReact();
}
function esc(s){ return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

// ---- synced lyrics on the TV --------------------------------------------
const tvLyrCache={};   // jid -> lines|null
let tvLyrJid=null, tvLyrIdx=-1;
setInterval(()=>{
  const now=state.queue.find(e=>e.entry_id===state.now_playing);
  const jid=now&&now.jid;
  if(!jid){ $('tvLyrics').classList.add('hidden'); tvLyrJid=null; return; }
  if(jid!==tvLyrJid){
    tvLyrJid=jid; tvLyrIdx=-1;
    if(!(jid in tvLyrCache)){
      tvLyrCache[jid]='loading';
      fetch('/lyrics?id='+jid).then(r=>r.json())
        .then(d=>{ tvLyrCache[jid]=(d.lyrics&&d.lyrics.length)?d.lyrics:null; })
        .catch(()=>{ tvLyrCache[jid]=null; });
    }
  }
  const L=tvLyrCache[jid];
  if(!L || L==='loading' || !backingAudio){ $('tvLyrics').classList.add('hidden'); return; }
  $('tvLyrics').classList.remove('hidden');
  const t=backingAudio.currentTime||0;
  let i=-1;
  while(i+1<L.length && L[i+1].t<=t) i++;
  if(i!==tvLyrIdx){
    tvLyrIdx=i;
    $('tvLyrNow').textContent = i>=0 ? L[i].text : '♪';
    $('tvLyrNext').textContent = (i+1<L.length) ? L[i+1].text : '';
  }
}, 300);

// ---- party MC: announcements between songs, hype after each one -----------
// The TV is the room's MC: when a song ends it celebrates the singer with a
// rank-appropriate comment, and when the next one starts it announces them.
// Pure client-side reaction to state transitions — no new server state.
const HYPE={
  S:["ABSOLUTE STAR POWER! 🌟","THE ROOF IS GONE! 🔥","SOMEBODY CALL A RECORD LABEL!","FLAWLESS. FRAME IT. 🖼️"],
  A:["The crowd goes WILD! 🎉","That was HOT! 🔥","Chills. Actual chills.","Encore! Encore!"],
  B:["Solid set of pipes! 👏","That's how it's done!","The party approves! 🍻"],
  C:["Heart of a champion! 💪","Sung with FEELING — love it!","The mic survived. Barely. 😄"],
  D:["Bravest performance of the night! 🫡","10/10 for courage!","Legends are built on nights like this!"]};
const INTROS=["Make some noise for","Put your hands together for","On the mic next…","Clear the floor for","The stage belongs to"];
let poTimer=null, prevNowId=null, prevScoreSig='';
function showOverlay(html, ms){
  clearTimeout(poTimer);
  $('poInner').innerHTML=html;
  const po=$('partyOverlay');
  po.classList.remove('hidden');
  // retrigger the pop-in
  const inner=$('poInner'); inner.style.animation='none'; void inner.offsetWidth; inner.style.animation='';
  poTimer=setTimeout(()=>po.classList.add('hidden'), ms);
}
function pick(a){ return a[Math.floor(Math.random()*a.length)]; }
function mcReact(){
  const nowId=state.now_playing||null;
  const scoreSig=JSON.stringify(state.scores||{});
  const scoresChanged = prevScoreSig && scoreSig!==prevScoreSig;
  const songChanged = nowId!==prevNowId;
  const scoringOnMc = state.scoring_enabled!==false;
  if(songChanged && prevNowId && !scoringOnMc){
    // scoring off: no score to celebrate, but the singer still deserves a
    // send-off before the next announcement
    const prevEntry=state.queue.find(e=>e.entry_id===prevNowId);
    if(prevEntry){
      showOverlay('<div class="poKicker">give it up for</div>'+
        '<div class="poBig">'+esc(prevEntry.singers.join(' & '))+' 👏</div>'+
        '<div class="poSub">'+esc(pick(HYPE.B))+'</div>', 4500);
    }
  }
  if(scoresChanged && scoringOnMc){
    // someone just finished — find who improved and celebrate them
    let who=null, prev={};
    try{ prev=JSON.parse(prevScoreSig)||{}; }catch(e){}
    for(const [name,rec] of Object.entries(state.scores||{})){
      const p=prev[name];
      if(!p || rec.songs>p.songs){ who={name, pts:rec.total-(p?p.total:0), rank:rec.best_rank}; break; }
    }
    if(who && who.pts>0){
      const rank=(who.rank||'C');
      showOverlay(
        '<div class="poKicker">that just happened</div>'+
        '<div class="poBig">'+esc(who.name)+'</div>'+
        '<div class="poRank">'+esc(rank)+'</div>'+
        '<div class="poPts">+'+who.pts.toLocaleString()+' pts</div>'+
        '<div class="poSub">'+esc(pick(HYPE[rank]||HYPE.C))+'</div>', 6000);
    }
  }
  if(songChanged && nowId){
    const now=state.queue.find(e=>e.entry_id===nowId);
    if(now){
      const announce=()=>showOverlay(
        '<div class="poKicker">'+esc(pick(INTROS))+'</div>'+
        '<div class="poBig">'+esc(now.singers.join(' & '))+' 🎤</div>'+
        '<div class="poSub">'+esc(now.title||'')+'</div>', 5000);
      // if a celebration/send-off is on screen, let it land first
      if(scoresChanged && scoringOnMc) setTimeout(announce, 6200);
      else if(prevNowId && !scoringOnMc) setTimeout(announce, 4700);
      else announce();
    }
  }
  prevNowId=nowId; prevScoreSig=scoreSig;
}

$('upnext').addEventListener('click',(ev)=>{
  const b=ev.target.closest('button'); if(!b) return;
  if(b.dataset.move) fetch('/queue/move',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({entry_id:b.dataset.id, direction:b.dataset.move})});
  else if(b.dataset.remove) fetch('/queue/remove',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({entry_id:b.dataset.remove})});
});

$('challengeList').addEventListener('click',(ev)=>{
  const b=ev.target.closest('button'); if(!b) return;
  if(b.dataset.accept) fetch('/challenge/accept',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({challenge_id:b.dataset.accept})});
  else if(b.dataset.decline) fetch('/challenge/decline',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({challenge_id:b.dataset.decline})});
});

// ---- drag-to-reorder: classic "move the dragged node live, commit on drop" -----
let dragging=null;
$('upnext').addEventListener('dragstart',(ev)=>{
  const li=ev.target.closest('li[draggable]'); if(!li) return;
  dragging=li; li.classList.add('dragging');
  ev.dataTransfer.effectAllowed='move';
});
$('upnext').addEventListener('dragend',()=>{
  if(dragging) dragging.classList.remove('dragging');
  dragging=null;
  const order=[...$('upnext').querySelectorAll('li[data-entry-id]')].map(li=>li.dataset.entryId);
  if(order.length) fetch('/queue/reorder',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({order})});
});
$('upnext').addEventListener('dragover',(ev)=>{
  if(!dragging) return;
  ev.preventDefault();
  const after=[...$('upnext').querySelectorAll('li[data-entry-id]:not(.dragging)')].find(li=>{
    const r=li.getBoundingClientRect();
    return ev.clientY < r.top + r.height/2;
  });
  if(after) $('upnext').insertBefore(dragging, after);
  else $('upnext').appendChild(dragging);
});

function loadYTApi(cb){
  if(window.YT && window.YT.Player){ cb(); return; }
  window.onYouTubeIframeAPIReady=cb;
  if(!document.getElementById('ytapi')){
    const s=document.createElement('script'); s.id='ytapi';
    s.src='https://www.youtube.com/iframe_api'; document.head.appendChild(s);
  }
}

async function mountEntry(entry){
  if(!entry || entry.entry_id===currentEntryId) return;
  currentEntryId=entry.entry_id;
  $('singers').textContent=entry.singers.join(' & ');
  $('songTitle').textContent=entry.title;
  const j=await (await fetch('/status?id='+entry.jid)).json();
  loadYTApi(()=>{
    $('stage').innerHTML='<div id="ytplayer"></div>';
    ytReady=false;
    ytPlayer=new YT.Player('ytplayer',{
      videoId:j.video_id, host:'https://www.youtube.com',
      playerVars:{mute:1,rel:0,modestbranding:1,playsinline:1,iv_load_policy:3,controls:0,origin:window.location.origin},
      events:{
        onReady:()=>{ try{ytPlayer.mute();}catch(_){} ytReady=true; ytPlayer.playVideo(); startAudio(entry); },
        onStateChange:onPartyVideoState,
      }
    });
  });
}

// The video is the master clock, but it isn't the only thing that can change
// its play state — a stray click on the (controls:0) player, buffering, or
// the tab losing focus can all pause/resume it without going through our own
// code. Previously only drift got corrected, never play/pause, so a paused
// video left the backing track playing on its own and fighting the sync
// loop's repeated seeks-back — that's the "glitchy" bug. Mirror state instead.
function onPartyVideoState(e){
  if(!backingAudio) return;
  const st=e.data;
  if(st===1){ // playing
    resyncAudio();
    backingAudio.play().catch(()=>{});
  } else if(st===2){ // paused
    backingAudio.pause();
  } else if(st===0){ // ended
    backingAudio.pause();
  }
  // buffering(3) / cued(5): leave backingAudio as-is, syncTimer will settle it
}

function resyncAudio(){
  if(!ytReady || !backingAudio) return;
  try{ backingAudio.currentTime=ytPlayer.getCurrentTime(); }catch(_){}
}

// Background tabs get their timers throttled hard, so drift (or a missed
// pause/play event) can pile up while this tab isn't visible — force a hard
// resync the moment the host switches back to it instead of waiting for the
// next 500ms tick to slowly correct a potentially multi-second gap.
document.addEventListener('visibilitychange',()=>{
  if(document.visibilityState!=='visible' || !ytReady || !backingAudio) return;
  resyncAudio();
  try{
    if(ytPlayer.getPlayerState()===1) backingAudio.play().catch(()=>{});
    else backingAudio.pause();
  }catch(_){}
});

function startAudio(entry){
  if(backingAudio){ try{backingAudio.pause();}catch(_){} }
  if(syncTimer) clearInterval(syncTimer);
  backingAudio=new Audio('/backing?id='+entry.jid+'&key='+entry.semitones);
  backingAudio.play().catch(()=>{});
  syncTimer=setInterval(()=>{
    if(!ytReady||!backingAudio) return;
    if(ytPlayer.getPlayerState()!==1) return;  // only chase drift while actually playing
    const vt=ytPlayer.getCurrentTime();
    if(Math.abs((backingAudio.currentTime||0)-vt)>0.15){ try{backingAudio.currentTime=vt;}catch(_){} }
  },500);
}
</script>
"""


# ---- Party mode: guest join page ---------------------------------------------
# Guests open this over the LAN (the host's LAN IP, shown as the join URL) —
# no YouTube iframe here at all, so the localhost-only embed restriction
# doesn't apply to this page.
JOIN_HTML = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Join the Party</title>
<link rel="stylesheet" href="/theme.css">
<script>
// applied before first paint to avoid a flash of the wrong theme — same
// localStorage key as the main studio page.
try{ const t=localStorage.getItem('kstudio.theme'); if(t) document.documentElement.dataset.theme=t; }catch(e){}
</script>
<style>
  *{-webkit-tap-highlight-color:transparent}
  /* width:auto overrides this page's blanket .btn{width:100%} (meant for
     full-width mobile action buttons) — without it, a fixed-position element
     stretches to its containing block's width, which becomes very visible
     once body itself establishes that containing block (see the
     backdrop-filter media query below). */
  #themeToggle{position:fixed;top:calc(14px + env(safe-area-inset-top));right:16px;z-index:50;width:auto}
  body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);
    font:16px/1.4 -apple-system,system-ui,sans-serif;
    padding:16px 16px calc(16px + env(safe-area-inset-bottom));max-width:480px;margin:0 auto}
  /* this page is built for a phone screen and stays that width even on a
     wide desktop browser (real usage is always a phone that scanned the QR)
     — but when it IS opened wide, frame it as an intentional phone-shaped
     card in the ambient backdrop instead of a website that forgot to use
     its space. No effect on an actual phone viewport. */
  @media (min-width:620px){
    body{margin-top:36px;padding:28px 28px calc(28px + env(safe-area-inset-bottom));
      border-radius:24px;background:var(--glass-fill);backdrop-filter:blur(24px) saturate(160%);
      -webkit-backdrop-filter:blur(24px) saturate(160%);border:1px solid var(--glass-border);
      box-shadow:0 24px 70px rgba(0,0,0,.5)}
    /* nudge down to stay aligned with the card's top edge instead of the
       bare viewport corner, since #themeToggle is fixed (viewport-relative)
       and body just gained a 36px top margin above. */
    #themeToggle{top:50px}
  }
  h1{font-size:20px;color:var(--amber);text-shadow:var(--glow-amber);margin:6px 0 14px;
    display:flex;align-items:center;gap:10px}
  label{display:block;font-size:13px;color:var(--muted);margin:14px 0 6px}
  .btn{width:100%;margin-top:10px;min-height:46px}
  .results{margin-top:10px}
  .result{display:flex;gap:10px;padding:10px;border:1px solid var(--edge);border-radius:12px;
    margin-bottom:8px;cursor:pointer;background:var(--panel);transition:border-color .12s,transform .12s}
  .result:hover{border-color:var(--pink2);transform:translateY(-1px)}
  .result img{width:72px;height:54px;object-fit:cover;border-radius:8px;flex:none}
  .result .t{font-size:14px}
  .result .c{font-size:12px;color:var(--muted)}
  .card{margin-top:16px}
  .slider{display:flex;align-items:center;gap:10px}
  input[type=range]{flex:1;min-height:32px}
  .msg{color:var(--ok);margin-top:10px}
  .err{color:var(--rec);margin-top:10px}
  .toggleRow{display:flex;align-items:center;gap:8px;margin-top:12px;font-size:13px;
    color:var(--muted);cursor:pointer;user-select:none;min-height:24px}
  .qlist li{list-style:none;padding:8px 0;border-bottom:1px solid var(--edge)}
  .qlist li:last-child{border-bottom:none}
  .qlist .who{font-weight:600;font-size:14px}
  .qlist .what{color:var(--muted);font-size:12.5px}
  .qlist .now .who{color:var(--amber);text-shadow:var(--glow-amber)}
  .chal{padding:10px;border:1px solid var(--pink2);border-radius:10px;margin-top:8px;
    background:rgba(255,61,129,.07);box-shadow:var(--glow-pink)}
  .chal .txt{font-size:13.5px;margin-bottom:8px}
  .chal .txt b{color:var(--pink)}
  .chal .btnrow{display:flex;gap:8px}
  .chal .btnrow .btn{margin-top:0}
  .chal .btnrow .btn:disabled{opacity:.35}

  /* join step */
  .whoRow{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--muted);margin-bottom:12px}
  .whoRow b{color:var(--ink)}

  /* tabs: keep the two things guests actually need one tap away, instead of
     one long scroll where the queue view ends up buried under search results */
  .tabbar{display:flex;gap:8px;margin:14px 0;position:sticky;top:0;background:var(--bg);
    padding:8px 0;z-index:5}
  .tabbtn{flex:1;padding:12px 10px;border-radius:12px;border:1px solid var(--edge);
    background:var(--metal);color:var(--muted);font-weight:650;font-size:13px;cursor:pointer;
    min-height:46px}
  .tabbtn.active{background:linear-gradient(180deg,var(--pink),var(--pink2));color:#fff;
    border-color:var(--pink2);box-shadow:var(--glow-pink)}
  .tabpane.hidden{display:none}

  .guestList{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
  .guestChip{background:var(--panel);border:1px solid var(--edge);border-radius:999px;
    padding:5px 10px;font-size:12px;color:var(--muted)}
  .guestChip.me{color:var(--pink);border-color:var(--pink2)}
/* your-turn scoring card + HUD */
.turnCard{border-color:var(--pink2);box-shadow:var(--glow-pink);text-align:center}
.turnFlag{font-family:var(--display);font-size:26px;font-weight:800;color:var(--amber2);
  animation:turnPulse 1.4s ease-in-out infinite}
@keyframes turnPulse{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}
.turnSong{color:var(--muted);font-size:13px;margin:6px 0 12px}
.turnHud{display:flex;align-items:center;justify-content:center;gap:18px;margin-top:10px}
.turnHud small{display:block;font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}
.hudForm span,.hudPts span{font-family:var(--mono);font-size:34px;font-weight:700;color:var(--ink)}
.hudMult{font-family:var(--mono);font-weight:800;font-size:22px;padding:6px 12px;border-radius:12px;
  border:1px solid var(--edge);color:var(--dim)}
.hudMult.m2{color:var(--ink)}
.hudMult.m3{color:var(--amber2);border-color:var(--amber2)}
.hudMult.m4{color:#000;background:var(--amber2);border-color:var(--amber2)}
.hudMult.pop{animation:hudPop .45s cubic-bezier(.2,2.2,.4,1)}
@keyframes hudPop{0%{transform:scale(1)}40%{transform:scale(1.5)}100%{transform:scale(1)}}
.turnNote{margin-top:10px;font-family:var(--mono);font-size:11px;color:var(--muted)}
/* full-screen turn pop-up */
.turnPopup{position:fixed;inset:0;z-index:120;display:flex;align-items:center;justify-content:center;
  background:rgba(4,6,5,.9);backdrop-filter:blur(8px);padding:24px}
.tpInner{text-align:center;animation:tpIn .5s cubic-bezier(.2,1.9,.35,1)}
@keyframes tpIn{0%{transform:scale(.5);opacity:0}100%{transform:scale(1);opacity:1}}
.tpBig{font-family:var(--display);font-weight:800;font-size:40px;color:var(--amber2);line-height:1.15}
.tpSub{margin-top:10px;font-size:15px;color:var(--muted)}
.tpDismiss{margin-top:20px}
</style>

<svg style="display:none" aria-hidden="true">
  <symbol id="i-mic" viewBox="0 0 24 24"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></symbol>
  <symbol id="i-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></symbol>
  <symbol id="i-moon" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></symbol>
  <symbol id="i-sparkle" viewBox="0 0 24 24"><path d="M12 3l1.8 4.9L19 9.5l-4.9 1.8L12 16l-1.8-4.7L5 9.5l5.2-1.6L12 3z"/><path d="M19 14l.9 2.5L22 17.5l-2.1.9L19 21l-.9-2.6L16 17.5l2.1-1L19 14z"/></symbol>
  <symbol id="i-search" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></symbol>
  <symbol id="i-music" viewBox="0 0 24 24"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></symbol>
  <symbol id="i-target" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></symbol>
  <symbol id="i-users" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></symbol>
</svg>

<button class="btn ghost small" id="themeToggle" type="button" title="Toggle light/dark theme"><svg class="icon"><use href="#i-moon"/></svg></button>
<h1><svg class="icon"><use href="#i-mic"/></svg> Join the party <span class="marquee"><i class="bulb"></i><i class="bulb"></i><i class="bulb"></i></span></h1>

<!-- STEP 0: join with a verified name — this is what lets challenge accept/
     decline later confirm "is this really the person being challenged," so
     nobody can be impersonated into being put on the spot to sing. -->
<div id="joinStep">
  <label>Room code</label>
  <input type="text" id="room" style="text-transform:uppercase" maxlength="4">
  <label>Your name</label>
  <input type="text" id="myName" placeholder="Alice" maxlength="40">
  <button class="btn pink" id="joinBtn" type="button">Join the party →</button>
  <div id="joinErr" class="err"></div>
</div>

<div id="mainApp" class="hidden">
  <div class="whoRow"><svg class="icon"><use href="#i-sparkle"/></svg> Joined as <b id="whoLabel"></b> · room <b id="whoRoom"></b>
    <button class="btn ghost small" id="leaveBtn" type="button" style="width:auto;margin-left:auto">Not you?</button></div>

  <!-- your-turn scoring: this phone becomes the scoring mic while its owner
       sings — held close, the voice dominates the room's speakers, which is
       what makes per-singer party scoring possible at all. -->
  <div class="turnPopup hidden" id="turnPopup">
    <div class="tpInner" id="tpInner"></div>
  </div>
  <div class="card turnCard hidden" id="turnCard">
    <div class="turnFlag">🎤 YOU'RE UP!</div>
    <div class="turnSong" id="turnSong"></div>
    <button class="btn pink" id="turnStartBtn" type="button">Start scoring — hold phone near your mouth</button>
    <div class="turnHud hidden" id="turnHud">
      <div class="hudForm"><span id="hudForm">--%</span><small>form</small></div>
      <div class="hudMult m1" id="hudMult">×1</div>
      <div class="hudPts"><span id="hudPts">0</span><small>pts</small></div>
    </div>
    <div class="turnNote" id="turnNote"></div>
  </div>

  <div class="tabbar">
    <button class="tabbtn active" id="tabBtnFind" type="button"><svg class="icon"><use href="#i-search"/></svg> Find a song</button>
    <button class="tabbtn" id="tabBtnQueue" type="button"><svg class="icon"><use href="#i-music"/></svg> Party queue</button>
  </div>

  <div class="tabpane" id="tabFind">
    <label>Search for a song</label>
    <input type="text" id="q" placeholder="Search YouTube...">
    <label class="toggleRow"><input type="checkbox" id="karaokeSuffix" checked> <svg class="icon"><use href="#i-mic"/></svg> Add "karaoke" to search (helps surface lyrics videos)</label>
    <button class="btn amber" id="searchBtn" type="button">Search</button>
    <div class="results" id="results"></div>

    <div class="card hidden" id="preview">
      <div id="prepStatus" style="color:var(--muted)">Preparing track…</div>
      <div id="previewControls" class="hidden">
        <div class="hint" style="color:var(--muted);font-size:12px;margin-bottom:6px">Preview it and pick your key before confirming:</div>
        <audio id="audio" controls style="width:100%"></audio>
        <div class="slider" style="margin-top:10px">
          <span>Key</span>
          <input type="range" id="key" min="-6" max="6" value="0">
          <span id="keyVal">0</span>
        </div>
        <label>Singing with anyone else? (optional, comma-separated)</label>
        <input type="text" id="duetWith" placeholder="Bob">
        <label class="toggleRow"><input type="checkbox" id="challengeMode"> <svg class="icon"><use href="#i-target"/></svg> Challenge someone else to sing this instead</label>
        <input type="text" id="challengeTarget" class="hidden" placeholder="Who are you challenging?" style="margin-top:8px">
        <button class="btn pink" id="addBtn" type="button">Confirm — add to queue ✓</button>
      </div>
    </div>
    <div id="feedback"></div>
  </div>

  <div class="tabpane hidden" id="tabQueue">
    <div class="card" id="queueView">
      <h3 style="margin-top:0"><svg class="icon"><use href="#i-music"/></svg> Party queue</h3>
      <div id="qEmpty" style="color:var(--dim);font-size:13px">Nobody's queued yet.</div>
      <ul class="qlist hidden" id="qList"></ul>
      <div id="qChallenges"></div>
    </div>
    <div class="card">
      <h3 style="margin-top:0"><svg class="icon"><use href="#i-users"/></svg> Who's here</h3>
      <div class="guestList" id="guestList"></div>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const params=new URLSearchParams(location.search);
if(params.get('room')) $('room').value=params.get('room').toUpperCase();

// ---- light/dark theme toggle (same pattern/localStorage key as the main
// studio page) -----------------------------------------------------------
function currentTheme(){
  return document.documentElement.dataset.theme ||
    (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
}
function applyThemeIcon(){ $('themeToggle').innerHTML = '<svg class="icon"><use href="#i-'+(currentTheme()==='light'?'sun':'moon')+'"/></svg>'; }
applyThemeIcon();
$('themeToggle').addEventListener('click',()=>{
  const next = currentTheme()==='light' ? 'dark' : 'light';
  document.documentElement.dataset.theme=next;
  try{ localStorage.setItem('kstudio.theme', next); }catch(e){}
  applyThemeIcon();
});

// a random per-browser token, persisted — this is what proves "this device
// is really Alice" so a challenge aimed at Alice can only be accepted or
// declined from the device that actually joined as Alice (or by the host).
function getDeviceId(){
  try{
    let id=localStorage.getItem('kstudio.device_id');
    if(!id){ id=Math.random().toString(36).slice(2)+Date.now().toString(36); localStorage.setItem('kstudio.device_id', id); }
    return id;
  }catch(e){ return 'nodevice'; }
}
const DEVICE_ID=getDeviceId();

let myName=null, currentJid=null, statusPoll=null, roomEvents=null, lastState={queue:[],challenges:[],guests:[]};

function esc(s){ return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

// ---- join step --------------------------------------------------------
$('joinBtn').addEventListener('click', doJoin);
$('myName').addEventListener('keydown', e=>{ if(e.key==='Enter') doJoin(); });
$('room').addEventListener('keydown', e=>{ if(e.key==='Enter') $('myName').focus(); });

async function doJoin(){
  const room=$('room').value.trim().toUpperCase();
  const name=$('myName').value.trim();
  $('joinErr').textContent='';
  if(room.length!==4){ $('joinErr').textContent='Enter the 4-letter room code.'; return; }
  if(!name){ $('joinErr').textContent='Enter your name.'; return; }
  $('joinBtn').disabled=true;
  let r,j;
  try{
    r=await fetch('/room/join',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name, device_id:DEVICE_ID})});
    j=await r.json();
  }catch(e){ $('joinBtn').disabled=false; $('joinErr').textContent='Could not reach the party host.'; return; }
  $('joinBtn').disabled=false;
  if(!r.ok){ $('joinErr').textContent=j.error||'Could not join.'; return; }
  myName=name;
  $('whoLabel').textContent=myName; $('whoRoom').textContent=room;
  $('joinStep').classList.add('hidden');
  $('mainApp').classList.remove('hidden');
  subscribeRoom();
}
$('leaveBtn').addEventListener('click',()=>{
  myName=null;
  $('mainApp').classList.add('hidden');
  $('joinStep').classList.remove('hidden');
  if(roomEvents){ roomEvents.close(); roomEvents=null; }
});

// ---- tabs ---------------------------------------------------------------
$('tabBtnFind').addEventListener('click',()=>switchTab('find'));
$('tabBtnQueue').addEventListener('click',()=>switchTab('queue'));
function switchTab(name){
  $('tabBtnFind').classList.toggle('active', name==='find');
  $('tabBtnQueue').classList.toggle('active', name==='queue');
  $('tabFind').classList.toggle('hidden', name!=='find');
  $('tabQueue').classList.toggle('hidden', name!=='queue');
}

// ---- live room state: queue, challenges, who's here ----------------------
function subscribeRoom(){
  if(roomEvents) roomEvents.close();
  roomEvents=new EventSource('/room/events');
  roomEvents.onmessage=(ev)=>{ lastState=JSON.parse(ev.data); renderQueueView(); renderGuests(); checkTurn(); };
}

function renderQueueView(){
  const state=lastState;
  const upcoming=state.queue.filter(e=>e.status==='queued');
  const now=state.queue.find(e=>e.entry_id===state.now_playing);
  $('qEmpty').classList.toggle('hidden', !!(now || upcoming.length));
  $('qList').classList.toggle('hidden', !(now || upcoming.length));
  const rows=[];
  if(now) rows.push(`<li class="now"><div class="who">▶ ${esc(now.singers.join(' & '))}</div><div class="what">${esc(now.title)}</div></li>`);
  upcoming.forEach(e=>rows.push(`<li><div class="who">${esc(e.singers.join(' & '))}</div><div class="what">${esc(e.title)}</div></li>`));
  $('qList').innerHTML=rows.join('');

  const pending=(state.challenges||[]).filter(c=>c.status==='pending');
  $('qChallenges').innerHTML=pending.map(c=>{
    const mine=c.to_singer===myName;
    const lockedTitle=mine?'':'Only '+esc(c.to_singer)+' can respond to this one';
    return `
    <div class="chal">
      <div class="txt"><b>${esc(c.from_singer)}</b> challenges <b>${esc(c.to_singer)}</b> to sing "${esc(c.title)}"</div>
      <div class="btnrow">
        <button class="btn amber small" data-accept="${c.challenge_id}" ${mine?'':'disabled'} title="${lockedTitle}">Accept</button>
        <button class="btn ghost small" data-decline="${c.challenge_id}" ${mine?'':'disabled'} title="${lockedTitle}">Decline</button>
      </div>
    </div>`;
  }).join('');
}
function renderGuests(){
  const guests=lastState.guests||[];
  $('guestList').innerHTML = guests.length
    ? guests.map(g=>`<span class="guestChip${g.name===myName?' me':''}">${esc(g.name)}</span>`).join('')
    : '<span class="hint">Nobody\'s joined yet.</span>';
}

$('qChallenges').addEventListener('click',(ev)=>{
  const b=ev.target.closest('button'); if(!b || b.disabled) return;
  if(b.dataset.accept) fetch('/challenge/accept',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({challenge_id:b.dataset.accept, device_id:DEVICE_ID})});
  else if(b.dataset.decline) fetch('/challenge/decline',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({challenge_id:b.dataset.decline, device_id:DEVICE_ID})});
});

// ---- search + preview-before-confirm --------------------------------------
$('searchBtn').addEventListener('click', doSearch);
$('q').addEventListener('keydown', e=>{ if(e.key==='Enter') doSearch(); });
$('challengeMode').addEventListener('change',()=>{
  const on=$('challengeMode').checked;
  $('challengeTarget').classList.toggle('hidden', !on);
  $('addBtn').textContent = on ? '⚡ Send challenge' : 'Confirm — add to queue ✓';
});

async function doSearch(){
  let q=$('q').value.trim();
  if(!q) return;
  if($('karaokeSuffix').checked && !/karaoke/i.test(q)) q+=' karaoke';
  $('results').innerHTML='Searching…';
  const r=await fetch('/search?q='+encodeURIComponent(q));
  const j=await r.json();
  const results=j.results||[];
  $('results').innerHTML = results.length ? results.map((res,i)=>
    `<div class="result" data-i="${i}"><img src="${res.thumb}"><div><div class="t">${esc(res.title)}</div><div class="c">${esc(res.channel||'')}</div></div></div>`
  ).join('') : 'No results.';
  [...document.querySelectorAll('.result')].forEach((el,i)=>{
    el.addEventListener('click',()=>pickSong(results[i]));
  });
}

async function pickSong(res){
  currentJid=null;
  $('preview').classList.remove('hidden');
  $('preview').scrollIntoView({behavior:'smooth', block:'start'});
  $('previewControls').classList.add('hidden');
  $('prepStatus').textContent='Downloading & preparing "'+res.title+'"…';
  $('feedback').innerHTML='';
  const r=await fetch('/prepare',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({url:'https://www.youtube.com/watch?v='+res.id})});
  const j=await r.json();
  currentJid=j.id;
  if(statusPoll) clearInterval(statusPoll);
  statusPoll=setInterval(async()=>{
    const s=await (await fetch('/status?id='+currentJid)).json();
    if(s.status==='ready'){
      clearInterval(statusPoll);
      $('prepStatus').textContent='Ready: '+s.title;
      $('previewControls').classList.remove('hidden');
      loadPreview(0);
    } else if(s.status==='error'){
      clearInterval(statusPoll);
      $('prepStatus').textContent='Failed: '+(s.error||'unknown error');
    } else {
      $('prepStatus').textContent='Preparing… '+Math.round(s.pct||0)+'%';
    }
  },1000);
}

function loadPreview(key){
  const a=$('audio');
  const wasPlaying=!a.paused;
  const t=a.currentTime||0;
  a.src='/backing?id='+currentJid+'&key='+key;
  a.onloadedmetadata=()=>{ try{a.currentTime=t;}catch(_){} if(wasPlaying) a.play().catch(()=>{}); };
}
$('key').addEventListener('input',()=>{ $('keyVal').textContent=$('key').value; });
$('key').addEventListener('change',()=>{ loadPreview(parseInt($('key').value,10)); });

$('addBtn').addEventListener('click', async()=>{
  const semitones=parseInt($('key').value,10);
  if(!currentJid){ $('feedback').innerHTML='<div class="err">Pick a song first.</div>'; return; }

  if($('challengeMode').checked){
    const target=$('challengeTarget').value.trim();
    if(!target){ $('feedback').innerHTML='<div class="err">Who are you challenging?</div>'; return; }
    const r=await fetch('/challenge/add',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({jid:currentJid, from_singer:myName, to_singer:target, semitones})});
    const j=await r.json();
    if(r.ok){ $('feedback').innerHTML='<div class="msg">Challenge sent to '+esc(target)+'! ⚡</div>'; }
    else { $('feedback').innerHTML='<div class="err">'+esc(j.error||'Could not send challenge.')+'</div>'; }
    return;
  }

  const extra=$('duetWith').value.split(',').map(s=>s.trim()).filter(Boolean);
  const singers=[myName, ...extra];
  const r=await fetch('/queue/add',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({jid:currentJid,singers,semitones})});
  const j=await r.json();
  if(r.ok){ $('feedback').innerHTML='<div class="msg">You\'re queued! 🎉</div>'; switchTab('queue'); }
  else { $('feedback').innerHTML='<div class="err">'+esc(j.error||'Could not queue.')+'</div>'; }
});

// ==== YOUR-TURN SCORING ======================================================
// The exact same engine as the studio's Record tab, compacted: confident
// pitch detection (level + periodicity + vocal-range gates), median-of-250ms
// scored against the SONG'S detected key, combo multipliers with a 300ms
// transition grace, points = quality x multiplier. Results stream to the TV.
const A4J=440, NOTEJ=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
let scoreCtx=null, scoreStream=null, scoreAnalyser=null, scoreRaf=0, scorePost=0;
let sc=null, activeEntry=null, turnKey=null;
function scReset(){ sc={hist:[],recent:[],frames:0,inTune:0,streak:0,off:0,mult:1,bestMult:1,points:0}; }
function jFreqToMidiCents(f){
  const midi=Math.round(12*Math.log2(f/A4J)+69);
  const cents=Math.round(1200*Math.log2(f/(A4J*Math.pow(2,(midi-69)/12))));
  return midi*100+cents;
}
function jAutoCorrelate(buf,sr){
  let rms=0; for(let i=0;i<buf.length;i++) rms+=buf[i]*buf[i];
  rms=Math.sqrt(rms/buf.length);
  if(rms<0.02) return -1;
  const n=buf.length, c=new Array(n).fill(0);
  for(let lag=0;lag<n;lag++){ for(let i=0;i<n-lag;i++) c[lag]+=buf[i]*buf[i+lag]; }
  let d=0; while(d<n-1 && c[d]>c[d+1]) d++;
  let maxv=-1,maxp=-1;
  for(let i=d;i<n;i++){ if(c[i]>maxv){maxv=c[i];maxp=i;} }
  if(maxv<0.45*c[0]) return -1;
  let T0=maxp;
  if(T0>0&&T0<n-1){ const x1=c[T0-1],x2=c[T0],x3=c[T0+1],a=(x1+x3-2*x2)/2,b=(x3-x1)/2; if(a) T0=T0-b/(2*a); }
  const f=sr/T0; return (f>=70&&f<=1200)?f:-1;
}
function scFrame(){
  const buf=new Float32Array(scoreAnalyser.fftSize);
  scoreAnalyser.getFloatTimeDomainData(buf);
  const f=jAutoCorrelate(buf,scoreCtx.sampleRate);
  if(f>0){
    sc.hist.push(jFreqToMidiCents(f)); if(sc.hist.length>15) sc.hist.shift();
    if(sc.hist.length>=9){
      const m=[...sc.hist].sort((a,b)=>a-b)[Math.floor(sc.hist.length/2)];
      let err,lo,hi;
      if(turnKey){
        const iv=turnKey.scale==='minor'?[0,2,3,5,7,8,10]:[0,2,4,5,7,9,11];
        const root=((turnKey.root+(activeEntry.semitones||0))%12+12)%12;
        const pos=((m%1200)+1200)%1200;
        err=1200;
        for(const x of iv){ const c0=((root+x)%12)*100; const d0=Math.abs(pos-c0);
          err=Math.min(err,Math.min(d0,1200-d0)); }
        lo=20; hi=60;
      } else { err=((m%100)+100)%100; err=Math.min(err,100-err); lo=15; hi=35; }
      const fs=err<=lo?1:(err>=hi?0:(hi-err)/(hi-lo));
      sc.frames++; sc.inTune+=fs;
      if(fs>=0.75){ sc.streak++; sc.off=0; } else if(++sc.off>18){ sc.streak=0; }
      const pm=sc.mult;
      sc.mult=sc.streak>=600?4:sc.streak>=300?3:sc.streak>=120?2:1;
      if(sc.mult>sc.bestMult) sc.bestMult=sc.mult;
      sc.points+=fs*sc.mult;
      sc.recent.push(fs); if(sc.recent.length>240) sc.recent.shift();
      const form=Math.round(100*sc.recent.reduce((a,b)=>a+b,0)/sc.recent.length);
      $('hudForm').textContent=form+'%';
      const hm=$('hudMult'); hm.textContent='×'+sc.mult; hm.className='hudMult m'+sc.mult;
      if(sc.mult>pm){ hm.classList.remove('pop'); void hm.offsetWidth; hm.classList.add('pop'); }
      $('hudPts').textContent=Math.round(sc.points).toLocaleString();
    }
  }
  scoreRaf=requestAnimationFrame(scFrame);
}
function scAccuracy(){ return sc.frames>0? Math.round(100*sc.inTune/sc.frames):0; }
function scRank(a){ return a>=90?'S':a>=78?'A':a>=64?'B':a>=50?'C':'D'; }
async function postScore(done){
  if(!activeEntry) return;
  try{ await fetch('/party/score',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({device_id:DEVICE_ID, entry_id:activeEntry.entry_id,
      points:Math.round(sc.points), form:sc.recent.length?Math.round(100*sc.recent.reduce((a,b)=>a+b,0)/sc.recent.length):0,
      mult:sc.mult, accuracy:scAccuracy(), rank:scRank(scAccuracy()), done:!!done})});
  }catch(e){}
}
async function startTurnScoring(){
  if(!activeEntry) return;
  $('turnStartBtn').disabled=true;
  // On plain http to a LAN address the browser doesn't expose the mic API at
  // all (not a secure context) — that's not the guest denying permission,
  // it's the URL they joined on. Hand them the HTTPS door instead of a
  // dead-end "blocked" message.
  if(!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)){
    const su=lastState.secure_join;
    $('turnNote').innerHTML = su
      ? 'Phones only allow the mic on a secure link. <a style="color:var(--amber2)" href="'+su+'/join?room='+encodeURIComponent(lastState.code||'')+'">Tap here to re-join over HTTPS</a> — accept the one-time certificate warning, then you can score.'
      : 'This link can\'t use the mic (not HTTPS). Ask the host to restart the app with `openssl` installed to enable phone scoring.';
    $('turnStartBtn').disabled=false; return;
  }
  try{
    scoreStream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false,channelCount:1}});
  }catch(e){ $('turnNote').textContent='Mic blocked — allow microphone access in your browser settings, then tap Start again.'; $('turnStartBtn').disabled=false; return; }
  scoreCtx=new (window.AudioContext||window.webkitAudioContext)({sampleRate:48000});
  const src=scoreCtx.createMediaStreamSource(scoreStream);
  scoreAnalyser=scoreCtx.createAnalyser(); scoreAnalyser.fftSize=2048;
  src.connect(scoreAnalyser);
  scReset();
  turnKey=null;
  fetch('/songkey?id='+activeEntry.jid).then(r=>r.json()).then(d=>{ turnKey=d.key||null;
    $('turnNote').textContent=turnKey? 'Scoring in '+NOTEJ[turnKey.root]+' '+turnKey.scale+' — sing!':''; }).catch(()=>{});
  $('turnStartBtn').classList.add('hidden');
  $('turnHud').classList.remove('hidden');
  scFrame();
  scorePost=setInterval(()=>postScore(false), 900);
}
function stopTurnScoring(finalize){
  cancelAnimationFrame(scoreRaf); scoreRaf=0;
  clearInterval(scorePost); scorePost=0;
  if(finalize && sc && sc.frames>60) postScore(true);
  if(scoreStream){ scoreStream.getTracks().forEach(t=>t.stop()); scoreStream=null; }
  if(scoreCtx){ try{ scoreCtx.close(); }catch(e){} scoreCtx=null; }
  $('turnHud').classList.add('hidden');
  $('turnStartBtn').classList.remove('hidden');
  $('turnStartBtn').disabled=false;
}
$('turnStartBtn').addEventListener('click', startTurnScoring);

// ---- turn pop-ups: a heads-up when you're NEXT, a hype blast when you're ON.
const PUMP_NEXT=["Warm up those pipes! 🎤","Stretch! Hydrate! Believe! 💧","Your moment is loading…","Deep breaths — you were born for this."];
const PUMP_NOW=["GO GO GO! 🔥","This is YOUR song!","Own it. Every note.","The room is yours!"];
const PUMP_SINGING=["You sound AMAZING — keep going! 🔥","Hold those notes for combo ×4!","The TV is watching — style points!","Louder! Prouder!","Chase that ×4!"];
let warnedNextId=null, poppedNowId=null, pumpTimer=null;
function showTurnPopup(big, sub, ms){
  $('tpInner').innerHTML='<div class="tpBig">'+big+'</div><div class="tpSub">'+sub+'</div>'+
    '<button class="btn ghost small tpDismiss" onclick="document.getElementById(\'turnPopup\').classList.add(\'hidden\')">Got it</button>';
  $('turnPopup').classList.remove('hidden');
  try{ navigator.vibrate && navigator.vibrate([180,70,180]); }catch(e){}
  setTimeout(()=>$('turnPopup').classList.add('hidden'), ms);
}
function pick(a){ return a[Math.floor(Math.random()*a.length)]; }
function checkTurn(){
  const now=lastState.queue.find(e=>e.entry_id===lastState.now_playing);
  const mine=now && myName && now.singers.includes(myName);
  // heads-up: my song is FIRST in the waiting queue -> "you're next"
  const upcoming=lastState.queue.filter(e=>e.status==='queued');
  const nextEntry=upcoming[0];
  if(nextEntry && myName && nextEntry.singers.includes(myName) && warnedNextId!==nextEntry.entry_id && !mine){
    warnedNextId=nextEntry.entry_id;
    showTurnPopup('You\'re up NEXT! ⏳', esc(nextEntry.title||'')+'<br>'+esc(pick(PUMP_NEXT)), 7000);
  }
  const scoringOn = lastState.scoring_enabled!==false;
  if(mine && (!activeEntry || activeEntry.entry_id!==now.entry_id)){
    if(scoreRaf) stopTurnScoring(true);   // a new song of mine started back-to-back
    activeEntry=now;
    $('turnSong').textContent=now.title||'';
    $('turnCard').classList.toggle('hidden', !scoringOn);
    if(poppedNowId!==now.entry_id){
      poppedNowId=now.entry_id;
      showTurnPopup('🎤 YOU\'RE ON!', esc(pick(PUMP_NOW))+'<br>'+(scoringOn?'Hit Start scoring and sing!':'Grab the mic and sing your heart out!'), 6000);
    }
    clearInterval(pumpTimer);
    pumpTimer=setInterval(()=>{ if(scoreRaf) $('turnNote').textContent=pick(PUMP_SINGING); }, 8000);
  } else if(mine && activeEntry && !scoringOn){
    if(scoreRaf) stopTurnScoring(true);   // host turned scoring off mid-song
    $('turnCard').classList.add('hidden');
  } else if(!mine && activeEntry){
    if(scoreRaf) stopTurnScoring(true);   // my turn ended (host hit Next)
    activeEntry=null;
    $('turnCard').classList.add('hidden');
    clearInterval(pumpTimer); pumpTimer=null;
  }
}
</script>
"""


# ---- Party mode: host remote control ------------------------------------
# A mobile-friendly page a host can open on THEIR OWN phone (over the LAN)
# to run the party without hovering at the laptop — everything /party's TV
# screen offers except the video/audio itself, which only works on the
# machine actually running as localhost. All actions here go through the
# same host-trusted (no device_id) path the TV screen uses.
HOST_HTML = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Party Host</title>
<link rel="stylesheet" href="/theme.css">
<script>
// applied before first paint to avoid a flash of the wrong theme — same
// localStorage key as the main studio page.
try{ const t=localStorage.getItem('kstudio.theme'); if(t) document.documentElement.dataset.theme=t; }catch(e){}
</script>
<style>
  body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);
    font:16px/1.4 -apple-system,system-ui,sans-serif;
    padding:16px 16px calc(16px + env(safe-area-inset-bottom));max-width:480px;margin:0 auto}
  /* same reasoning as JOIN_HTML: frame it as an intentional phone-shaped
     card when opened wide, no effect on an actual phone viewport. */
  @media (min-width:620px){
    body{margin-top:36px;padding:28px 28px calc(28px + env(safe-area-inset-bottom));
      border-radius:24px;background:var(--glass-fill);backdrop-filter:blur(24px) saturate(160%);
      -webkit-backdrop-filter:blur(24px) saturate(160%);border:1px solid var(--glass-border);
      box-shadow:0 24px 70px rgba(0,0,0,.5)}
    /* nudge down to stay aligned with the card's top edge instead of the
       bare viewport corner, since #themeToggle is fixed (viewport-relative)
       and body just gained a 36px top margin above. */
    #themeToggle{top:50px}
  }
  /* width:auto overrides this page's blanket .btn{width:100%} (meant for
     full-width mobile action buttons) — without it, a fixed-position element
     stretches to its containing block's width, which becomes very visible
     once body itself establishes that containing block (see the
     backdrop-filter media query below). */
  #themeToggle{position:fixed;top:calc(14px + env(safe-area-inset-top));right:16px;z-index:50;width:auto}
  h1{font-size:20px;color:var(--amber);text-shadow:var(--glow-amber);margin:6px 0 4px;
    display:flex;align-items:center;gap:10px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:14px}
  .btn{min-height:46px}
  .nowCard{margin-bottom:16px}
  .nowSingers{font-size:20px;font-weight:700}
  .nowTitle{color:var(--muted);font-size:13px;margin-bottom:10px}
  .upnext{list-style:none;margin:0;padding:0}
  .upnext li{padding:10px 0;border-bottom:1px solid var(--edge)}
  .upnext li:last-child{border-bottom:none}
  .upnext .who{font-weight:600}
  .upnext .what{color:var(--muted);font-size:12.5px}
  .rowBtns{display:flex;gap:6px;margin-top:6px}
  .rowBtns .btn{margin-top:0;flex:1;min-height:38px}
  .empty{color:var(--dim);padding:10px 0}
  .chal{padding:10px;border:1px solid var(--pink2);border-radius:10px;margin-bottom:8px;
    background:rgba(255,61,129,.07);box-shadow:var(--glow-pink)}
  .chal .txt{font-size:13.5px;margin-bottom:8px}
  .chal .txt b{color:var(--pink)}
  .chal .btnrow{display:flex;gap:8px}
  .chal .btnrow .btn{margin-top:0;flex:1}
  .guestList{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  .guestChip{background:var(--panel);border:1px solid var(--edge);border-radius:999px;
    padding:5px 10px;font-size:12px;color:var(--muted)}
</style>

<svg style="display:none" aria-hidden="true">
  <symbol id="i-mic" viewBox="0 0 24 24"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></symbol>
  <symbol id="i-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></symbol>
  <symbol id="i-moon" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></symbol>
  <symbol id="i-zap" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></symbol>
  <symbol id="i-users" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></symbol>
</svg>

<button class="btn ghost small" id="themeToggle" type="button" title="Toggle light/dark theme"><svg class="icon"><use href="#i-moon"/></svg></button>
<h1><svg class="icon"><use href="#i-mic"/></svg> Party Host <span class="marquee"><i class="bulb"></i><i class="bulb"></i><i class="bulb"></i></span></h1>
<div class="sub">Remote control for the TV screen — the video stays on the host laptop, but you can run the queue from here.</div>

<div id="noRoom">
  <div class="card">No party running yet. Start one from the TV screen at <code>http://localhost:8770/party</code>, then reload this page.</div>
</div>

<div id="live" class="hidden">
  <div class="card nowCard">
    <div class="metlabel" style="font-family:var(--mono);font-size:10px;color:var(--dim);letter-spacing:.1em">ROOM <span id="roomCode"></span> · NOW PLAYING</div>
    <div class="nowSingers" id="singers">—</div>
    <div class="nowTitle" id="songTitle"></div>
    <button class="btn pink wide" id="nextBtn" type="button">Next ▶</button>
  </div>

  <div class="card" id="challengesCard" style="display:none">
    <h3 style="margin-top:0"><svg class="icon"><use href="#i-zap"/></svg> Challenges</h3>
    <div id="challengeList"></div>
  </div>

  <div class="card">
    <h3 style="margin-top:0">Up next</h3>
    <ul class="upnext" id="upnext"><li class="empty">Queue is empty.</li></ul>
  </div>

  <div class="card">
    <h3 style="margin-top:0"><svg class="icon"><use href="#i-users"/></svg> Who's here <span id="guestCount" style="color:var(--dim);font-weight:400"></span></h3>
    <div class="guestList" id="guestList"><span class="empty">Waiting for guests to join…</span></div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let state={code:null,queue:[],now_playing:null,challenges:[],guests:[]};

// ---- light/dark theme toggle (same pattern/localStorage key as the main
// studio page) -----------------------------------------------------------
function currentTheme(){
  return document.documentElement.dataset.theme ||
    (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
}
function applyThemeIcon(){ $('themeToggle').innerHTML = '<svg class="icon"><use href="#i-'+(currentTheme()==='light'?'sun':'moon')+'"/></svg>'; }
applyThemeIcon();
$('themeToggle').addEventListener('click',()=>{
  const next = currentTheme()==='light' ? 'dark' : 'light';
  document.documentElement.dataset.theme=next;
  try{ localStorage.setItem('kstudio.theme', next); }catch(e){}
  applyThemeIcon();
});

function esc(s){ return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

fetch('/room/state').then(r=>r.json()).then(j=>{
  if(j.code){ state=j; showLive(); render(); }
  subscribe();
}).catch(subscribe);

function subscribe(){
  const es=new EventSource('/room/events');
  es.onmessage=(ev)=>{
    state=JSON.parse(ev.data);
    if(state.code) showLive();
    render();
  };
}
function showLive(){
  $('noRoom').classList.add('hidden');
  $('live').classList.remove('hidden');
  $('roomCode').textContent=state.code;
}

function render(){
  const upcoming=state.queue.filter(e=>e.status==='queued');
  $('upnext').innerHTML = upcoming.length
    ? upcoming.map((e,i)=>{
        const tag=e.from_singer?`<div class="what">⚡ challenged by ${esc(e.from_singer)}</div>`:'';
        return `<li>
        <div class="who">${esc(e.singers.join(' & '))}</div>
        <div class="what">${esc(e.title)}</div>
        ${tag}
        <div class="rowBtns">
          <button class="btn ghost small" data-move="up" data-id="${e.entry_id}" ${i===0?'disabled':''}>▲ Up</button>
          <button class="btn ghost small" data-move="down" data-id="${e.entry_id}" ${i===upcoming.length-1?'disabled':''}>▼ Down</button>
          <button class="btn ghost small" data-remove="${e.entry_id}">✕ Remove</button>
        </div>
      </li>`; }).join('')
    : '<li class="empty">Queue is empty.</li>';

  const now=state.queue.find(e=>e.entry_id===state.now_playing);
  $('singers').textContent = now ? now.singers.join(' & ') : '—';
  $('songTitle').textContent = now ? now.title : '';

  const pending=(state.challenges||[]).filter(c=>c.status==='pending');
  $('challengesCard').style.display=pending.length?'':'none';
  $('challengeList').innerHTML=pending.map(c=>`
    <div class="chal">
      <div class="txt"><b>${esc(c.from_singer)}</b> challenges <b>${esc(c.to_singer)}</b> to sing "${esc(c.title)}"</div>
      <div class="btnrow">
        <button class="btn amber small" data-accept="${c.challenge_id}">Accept</button>
        <button class="btn ghost small" data-decline="${c.challenge_id}">Decline</button>
      </div>
    </div>`).join('');

  const guests=state.guests||[];
  $('guestCount').textContent=guests.length?'('+guests.length+')':'';
  $('guestList').innerHTML = guests.length
    ? guests.map(g=>'<span class="guestChip">'+esc(g.name)+'</span>').join('')
    : '<span class="empty">Waiting for guests to join…</span>';
}

$('nextBtn').addEventListener('click',()=>{ fetch('/queue/next',{method:'POST'}); });

$('upnext').addEventListener('click',(ev)=>{
  const b=ev.target.closest('button'); if(!b) return;
  if(b.dataset.move) fetch('/queue/move',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({entry_id:b.dataset.id, direction:b.dataset.move})});
  else if(b.dataset.remove) fetch('/queue/remove',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({entry_id:b.dataset.remove})});
});

$('challengeList').addEventListener('click',(ev)=>{
  const b=ev.target.closest('button'); if(!b) return;
  // no device_id sent -> host-trusted path, same as the TV screen
  if(b.dataset.accept) fetch('/challenge/accept',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({challenge_id:b.dataset.accept})});
  else if(b.dataset.decline) fetch('/challenge/decline',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({challenge_id:b.dataset.decline})});
});
</script>
"""


def main():
    if not os.path.isfile(INDEX_PATH):
        raise SystemExit(
            f"Fatal: index.html not found at {INDEX_PATH}\n"
            "It must sit next to studio.py — check your install/bundle."
        )
    if not os.path.isfile(THEME_PATH):
        raise SystemExit(
            f"Fatal: theme.css not found at {THEME_PATH}\n"
            "It must sit next to studio.py — check your install/bundle."
        )
    srv = ThreadingHTTPServer((BIND_HOST, PORT), Handler)
    print(f"\n  🎤 MicDrop  →  http://{HOST}:{PORT}\n")

    global PHONE_HTTPS_READY
    lan = _lan_ip()
    phone_srv = None
    if lan:
        cert_path, key_path = _ensure_phone_cert(lan)
        if cert_path:
            try:
                phone_srv = ThreadingHTTPServer((BIND_HOST, PHONE_PORT), Handler)
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(cert_path, key_path)
                phone_srv.socket = ctx.wrap_socket(phone_srv.socket, server_side=True)
                threading.Thread(target=phone_srv.serve_forever, daemon=True).start()
                PHONE_HTTPS_READY = True
            except OSError as e:
                phone_srv = None
                print(f"  (Phone-camera HTTPS listener failed to start: {e})")
        if PHONE_HTTPS_READY:
            print(f"  Phone-camera pairing (same WiFi): https://{lan}:{PHONE_PORT}/phone")
            print("    (self-signed cert — your phone will show a security warning once;")
            print("     tap Advanced/Details → Proceed to continue)\n")
        else:
            print("  Phone-camera pairing unavailable: needs `openssl` on PATH to generate")
            print("  a certificate (phone browsers block camera access without HTTPS).\n")
    print(f"  Party mode: open http://{HOST}:{PORT}/party on this machine (TV screen)")
    if lan:
        if PHONE_HTTPS_READY:
            print(f"              guests join at https://{lan}:{PHONE_PORT}/join (once you start a room)")
            print("              (HTTPS so guest phones can use the mic for scoring)\n")
        else:
            print(f"              guests join at http://{lan}:{PORT}/join (once you start a room)")
            print("              (no HTTPS listener - guest phones will NOT be able to score)\n")
    print(f"  Sessions saved in: {SESS_DIR}")
    print("  Ctrl+C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        if phone_srv:
            phone_srv.shutdown()


if __name__ == "__main__":
    main()
