#!/usr/bin/env python3
"""
Karaoke Studio — local web app.

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
import json
import os
import queue as pyqueue
import random
import re
import socket
import ssl
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import audio_fx

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


def set_job(jid, **kw):
    with LOCK:
        JOBS.setdefault(jid, {}).update(kw)


def get_job(jid):
    with LOCK:
        return dict(JOBS.get(jid, {}))


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
        out.append({"name": d.get("display_name") or fn[:-5], "slug": fn[:-5]})
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


def list_recent_sessions(profile=None, limit=8):
    """Sessions with a downloaded backing track, newest first, for the
    'resume a recent song' panel. Survives server restarts (reads meta.json).
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
        out.append({
            "id": jid,
            "title": meta.get("title") or "Karaoke",
            "video_id": vid,
            "duration": meta.get("duration"),
            "thumb": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg" if vid else None,
            "created_at": created_at,
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
              punch_sec=None, source_take_ids=None, tid=None, pitch_score=None):
    tid = tid or uuid.uuid4().hex[:8]
    meta = _read_meta(jid)
    takes = meta.get("takes") or []
    take = {
        "id": tid, "file": file, "duration": duration, "kind": kind,
        "fx_snapshot": fx_snapshot, "punch_sec": punch_sec,
        "source_take_ids": source_take_ids, "deleted": False,
        "pitch_score": pitch_score,
    }
    takes.append(take)
    _write_meta(jid, takes=takes)
    return take


def _get_take(jid, tid):
    for t in _read_meta(jid).get("takes") or []:
        if t.get("id") == tid:
            return t
    return None


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

def do_prepare(jid, url, profile=None):
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

    dur = _probe_duration(backing)
    set_job(jid, status="ready", stage="ready", pct=100,
            video_id=vid, title=title or "Karaoke", duration=dur)
    meta_kw = {"video_id": vid, "title": title or "Karaoke", "duration": dur,
               "url": url, "created_at": time.time()}
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


def do_render(jid, vocal_path, fx, mix_opts):
    set_job(jid, status="rendering", stage="starting", error=None)
    sdir = os.path.join(SESS_DIR, jid)
    # mix against whatever key the singer settled on (0 = original)
    try:
        music_key = int(mix_opts.get("music_key", 0) or 0)
    except (TypeError, ValueError):
        music_key = 0
    music_key = max(-6, min(6, music_key))
    backing = backing_for_key(jid, music_key) or os.path.join(sdir, "backing.wav")
    if not backing or not os.path.exists(backing):
        set_job(jid, status="error", error="Backing track missing; prepare again.")
        return

    tmp = os.path.join(sdir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    out_fmt = mix_opts.get("format", "wav")

    try:
        processed_vocal = os.path.join(sdir, "vocal_fx.wav")
        audio_fx.process_vocal(
            vocal_path, processed_vocal, fx, tmp,
            progress=lambda stage: set_job(jid, stage=stage),
        )

        harmony_path = None
        harmony_take_id = mix_opts.get("harmony_take_id")
        if harmony_take_id:
            harmony_take = _get_take(jid, harmony_take_id)
            if harmony_take:
                harmony_path = os.path.join(sdir, harmony_take["file"])

        set_job(jid, stage="mixing", format=out_fmt)
        final = os.path.join(sdir, f"final.{out_fmt}")
        audio_fx.mixdown(
            processed_vocal, backing, final,
            vocal_gain_db=float(mix_opts.get("vocal_gain_db", 0)),
            music_gain_db=float(mix_opts.get("music_gain_db", -3)),
            harmony_path=harmony_path,
            harmony_gain_db=float(mix_opts.get("harmony_gain_db", -3)),
            out_format=out_fmt,
            loudnorm=bool(mix_opts.get("loudnorm", True)),
        )
    except Exception as e:
        set_job(jid, status="error", error=str(e)[:400])
        return

    title = get_job(jid).get("title", "song")
    safe = re.sub(r'[\\/:*?"<>|]+', "_", title)[:100].strip() or "song"
    set_job(jid, status="done", stage="done",
            final_file=os.path.basename(final),
            display_name=f"{safe} (karaoke).{out_fmt}")


def do_combine_video(jid):
    """Mux the phone's video with the finished audio mix into one final
    video file. No alignment step needed: the desktop starts recording the
    live WebRTC stream from the phone in the same tick it starts the vocal
    recording, so both are already on the same clock from t=0 — unlike the
    old design, which had the phone record+upload independently and relied
    on a sync chirp (audible only through a real speaker, not headphones)
    to find the offset after the fact."""
    sdir = os.path.join(SESS_DIR, jid)
    j = get_job(jid)
    phone_video = os.path.join(sdir, j["phone_video_file"])
    final_mix = os.path.join(sdir, j["final_file"])
    out_video = os.path.join(sdir, "final_video.mp4")
    try:
        audio_fx.mux_video(phone_video, final_mix, out_video)
    except Exception as e:
        set_job(jid, video_status="error", video_error=str(e)[:400])
        return
    set_job(jid, video_status="done", final_video_file="final_video.mp4")


def _run_render_locked(jid, vocal_path, fx, mix_opts, lock):
    """Thread target for /render: holds `lock` for the whole render so a
    second request for the same session can't start until this one is done
    (the lock itself was already acquired by the HTTP handler)."""
    try:
        do_render(jid, vocal_path, fx, mix_opts)
    finally:
        lock.release()


# ---- Party mode: room + queue ------------------------------------------------
# One party room per running server (a party is one physical event, so there's
# no need for multi-room isolation). The host's TV screen and every guest phone
# share this state; ROOM_LOCK guards it, and SUBSCRIBERS holds an SSE queue per
# connected browser tab so everyone sees queue changes live. Songs themselves
# still go through the existing JOBS/do_prepare pipeline — a queue entry just
# points at a jid once that job's status is "ready".

ROOM = {"code": None, "queue": [], "now_playing": None, "challenges": [], "guests": []}
ROOM_LOCK = threading.Lock()
SUBSCRIBERS = []
SUB_LOCK = threading.Lock()


def room_state():
    with ROOM_LOCK:
        return {"code": ROOM["code"], "queue": list(ROOM["queue"]),
                "now_playing": ROOM["now_playing"], "challenges": list(ROOM["challenges"]),
                "guests": list(ROOM["guests"])}


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


def queue_advance():
    """Mark the current now-playing entry done (if any) and promote the next
    queued entry."""
    with ROOM_LOCK:
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

        if p.path == "/sessions":
            profile = (parse_qs(p.query).get("profile") or [""])[0]
            self._json(200, {"sessions": list_recent_sessions(profile or None)})
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
            jid = (parse_qs(p.query).get("id") or [""])[0]
            # job_or_disk (not get_job) so a resumed session — from the
            # "resume a recent song" panel, or a bookmarked ?sid= link —
            # still works after a server restart wiped the in-memory JOBS
            # dict, as long as the session's files are still on disk.
            j = job_or_disk(jid)
            if not j:
                self._json(404, {"error": "unknown job"})
            else:
                self._json(200, j)
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
            jid = (parse_qs(p.query).get("id") or [""])[0]
            j = get_job(jid)
            if not j or j.get("status") != "done":
                self._json(404, {"error": "not ready"})
                return
            f = os.path.join(SESS_DIR, jid, j["final_file"])
            self._send_file(f, "application/octet-stream",
                            j.get("display_name"))
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
            if not j or j.get("status") != "done":
                self._json(404, {"error": "not ready"})
                return
            f = os.path.join(SESS_DIR, jid, "vocal_fx.wav")
            if not os.path.isfile(f):
                self._json(404, {"error": "vocal stem not found"})
                return
            name = (j.get("display_name") or "song").rsplit(".", 1)[0] + " (vocal only).wav"
            self._send_file(f, "audio/wav", name)
            return

        if p.path == "/final_video":
            jid = (parse_qs(p.query).get("id") or [""])[0]
            j = get_job(jid)
            if not j or j.get("video_status") != "done":
                self._json(404, {"error": "not ready"})
                return
            f = os.path.join(SESS_DIR, jid, j["final_video_file"])
            name = (j.get("display_name") or "karaoke").rsplit(".", 1)[0] + ".mp4"
            self._send_file(f, "video/mp4", name)
            return

        if p.path == "/phone":
            b = PHONE_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
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
            self.end_headers()
            self.wfile.write(b)
            return

        if p.path == "/join":
            b = JOIN_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return

        if p.path == "/host":
            b = HOST_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
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
            jid = uuid.uuid4().hex[:12]
            set_job(jid, status="queued")
            threading.Thread(target=do_prepare, args=(jid, url, profile), daemon=True).start()
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

            lock = _render_lock(jid)
            if not lock.acquire(blocking=False):
                self._json(409, {"error": "A render is already running for this session."})
                return
            fx = body.get("fx", {})
            mix_opts = body.get("mix", {})
            set_job(jid, status="rendering")
            threading.Thread(target=_run_render_locked,
                             args=(jid, vocal_path, fx, mix_opts, lock),
                             daemon=True).start()
            self._json(200, {"ok": True})
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
            video_path = os.path.join(sdir, f"phone_raw{ext}")
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
            set_job(jid, phone_status="uploaded",
                    phone_video_file=os.path.basename(video_path))
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
            if not j or j.get("status") != "done" or j.get("phone_status") != "uploaded":
                self._json(409, {"error": "not ready"})
                return
            if j.get("video_status") in ("aligning", "done"):
                self._json(200, {"ok": True})  # already running / done, avoid double-combine
                return
            set_job(jid, video_status="aligning", video_error=None)
            threading.Thread(target=do_combine_video, args=(jid,), daemon=True).start()
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
            tid = uuid.uuid4().hex[:8]
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
                tid=tid,
                pitch_score=body.get("pitch_score"),
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

        if p.path == "/comp":
            body = self._read_body()
            jid = body.get("id")
            head = _get_take(jid, body.get("head_take"))
            tail = _get_take(jid, body.get("tail_take"))
            if not head or not tail:
                self._json(404, {"error": "unknown take"})
                return
            try:
                punch_sec = float(body.get("punch_sec", 0))
            except (TypeError, ValueError):
                self._json(400, {"error": "punch_sec must be a number"})
                return
            sdir = os.path.join(SESS_DIR, jid)
            tid = uuid.uuid4().hex[:8]
            fname = f"take_{tid}.wav"
            out_path = os.path.join(sdir, fname)
            try:
                audio_fx.stitch_vocals(
                    os.path.join(sdir, head["file"]),
                    os.path.join(sdir, tail["file"]),
                    out_path, punch_sec,
                    crossfade_ms=float(body.get("crossfade_ms", 40)),
                )
            except Exception as e:
                self._json(500, {"error": "Could not join those takes: " + str(e)[:300]})
                return
            take = _add_take(
                jid, file=fname, duration=_probe_duration(out_path), kind="comp",
                punch_sec=punch_sec, source_take_ids=[head["id"], tail["id"]],
                tid=tid,
            )
            self._json(200, {"take": take})
            return

        if p.path == "/room/start":
            code = start_room()
            lan = _lan_ip()
            join_url = f"http://{lan}:{PORT}/join?room={code}" if lan else None
            host_url = f"http://{lan}:{PORT}/host" if lan else None
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

        self._json(404, {"error": "not found"})


PHONE_HTML = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Karaoke Studio — Phone Camera</title>
<style>
  html,body{height:100%;margin:0;background:#0b0b10;color:#eee;
    font:15px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;}
  #wrap{display:flex;flex-direction:column;height:100%;}
  #preview{flex:1;width:100%;object-fit:cover;background:#000;}
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
  <div style="position:relative;flex:1;">
    <video id="preview" autoplay muted playsinline></video>
    <button id="flip" onclick="flipCamera()">🔄 flip</button>
  </div>
  <div id="bar">
    <div id="status"><span class="dot" id="dot"></span><span id="statusText">Starting camera…</span></div>
    <div id="hint">Keep this tab open and the phone plugged in / screen on while you record.</div>
    <div id="err"></div>
  </div>
</div>
<script>
const params = new URLSearchParams(location.search);
const id = params.get('id');
const statusText = document.getElementById('statusText');
const dot = document.getElementById('dot');
const errEl = document.getElementById('err');
const video = document.getElementById('preview');
let stream = null, facing = 'environment';
let wakeLock = null;

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
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: {ideal:1920}, height: {ideal:1080} },
      audio: false,   // no phone-side audio needed anymore — nothing here
                       // ever gets used for sync or for the final mix.
    });
    video.srcObject = stream;
    await register();
    requestWakeLock();
    setStatus('Connected — streaming to desktop', 'ok');
    pollLoop();
  }catch(e){
    setStatus('Camera access failed', 'warn');
    showErr((e && e.message) || String(e));
  }
}

// Swap the camera in place (new getUserMedia + replaceTrack on both the
// recorder's stream and the live WebRTC sender) rather than tearing down
// and re-registering — a full re-register would bump webrtc_gen and force
// the desktop to rebuild its peer connection, causing a visible glitch in
// the live self-view every time you just want to flip front/back camera.
async function flipCamera(){
  facing = facing === 'environment' ? 'user' : 'environment';
  let newStream;
  try{
    newStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: {ideal:1920}, height: {ideal:1080} }, audio: false });
  }catch(e){ showErr((e && e.message) || String(e)); return; }
  const oldStream = stream;
  stream = newStream;
  video.srcObject = stream;
  if(pc){
    const sender = pc.getSenders().find(s=>s.track && s.track.kind==='video');
    const newVideoTrack = stream.getVideoTracks()[0];
    if(sender && newVideoTrack){ try{ await sender.replaceTrack(newVideoTrack); }catch(_){} }
  }
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
  stream.getVideoTracks().forEach(t=>pc.addTrack(t, stream));
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
document.addEventListener('visibilitychange', async ()=>{
  if(document.visibilityState==='visible') requestWakeLock();
});

async function pollLoop(){
  while(true){
    await pollWebRTC();
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
<style>
  body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);
    font:16px/1.4 -apple-system,system-ui,sans-serif;padding:24px;}
  h1{font-size:24px;margin:0 0 4px;color:var(--amber);text-shadow:var(--glow-amber);
    display:flex;align-items:center;gap:12px}
  .sub{color:var(--muted);margin:0 0 20px}
  .code{font:700 64px/1 var(--mono);letter-spacing:.1em;color:var(--amber);text-shadow:var(--glow-amber)}
  .code span:nth-child(2n){color:var(--pink);text-shadow:var(--glow-pink)}
  .join{color:var(--muted);font-size:14px}
  .join b{color:var(--ink)}
  .grid{display:grid;grid-template-columns:2fr 1fr;gap:20px;margin-top:20px}
  #stage{aspect-ratio:16/9;background:#000;border-radius:10px;overflow:hidden}
  #stage iframe,#ytplayer{width:100%;height:100%}
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
  .codeRow{display:flex;align-items:center;gap:20px}
  #qrImg{width:120px;height:120px;background:#fff;border-radius:10px;padding:6px;
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
</style>

<div id="start">
  <a href="/" style="color:var(--muted);font-size:13px;text-decoration:none">← Back to Studio</a>
  <h1>🎤 Karaoke Party <span class="marquee"><i class="bulb"></i><i class="bulb"></i><i class="bulb"></i><i class="bulb"></i></span></h1>
  <p class="sub">Start a room, then have guests join from their phones on the same WiFi.</p>
  <button class="btn amber" id="startBtn">Start Party</button>
</div>

<div id="live" class="hidden">
  <h1>🎤 Karaoke Party <span class="marquee"><i class="bulb"></i><i class="bulb"></i><i class="bulb"></i><i class="bulb"></i></span></h1>
  <div class="codeRow">
    <div>
      <div class="code" id="roomCode"></div>
      <div class="join">Guests: join at <b id="joinUrl"></b></div>
      <div class="join">📱 Run this from your phone: <b id="hostUrl"></b></div>
    </div>
    <img id="qrImg" class="hidden" alt="Scan to join">
  </div>
  <div class="grid">
    <div class="card">
      <div id="stage"><div class="empty">Nobody queued yet — waiting for guests to join and add songs.</div></div>
      <div class="nowSingers" id="singers"></div>
      <div class="nowTitle" id="songTitle"></div>
      <div style="margin-top:14px"><button class="btn pink" id="nextBtn">Next ▶</button></div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Up next <span style="font-size:11px;color:var(--dim);font-weight:400">— drag to reorder</span></h3>
      <ul class="upnext" id="upnext"><li class="empty">Queue is empty.</li></ul>
      <h3>👥 Who's here <span id="guestCount" style="color:var(--dim);font-weight:400"></span></h3>
      <div class="guestList" id="guestList"><span class="empty">Waiting for guests to join…</span></div>
    </div>
  </div>
  <div class="card challenges hidden" id="challengesCard">
    <h3>⚡ Challenges <span class="marquee"><i class="bulb"></i><i class="bulb"></i></span></h3>
    <div id="challengeList"></div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let state={code:null,queue:[],now_playing:null};
let currentEntryId=null, ytPlayer=null, ytReady=false, backingAudio=null, syncTimer=null;

$('startBtn').addEventListener('click',async()=>{
  const r=await fetch('/room/start',{method:'POST'}); const j=await r.json();
  $('start').classList.add('hidden'); $('live').classList.remove('hidden');
  $('roomCode').innerHTML=[...j.code].map(c=>`<span>${esc(c)}</span>`).join('');
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
}
function esc(s){ return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

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
<style>
  *{-webkit-tap-highlight-color:transparent}
  body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);
    font:16px/1.4 -apple-system,system-ui,sans-serif;
    padding:16px 16px calc(16px + env(safe-area-inset-bottom));max-width:480px;margin:0 auto}
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
</style>

<h1>🎤 Join the party <span class="marquee"><i class="bulb"></i><i class="bulb"></i><i class="bulb"></i></span></h1>

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
  <div class="whoRow">🎉 Joined as <b id="whoLabel"></b> · room <b id="whoRoom"></b>
    <button class="btn ghost small" id="leaveBtn" type="button" style="width:auto;margin-left:auto">Not you?</button></div>

  <div class="tabbar">
    <button class="tabbtn active" id="tabBtnFind" type="button">🔍 Find a song</button>
    <button class="tabbtn" id="tabBtnQueue" type="button">🎶 Party queue</button>
  </div>

  <div class="tabpane" id="tabFind">
    <label>Search for a song</label>
    <input type="text" id="q" placeholder="Search YouTube...">
    <label class="toggleRow"><input type="checkbox" id="karaokeSuffix" checked> 🎤 Add "karaoke" to search (helps surface lyrics videos)</label>
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
        <label class="toggleRow"><input type="checkbox" id="challengeMode"> 🎯 Challenge someone else to sing this instead</label>
        <input type="text" id="challengeTarget" class="hidden" placeholder="Who are you challenging?" style="margin-top:8px">
        <button class="btn pink" id="addBtn" type="button">Confirm — add to queue ✓</button>
      </div>
    </div>
    <div id="feedback"></div>
  </div>

  <div class="tabpane hidden" id="tabQueue">
    <div class="card" id="queueView">
      <h3 style="margin-top:0">🎶 Party queue</h3>
      <div id="qEmpty" style="color:var(--dim);font-size:13px">Nobody's queued yet.</div>
      <ul class="qlist hidden" id="qList"></ul>
      <div id="qChallenges"></div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">👥 Who's here</h3>
      <div class="guestList" id="guestList"></div>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const params=new URLSearchParams(location.search);
if(params.get('room')) $('room').value=params.get('room').toUpperCase();

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
  roomEvents.onmessage=(ev)=>{ lastState=JSON.parse(ev.data); renderQueueView(); renderGuests(); };
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
<style>
  body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);
    font:16px/1.4 -apple-system,system-ui,sans-serif;
    padding:16px 16px calc(16px + env(safe-area-inset-bottom));max-width:480px;margin:0 auto}
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

<h1>🎤 Party Host <span class="marquee"><i class="bulb"></i><i class="bulb"></i><i class="bulb"></i></span></h1>
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
    <h3 style="margin-top:0">⚡ Challenges</h3>
    <div id="challengeList"></div>
  </div>

  <div class="card">
    <h3 style="margin-top:0">Up next</h3>
    <ul class="upnext" id="upnext"><li class="empty">Queue is empty.</li></ul>
  </div>

  <div class="card">
    <h3 style="margin-top:0">👥 Who's here <span id="guestCount" style="color:var(--dim);font-weight:400"></span></h3>
    <div class="guestList" id="guestList"><span class="empty">Waiting for guests to join…</span></div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let state={code:null,queue:[],now_playing:null,challenges:[],guests:[]};

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
    print(f"\n  🎤 Karaoke Studio  →  http://{HOST}:{PORT}\n")

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
        print(f"              guests join at http://{lan}:{PORT}/join (once you start a room)\n")
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
