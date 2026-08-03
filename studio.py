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
    """Align the paired phone's video to the finished audio mix (via the sync
    chirp) and mux them into one final video file."""
    sdir = os.path.join(SESS_DIR, jid)
    j = get_job(jid)
    phone_video = os.path.join(sdir, j["phone_video_file"])
    final_mix = os.path.join(sdir, j["final_file"])
    out_video = os.path.join(sdir, "final_video.mp4")
    tmp = os.path.join(sdir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    try:
        audio_fx.align_and_mux_video(
            phone_video, final_mix, out_video,
            chirp_music_pos_sec=float(j.get("chirp_music_pos_sec", 0.0)),
            tmp_dir=tmp,
        )
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
            j = get_job(jid)
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
            self._json(200, {
                "lan_ip": lan,
                "port": PORT,
                "phone_url": f"http://{lan}:{PORT}/phone?id={jid}" if lan else None,
                "chirp": {
                    "f0": audio_fx.CHIRP_F0,
                    "f1": audio_fx.CHIRP_F1,
                    "dur_ms": int(audio_fx.CHIRP_DUR * 1000),
                },
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

        if p.path == "/phone/poll":
            q = parse_qs(p.query)
            jid = (q.get("id") or [""])[0]
            j = get_job(jid)
            if not j or not j.get("phone_paired"):
                self._json(404, {"error": "not paired"})
                return
            set_job(jid, phone_last_seen=time.time())
            self._json(200, {"cmd": j.get("phone_cmd"), "seq": j.get("phone_cmd_seq", 0)})
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
            chirp_pos = body.get("chirp_music_pos_sec")
            set_job(jid, status="rendering",
                    chirp_music_pos_sec=float(chirp_pos) if chirp_pos is not None else 0.0)
            threading.Thread(target=_run_render_locked,
                             args=(jid, vocal_path, fx, mix_opts, lock),
                             daemon=True).start()
            self._json(200, {"ok": True})
            return

        if p.path == "/phone/register":
            body = self._read_body()
            jid = body.get("id")
            if not job_or_disk(jid):
                self._json(404, {"error": "unknown session"})
                return
            set_job(jid, phone_paired=True, phone_status="paired",
                    phone_cmd=None, phone_cmd_seq=0, phone_last_seen=time.time())
            self._json(200, {"ok": True})
            return

        if p.path == "/phone/cmd":
            body = self._read_body()
            jid = body.get("id")
            cmd = body.get("cmd")
            if cmd not in ("start", "stop"):
                self._json(400, {"error": "bad cmd"})
                return
            j = get_job(jid)
            if not j or not j.get("phone_paired"):
                self._json(409, {"error": "phone not paired"})
                return
            seq = j.get("phone_cmd_seq", 0) + 1
            set_job(jid, phone_cmd=cmd, phone_cmd_seq=seq,
                    phone_status="recording" if cmd == "start" else "stopping")
            self._json(200, {"ok": True, "seq": seq})
            return

        if p.path == "/phone/upload":
            body = self._read_body()
            jid = body.get("id")
            j = job_or_disk(jid)
            if not j:
                self._json(404, {"error": "unknown session"})
                return
            b64 = body.get("video_b64", "")
            if not b64:
                self._json(400, {"error": "No video received."})
                return
            mime = body.get("mime", "video/webm")
            ext = ".mp4" if "mp4" in mime else ".webm"
            sdir = os.path.join(SESS_DIR, jid)
            os.makedirs(sdir, exist_ok=True)
            video_path = os.path.join(sdir, f"phone_raw{ext}")
            try:
                with open(video_path, "wb") as fh:
                    fh.write(base64.b64decode(b64))
            except Exception:
                self._json(400, {"error": "Could not decode video."})
                return
            set_job(jid, phone_status="uploaded",
                    phone_video_file=os.path.basename(video_path))
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
let stream = null, recorder = null, chunks = [], facing = 'environment';
let lastSeq = 0, wakeLock = null;

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

async function startCamera(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: {ideal:1280}, height: {ideal:720} },
      audio: true,   // kept ON purely as a sync reference (picks up the chirp);
                     // discarded later — final output uses the app's own mix.
    });
    video.srcObject = stream;
    await register();
    requestWakeLock();
    setStatus('Connected — waiting for Record on desktop', 'ok');
    pollLoop();
  }catch(e){
    setStatus('Camera access failed', 'warn');
    showErr((e && e.message) || String(e));
  }
}

async function flipCamera(){
  facing = facing === 'environment' ? 'user' : 'environment';
  if(stream) stream.getTracks().forEach(t=>t.stop());
  await startCamera();
}

async function register(){
  try{
    await fetch('/phone/register', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({id})});
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

function pickMime(){
  for(const m of ['video/webm;codecs=vp8,opus','video/webm','video/mp4']){
    if(window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  }
  return '';
}

function startPhoneRec(){
  chunks = [];
  recorder = new MediaRecorder(stream, {
    mimeType: pickMime(),
    videoBitsPerSecond: 2500000,
    audioBitsPerSecond: 128000,
  });
  recorder.ondataavailable = e=>{ if(e.data.size) chunks.push(e.data); };
  recorder.onstop = uploadPhoneVideo;
  recorder.start();
  setStatus('🔴 Recording', 'rec');
}

function stopPhoneRec(){
  if(recorder && recorder.state !== 'inactive') recorder.stop();
}

function blobToB64(blob){
  return new Promise((res, rej)=>{
    const fr = new FileReader();
    fr.onload = ()=>res(fr.result.split(',')[1]);
    fr.onerror = rej;
    fr.readAsDataURL(blob);
  });
}

async function uploadPhoneVideo(){
  setStatus('Uploading…', 'warn');
  try{
    const blob = new Blob(chunks, {type: recorder.mimeType || 'video/webm'});
    const b64 = await blobToB64(blob);
    const r = await fetch('/phone/upload', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({id, video_b64: b64, mime: blob.type})});
    setStatus(r.ok ? 'Uploaded ✓ — you can put the phone down' : 'Upload failed', r.ok ? 'ok' : 'warn');
  }catch(e){
    setStatus('Upload failed', 'warn');
    showErr((e && e.message) || String(e));
  }
}

async function pollLoop(){
  while(true){
    try{
      const r = await fetch('/phone/poll?id=' + encodeURIComponent(id) + '&seq=' + lastSeq);
      if(r.ok){
        const j = await r.json();
        if(j.seq > lastSeq){
          lastSeq = j.seq;
          if(j.cmd === 'start' && (!recorder || recorder.state === 'inactive')) startPhoneRec();
          else if(j.cmd === 'stop' && recorder && recorder.state !== 'inactive') stopPhoneRec();
        }
      }
    }catch(_){}
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
  .guestChip{background:var(--track);border:1px solid var(--edge);border-radius:999px;
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


INDEX_HTML = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Karaoke Studio</title>
<style>
  :root{
    --panel:#15171c; --panel2:#1c1f26; --rack:#0e0f13; --edge:#2a2e38;
    --ink:#e8ebf0; --muted:#8b93a3; --dim:#5a6172;
    --amber:#ffb454; --amber2:#ff9b3d; --rec:#ff3b47; --ok:#39d98a;
    --fader:#3a3f4c; --track:#0a0b0e;
    --mono:'SF Mono',ui-monospace,Menlo,monospace;
  }
  *{box-sizing:border-box}
  body{
    margin:0;min-height:100vh;background:
      radial-gradient(1200px 600px at 70% -10%, #20242e 0%, transparent 60%),
      #0b0c0f;
    color:var(--ink);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    padding:20px;
  }
  .wrap{max-width:1180px;margin:0 auto}
  header{display:flex;align-items:baseline;gap:14px;margin-bottom:18px}
  .logo{font-family:var(--mono);font-weight:700;font-size:19px;letter-spacing:.14em;
    text-transform:uppercase;color:var(--ink)}
  .logo b{color:var(--amber)}
  .tag{color:var(--dim);font-size:12px;font-family:var(--mono);letter-spacing:.08em}

  .grid{display:grid;grid-template-columns:1.35fr .95fr;gap:16px}
  @media(max-width:900px){.grid{grid-template-columns:1fr}}

  .card{background:linear-gradient(180deg,var(--panel2),var(--panel));
    border:1px solid var(--edge);border-radius:12px;padding:16px;
    box-shadow:0 10px 30px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.03)}
  .card h2{margin:0 0 12px;font-family:var(--mono);font-size:11px;letter-spacing:.16em;
    text-transform:uppercase;color:var(--muted);font-weight:600;
    display:flex;align-items:center;gap:8px}
  .card h2::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--amber)}

  input[type=text]{width:100%;padding:11px 13px;border-radius:8px;border:1px solid var(--edge);
    background:#0a0b0e;color:var(--ink);font-size:13px;outline:none;font-family:var(--mono)}
  input[type=text]:focus{border-color:var(--amber2)}
  .btn{padding:11px 18px;border:none;border-radius:8px;cursor:pointer;font-weight:650;font-size:13px;
    background:linear-gradient(180deg,#3a3f4c,#2b2f39);color:var(--ink);
    border:1px solid var(--edge);transition:filter .12s}
  .btn:hover{filter:brightness(1.15)}
  .btn:disabled{opacity:.4;cursor:default}
  .btn.amber{background:linear-gradient(180deg,var(--amber),var(--amber2));color:#211500;border-color:#a5651f}
  .btn.rec{background:linear-gradient(180deg,#ff4d58,#e0313c);color:#fff;border-color:#a51f27}
  .btn.wide{width:100%}

  .row{display:flex;gap:10px;align-items:center}
  .mt{margin-top:12px}

  /* video / lyrics stage */
  .stage{position:relative;background:#000;border-radius:10px;overflow:hidden;aspect-ratio:16/9;
    border:1px solid #000;box-shadow:inset 0 0 40px rgba(0,0,0,.6)}
  .stage iframe{width:100%;height:100%;border:0;display:block}
  .stage .placeholder{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
    flex-direction:column;gap:8px;color:var(--dim);font-family:var(--mono);font-size:12px;letter-spacing:.08em}
  .placeholder .big{font-size:40px;opacity:.4}
  .stage #ytplayer,.stage iframe{width:100%;height:100%}
  .stage{position:relative}
  .novideo{position:absolute;bottom:8px;right:8px;z-index:5;font-family:var(--mono);font-size:10px;
    letter-spacing:.06em;color:var(--muted);background:rgba(10,11,14,.82);border:1px solid var(--edge);
    border-radius:6px;padding:4px 8px;cursor:pointer}
  .novideo:hover{color:var(--amber);border-color:var(--amber2)}
  .vfallback{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;
    gap:12px;text-align:center;padding:22px;
    background:radial-gradient(600px 300px at 50% 0%,#20242e 0%,transparent 70%),#0b0c0f}
  .vf-note{font-family:var(--mono);font-size:11px;letter-spacing:.1em;color:var(--amber);text-transform:uppercase}
  .vf-title{font-size:16px;font-weight:650;color:var(--ink);max-width:80%}
  .vf-clock{font-family:var(--mono);font-size:44px;font-weight:700;color:var(--amber);letter-spacing:.04em;line-height:1}
  .vf-link{text-decoration:none;background:linear-gradient(180deg,#3a3f4c,#2b2f39);border:1px solid var(--edge);
    color:var(--ink);padding:9px 16px;border-radius:8px;font-size:12.5px;font-weight:600}
  .vf-link:hover{filter:brightness(1.15)}
  .vf-hint{font-family:var(--mono);font-size:11px;color:var(--dim);max-width:82%;line-height:1.6}

  /* transport */
  .transport{display:flex;align-items:center;gap:12px;margin-top:12px;flex-wrap:wrap}
  .seekwrap{display:flex;align-items:center;gap:12px}
  #seek{flex:1}
  #seek:disabled{opacity:.4}
  .keybox{display:flex;align-items:center;gap:8px;background:#0a0b0e;border:1px solid var(--edge);
    border-radius:8px;padding:5px 10px}
  .klabel{font-family:var(--mono);font-size:10px;letter-spacing:.14em;color:var(--dim)}
  .kval{font-family:var(--mono);font-size:14px;color:var(--amber);min-width:26px;text-align:center;font-weight:700}
  .ksmall{width:24px;height:24px;border-radius:6px;border:1px solid var(--edge);background:linear-gradient(180deg,#3a3f4c,#2b2f39);
    color:var(--ink);cursor:pointer;font-size:15px;line-height:1;font-weight:700;padding:0}
  .ksmall:hover{filter:brightness(1.2)} .ksmall:disabled{opacity:.4;cursor:default}
  .keyhint{font-family:var(--mono);font-size:11px;color:var(--dim)}
  .punch{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px;color:var(--muted);
    letter-spacing:.04em;cursor:pointer;user-select:none}
  .punch input{accent-color:var(--amber2)}
  #punchhint{color:var(--amber);font-family:var(--mono);font-size:11px}
  .time{font-family:var(--mono);font-size:13px;color:var(--amber);min-width:104px;letter-spacing:.05em}
  .reclamp{width:11px;height:11px;border-radius:50%;background:#3a2225;box-shadow:inset 0 0 4px #000}
  .reclamp.on{background:var(--rec);box-shadow:0 0 10px var(--rec),0 0 3px #fff}

  /* level meter */
  .meter{height:12px;border-radius:6px;background:var(--track);overflow:hidden;border:1px solid #000;position:relative}
  .meter .lvl{height:100%;width:0%;background:linear-gradient(90deg,var(--ok) 0%,var(--ok) 60%,#e8c341 78%,var(--rec) 100%);
    transition:width .05s linear}
  .metlabel{font-family:var(--mono);font-size:10px;color:var(--dim);letter-spacing:.1em;margin-bottom:5px}

  /* fx rack */
  .rack{background:var(--rack);border:1px solid #000;border-radius:10px;padding:14px;
    box-shadow:inset 0 2px 10px rgba(0,0,0,.6)}
  .fxrow{display:flex;align-items:center;gap:12px;padding:9px 0;border-bottom:1px dashed #23262e}
  .fxrow:last-child{border-bottom:none}
  .fxgroup{font-family:var(--mono);font-size:10px;letter-spacing:.12em;color:var(--dim);
    text-transform:uppercase;margin-top:10px;padding-top:8px;border-top:1px dashed #23262e}
  .fxgroup:first-child{margin-top:0;padding-top:0;border-top:none}
  .fxname{width:118px;font-family:var(--mono);font-size:11px;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}
  .fxname.on{color:var(--amber)}
  input[type=range]{-webkit-appearance:none;appearance:none;height:4px;border-radius:2px;
    background:#2a2e38;flex:1;outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;
    background:radial-gradient(circle at 35% 30%,#fff 0%,var(--amber) 40%,var(--amber2) 100%);
    cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,.6),0 0 0 1px #a5651f}
  input[type=range]:disabled::-webkit-slider-thumb{background:#4a4f5c;box-shadow:none}
  .val{font-family:var(--mono);font-size:11px;color:var(--muted);width:52px;text-align:right}
  .tog{width:34px;height:19px;border-radius:11px;background:#2a2e38;position:relative;cursor:pointer;
    border:1px solid #000;flex:none;transition:background .15s}
  .tog.on{background:var(--amber2)}
  .tog::after{content:"";position:absolute;top:2px;left:2px;width:13px;height:13px;border-radius:50%;
    background:#dfe3ea;transition:left .15s}
  .tog.on::after{left:16px}

  select{padding:8px 10px;border-radius:7px;border:1px solid var(--edge);background:#0a0b0e;
    color:var(--ink);font-size:12px;font-family:var(--mono);outline:none}
  select:focus{border-color:var(--amber2)}

  .mixrow{display:flex;align-items:center;gap:12px;padding:8px 0}
  .mixrow .fxname{width:90px}

  /* search tabs + results */
  .tabs{display:flex;gap:6px;margin-bottom:14px}
  .tab{background:transparent;border:1px solid var(--edge);color:var(--muted);font-family:var(--mono);
    font-size:11px;letter-spacing:.08em;text-transform:uppercase;padding:7px 14px;border-radius:8px;cursor:pointer}
  .tab.on{background:linear-gradient(180deg,#2b2f39,#232730);color:var(--amber);border-color:var(--amber2)}
  .results{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;margin-top:14px}
  .rcard{display:flex;flex-direction:column;background:#0a0b0e;border:1px solid var(--edge);border-radius:10px;
    overflow:hidden;cursor:pointer;transition:border-color .12s,transform .12s}
  .rcard:hover{border-color:var(--amber2);transform:translateY(-2px)}
  .rthumb{position:relative;width:100%;aspect-ratio:16/9;background:#111;object-fit:cover;display:block}
  .rdur{position:absolute;bottom:6px;right:6px;background:rgba(0,0,0,.82);color:#fff;font-family:var(--mono);
    font-size:10px;padding:2px 5px;border-radius:4px}
  .rmeta{padding:9px 10px}
  .rtitle{font-size:12.5px;line-height:1.35;color:var(--ink);display:-webkit-box;-webkit-line-clamp:2;
    -webkit-box-orient:vertical;overflow:hidden}
  .rchan{font-family:var(--mono);font-size:10px;color:var(--dim);margin-top:5px;letter-spacing:.03em;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .rcard.loading{opacity:.5;pointer-events:none}
  /* live pitch tuner */
  .tuner{background:#0a0b0e;border:1px solid var(--edge);border-radius:12px;padding:14px;margin-top:12px;text-align:center}
  .tuner-note{font-family:var(--mono);font-size:34px;font-weight:700;color:var(--dim);line-height:1;letter-spacing:.02em;transition:color .1s}
  .tuner-note.good{color:var(--ok)} .tuner-note.off{color:var(--amber)}
  .tuner-scale{position:relative;height:26px;margin:10px auto 4px;max-width:340px;
    background:linear-gradient(90deg,rgba(248,113,113,.18),rgba(74,217,138,.22) 50%,rgba(248,113,113,.18));
    border-radius:6px;border:1px solid #000;overflow:hidden}
  .tuner-center{position:absolute;left:50%;top:0;bottom:0;width:2px;background:var(--ok);transform:translateX(-1px);opacity:.7}
  .tuner-needle{position:absolute;top:2px;bottom:2px;width:4px;left:50%;border-radius:2px;background:#fff;
    transform:translateX(-2px);transition:left .06s linear,background .1s}
  .tuner-msg{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.04em}
  /* live monitor volumes */
  .mons{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin-top:12px;
    background:#0a0b0e;border:1px solid var(--edge);border-radius:10px;padding:10px 12px}
  .monrow{display:flex;align-items:center;gap:8px;flex:1;min-width:180px}
  .monlbl{font-family:var(--mono);font-size:10px;letter-spacing:.1em;color:var(--dim);text-transform:uppercase;white-space:nowrap}
  .monrow input[type=range]{flex:1}
  .monval{font-family:var(--mono);font-size:11px;color:var(--amber);min-width:38px;text-align:right}
  .tunertog{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:10px;
    letter-spacing:.08em;color:var(--muted);text-transform:uppercase;cursor:pointer}
  .tunertog input{accent-color:var(--amber2)}
  /* continue/keep choice bar */
  .choice{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:12px;padding:12px;
    background:linear-gradient(180deg,#20242e,#191c23);border:1px solid var(--amber2);border-radius:10px}
  .choice-msg{font-family:var(--mono);font-size:12px;color:var(--amber);flex:1;min-width:160px}
  /* take mini-mixer + test mode */
  .takemix{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin-top:12px;
    background:#0a0b0e;border:1px solid var(--edge);border-radius:10px;padding:10px 12px}
  .takemix .monrow{min-width:190px}
  #testModeBtn{margin-left:auto}
  .testpanel{margin-top:12px;padding:14px;border-radius:12px;
    background:linear-gradient(180deg,#1a1f2a,#141821);border:1px solid var(--accent2)}
  .testpanel h3{margin:0 0 4px;font-family:var(--mono);font-size:12px;letter-spacing:.14em;
    text-transform:uppercase;color:var(--accent)}
  .testpanel .tp-sub{font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:12px}
  .loopbar{position:relative;height:34px;background:#0a0b0e;border:1px solid #000;border-radius:8px;
    margin:10px 0;overflow:hidden;cursor:pointer}
  .loopregion{position:absolute;top:0;bottom:0;background:rgba(110,168,254,.28);border-left:2px solid var(--accent);
    border-right:2px solid var(--accent)}
  .loopplay{position:absolute;top:0;bottom:0;width:2px;background:#fff;opacity:.85}
  .looprow{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:8px}
  .tp-btn{padding:9px 16px;border:1px solid var(--edge);border-radius:8px;cursor:pointer;font-weight:650;font-size:13px;
    background:linear-gradient(180deg,#3a3f4c,#2b2f39);color:var(--ink)}
  .tp-btn.on{background:linear-gradient(180deg,var(--accent),var(--accent2));color:#0b0e13;border-color:var(--accent2)}
  .tp-btn:hover{filter:brightness(1.15)}
  .tp-note{font-family:var(--mono);font-size:11px;color:var(--dim);letter-spacing:.03em}
  .hint{color:var(--dim);font-size:11.5px;line-height:1.6;margin-top:10px;font-family:var(--mono);letter-spacing:.02em}
  .err{color:var(--rec);font-size:12.5px;margin-top:10px;font-family:var(--mono)}
  .status{font-family:var(--mono);font-size:12px;color:var(--muted);letter-spacing:.04em}
  .dl{display:inline-block;margin-top:12px;text-decoration:none;background:linear-gradient(180deg,var(--ok),#28b873);
    color:#052012;padding:11px 20px;border-radius:8px;font-weight:700;font-size:13px}
  /* reusable mini-player control row (used for Play take + final song) */
  .player{background:#0a0b0e;border:1px solid var(--edge);border-radius:10px;padding:12px;margin-top:12px}
  .player .ctrls{display:flex;align-items:center;gap:8px}
  .pbtn{width:38px;height:34px;border-radius:8px;border:1px solid var(--edge);cursor:pointer;
    background:linear-gradient(180deg,#3a3f4c,#2b2f39);color:var(--ink);font-size:14px;line-height:1;
    display:flex;align-items:center;justify-content:center}
  .pbtn:hover{filter:brightness(1.18)} .pbtn:disabled{opacity:.4;cursor:default}
  .pbtn.play{background:linear-gradient(180deg,var(--amber),var(--amber2));color:#211500;border-color:#a5651f}
  .player .pseek{flex:1}
  .player .ptime{font-family:var(--mono);font-size:11px;color:var(--amber);min-width:96px;text-align:right;letter-spacing:.04em}
  .player .plabel{font-family:var(--mono);font-size:10px;letter-spacing:.14em;color:var(--dim);
    text-transform:uppercase;margin-bottom:8px}
  .pwave{width:100%;height:64px;display:block;border-radius:6px;background:var(--track);
    border:1px solid var(--edge);margin-bottom:10px;cursor:pointer}
  .step{opacity:.4;pointer-events:none;transition:opacity .3s}
  .step.active{opacity:1;pointer-events:auto}
  .bar{height:6px;background:#0a0b0e;border-radius:4px;overflow:hidden;border:1px solid #000;margin-top:8px}
  .bar .fill{height:100%;width:0;background:linear-gradient(90deg,var(--amber2),var(--amber));transition:width .3s}
  kbd{font-family:var(--mono);background:#0a0b0e;border:1px solid var(--edge);border-radius:4px;padding:1px 6px;font-size:11px;color:var(--amber)}
</style>

<div class="wrap">
  <header>
    <div class="logo">KARAOKE <b>STUDIO</b></div>
    <div class="tag">// record · tune · mix · export</div>
  </header>

  <!-- STEP 1: find a song -->
  <div class="card" style="margin-bottom:16px">
    <h2>1 · Find a song</h2>
    <div class="tabs">
      <button class="tab on" id="tabSearch">Search YouTube</button>
      <button class="tab" id="tabUrl">Paste a link</button>
    </div>

    <!-- search mode -->
    <div id="searchMode">
      <div class="row">
        <input id="q" type="text" placeholder="Search e.g. “tum hi ho karaoke lyrics”…" spellcheck="false" autocomplete="off">
        <button id="searchBtn" class="btn amber">Search</button>
      </div>
      <div class="status mt" id="searchStatus"></div>
      <div class="results" id="results"></div>
    </div>

    <!-- url mode -->
    <div id="urlMode" style="display:none">
      <div class="row">
        <input id="url" type="text" placeholder="Paste a YouTube URL…" spellcheck="false" autocomplete="off">
        <button id="load" class="btn amber">Load</button>
      </div>
    </div>

    <div class="bar" id="loadbar" style="display:none"><div class="fill" id="loadfill"></div></div>
    <div class="status mt" id="loadstatus"></div>
    <div class="err" id="loaderr"></div>
  </div>

  <div class="grid">
    <!-- LEFT: stage + transport + recorder -->
    <div>
      <div class="card step" id="stageCard">
        <h2>2 · Sing along · record vocals</h2>
        <div class="stage" id="stage">
          <div class="placeholder"><div class="big">♪</div>load a track to see the video</div>
        </div>

        <!-- seek / scrub bar -->
        <div class="seekwrap mt">
          <input type="range" id="seek" min="0" max="1000" value="0" step="1" disabled>
          <div class="time" id="time">00:00 / 00:00</div>
        </div>

        <!-- music transport (independent of recording) -->
        <div class="transport">
          <button id="play" class="btn" disabled title="Play / pause the soundtrack">▶ Play music</button>
          <button id="stopmusic" class="btn" disabled title="Stop and go back to the start">■ Stop</button>
          <div class="keybox" title="Transpose the soundtrack to your key">
            <span class="klabel">KEY</span>
            <button id="keydn" class="ksmall" disabled>−</button>
            <span class="kval" id="keyval">0</span>
            <button id="keyup" class="ksmall" disabled>+</button>
          </div>
          <div class="keyhint" id="keyhint"></div>
        </div>

        <!-- live pitch tuner: shows the note you're singing so you stay on track -->
        <div class="tuner" id="tuner" style="display:none">
          <div class="tuner-note" id="tunerNote">—</div>
          <div class="tuner-scale">
            <div class="tuner-needle" id="tunerNeedle"></div>
            <div class="tuner-center"></div>
          </div>
          <div class="tuner-msg" id="tunerMsg">sing a note…</div>
        </div>

        <div class="metlabel mt">MIC INPUT</div>
        <div class="meter"><div class="lvl" id="mic"></div></div>

        <!-- live monitor volumes (in your headphones; do not change recording) -->
        <div class="mons">
          <div class="monrow">
            <span class="monlbl">Music vol</span>
            <input type="range" id="musVol" min="0" max="100" value="100">
            <span class="monval" id="musVolV">100%</span>
          </div>
          <div class="monrow">
            <span class="monlbl">My voice</span>
            <input type="range" id="monVol" min="0" max="100" value="0" title="Hear yourself in headphones">
            <span class="monval" id="monVolV">off</span>
          </div>
          <label class="tunertog"><input type="checkbox" id="tunerToggle" checked> pitch guide</label>
        </div>

        <!-- vocal timing calibration: pulls your recorded voice earlier to
             cancel mic input latency (the "vocals lag the track" problem) -->
        <div class="mons" style="margin-top:8px">
          <div class="monrow">
            <span class="monlbl" title="Move your recorded vocals earlier/later to line up with the music">Vocal timing</span>
            <input type="range" id="vocalDelay" min="-100" max="400" value="140" step="5">
            <span class="monval" id="vocalDelayV">140ms</span>
          </div>
          <span class="tunertog" id="delayHint" style="cursor:default">vocals lag → slide right</span>
        </div>

        <!-- record transport -->
        <div class="transport">
          <div class="reclamp" id="lamp"></div>
          <button id="rec" class="btn rec" disabled>● Record</button>
          <button id="stop" class="btn" disabled>■ Stop</button>
          <button id="playtake" class="btn" disabled title="Hear your take over the music">▶ Play take</button>
        </div>

        <!-- after-stop choice: continue from here, or keep what I sang -->
        <div class="choice" id="stopChoice" style="display:none">
          <span class="choice-msg" id="choiceMsg"></span>
          <button class="btn" id="continueBtn" title="Re-sing from where you stopped, keeping the earlier part">▶ Continue from here</button>
          <button class="btn" id="keepBtn" title="Keep this take as it is">✓ Keep as-is</button>
          <button class="btn" id="redoBtn" title="Throw this take away and sing again from the start">↺ Redo all</button>
        </div>

        <div class="hint">
          Wear headphones so your mic captures only your voice. The clean track is what
          records &amp; exports — the sliders above only change what YOU hear.
          <kbd>Space</kbd> record · <kbd>P</kbd> play/pause.
        </div>

        <!-- phone camera pairing: film with your phone, synced to this app's audio -->
        <div class="hint" id="phonePairPanel" style="margin-top:8px">
          📱 <span id="phoneStatusText">Loading phone-pairing link…</span>
          <div id="phoneUrlBox" style="display:none;margin-top:6px">
            <code id="phoneUrlText" style="user-select:all"></code>
            <button class="btn" id="phoneCopyBtn" type="button" style="margin-left:8px">Copy link</button>
            <div style="opacity:.8;margin-top:4px">Open on your phone (same WiFi). Android only for now —
              camera access over a plain link needs a one-time browser setting; see the
              <a href="https://github.com/TanSin18/Karaoke-Studio#readme" target="_blank" rel="noopener">README</a>.</div>
          </div>
        </div>

        <!-- your recorded take: full play/pause/stop/seek, appears after recording -->
        <div class="player" id="takePlayer" style="display:none"></div>
      </div>
    </div>

    <!-- RIGHT: fx rack + mixer + export -->
    <div>
      <div class="card step" id="fxCard">
        <h2>3 · Voice FX rack</h2>
        <div class="rack" id="rack">
          <!-- pitch -->
          <div class="fxrow">
            <div class="tog" data-fx="pitch"></div>
            <div class="fxname" data-name="pitch">Pitch fix</div>
            <select id="pkey">
              <option>C</option><option>C#</option><option>D</option><option>D#</option>
              <option>E</option><option>F</option><option>F#</option><option>G</option>
              <option>G#</option><option>A</option><option>A#</option><option>B</option>
            </select>
            <select id="pscale">
              <option value="major">major</option>
              <option value="minor">minor</option>
              <option value="chromatic">chromatic</option>
            </select>
          </div>
          <div class="fxrow">
            <div style="width:34px"></div>
            <div class="fxname">↳ strength</div>
            <input type="range" id="pstr" min="0" max="100" value="80">
            <div class="val" id="pstrv">80%</div>
          </div>
          <!-- pro vocal chain: clean EQ -> fast/smooth comp -> de-esser ->
               tone EQ -> saturation -> reverb/delay sends. All stages are
               always applied together (it's one signal path, not toggles
               per effect) but every parameter below is adjustable. -->
          <div class="fxrow">
            <div class="tog" data-fx="pro"></div>
            <div class="fxname" data-name="pro">Pro vocal chain</div>
            <div style="flex:1;color:var(--dim);font-family:var(--mono);font-size:11px">
              studio EQ / compression / de-ess / reverb chain — adjustable below
            </div>
            <button id="proReset" class="btn" style="padding:4px 10px;font-size:11px" title="Reset every value below to the studio defaults">↺ Reset</button>
          </div>
          <div id="proChainRack"></div>
          <!-- denoise -->
          <div class="fxrow">
            <div class="tog" data-fx="denoise"></div>
            <div class="fxname" data-name="denoise">De-noise</div>
            <input type="range" id="denoise" min="0" max="100" value="40" disabled>
            <div class="val" id="denoisev">40%</div>
          </div>
        </div>
      </div>

      <div class="card step mt" id="mixCard" style="margin-top:16px">
        <h2>4 · Mix &amp; export</h2>
        <div class="mixrow">
          <div class="fxname">Vocal</div>
          <input type="range" id="vgain" min="-12" max="12" value="0">
          <div class="val" id="vgainv">0 dB</div>
        </div>
        <div class="mixrow">
          <div class="fxname">Music</div>
          <input type="range" id="mgain" min="-18" max="6" value="-3">
          <div class="val" id="mgainv">-3 dB</div>
        </div>
        <div class="row mt">
          <select id="ofmt">
            <option value="wav">WAV · 24-bit</option>
            <option value="flac">FLAC · lossless</option>
            <option value="mp3">MP3 · 320k</option>
          </select>
          <button id="render" class="btn amber wide" disabled>Render final song</button>
        </div>
        <div class="bar" id="renderbar" style="display:none"><div class="fill" id="renderfill"></div></div>
        <div class="status mt" id="renderstatus"></div>
        <div id="result"></div>
        <div class="err" id="rendererr"></div>
      </div>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let SID=null, poll=null;
let mediaStream=null, recorder=null, recChunks=[], audioCtx=null;
let backingAudio=null, recording=false, startT=0, dur=0, rafMeter=null;
// declared early so applyLiveFx() (called during init via bindVal) never hits a
// temporal-dead-zone error that would halt the whole top-level script.
let takeAudio=null, takePlayerUI=null, takeGraph=null;

// ---- FX toggle state ----
const fxOn = {pitch:true, pro:true, denoise:false};

// ---- pro vocal chain: Clean EQ -> Fast/Smooth Comp -> De-esser -> Tone EQ ->
// Saturation -> Reverb/Delay sends. Defaults here MUST match
// audio_fx.PRO_CHAIN_DEFAULTS on the backend — the "Reset" button re-fetches
// the backend's copy via /pro_chain_defaults so the two can never drift for
// long, but the initial render uses this local copy to avoid a fetch race
// before the rack can be shown.
const PRO_CHAIN_FIELDS = [
  {group:'Clean EQ', fields:[
    {key:'hpf_freq', label:'HPF freq', min:20, max:200, step:1, unit:'Hz', dp:0, def:85},
    {key:'dip_freq', label:'Low-mid dip freq', min:100, max:500, step:1, unit:'Hz', dp:0, def:250},
    {key:'dip_gain', label:'Low-mid dip gain', min:-12, max:0, step:0.1, unit:'dB', dp:1, def:-2.5},
    {key:'dip_q', label:'Low-mid dip Q', min:0.3, max:5, step:0.1, unit:'', dp:1, def:1.4},
    {key:'nasal_freq', label:'Nasal cut freq', min:500, max:1500, step:1, unit:'Hz', dp:0, def:900},
    {key:'nasal_gain', label:'Nasal cut gain', min:-12, max:0, step:0.1, unit:'dB', dp:1, def:-1.5},
    {key:'nasal_q', label:'Nasal cut Q', min:0.3, max:5, step:0.1, unit:'', dp:1, def:2.0},
  ]},
  {group:'Fast compressor', fields:[
    {key:'fast_ratio', label:'Ratio', min:1, max:20, step:0.1, unit:':1', dp:1, def:4},
    {key:'fast_attack', label:'Attack', min:0.1, max:50, step:0.1, unit:'ms', dp:1, def:1},
    {key:'fast_release', label:'Release', min:10, max:500, step:1, unit:'ms', dp:0, def:80},
    {key:'fast_threshold_db', label:'Threshold', min:-40, max:0, step:0.5, unit:'dB', dp:1, def:-18},
  ]},
  {group:'Smooth compressor', fields:[
    {key:'smooth_ratio', label:'Ratio', min:1, max:20, step:0.1, unit:':1', dp:1, def:3},
    {key:'smooth_attack', label:'Attack', min:1, max:200, step:1, unit:'ms', dp:0, def:25},
    {key:'smooth_release', label:'Release', min:50, max:2000, step:10, unit:'ms', dp:0, def:500},
    {key:'smooth_threshold_db', label:'Threshold', min:-40, max:0, step:0.5, unit:'dB', dp:1, def:-20},
  ]},
  {group:'De-esser', fields:[
    {key:'deess_freq', label:'Frequency', min:2000, max:10000, step:100, unit:'Hz', dp:0, def:6000},
    {key:'deess_intensity', label:'Intensity', min:0, max:100, step:1, unit:'%', dp:0, def:40},
  ]},
  {group:'Tone EQ', fields:[
    {key:'tone1_freq', label:'Chest warmth freq', min:80, max:400, step:1, unit:'Hz', dp:0, def:180},
    {key:'tone1_gain', label:'Chest warmth gain', min:-6, max:6, step:0.1, unit:'dB', dp:1, def:1.0},
    {key:'tone1_q', label:'Chest warmth Q', min:0.3, max:5, step:0.1, unit:'', dp:1, def:1.0},
    {key:'tone2_freq', label:'Presence freq', min:2000, max:6000, step:10, unit:'Hz', dp:0, def:3800},
    {key:'tone2_gain', label:'Presence gain', min:-6, max:6, step:0.1, unit:'dB', dp:1, def:2.0},
    {key:'tone2_q', label:'Presence Q', min:0.3, max:5, step:0.1, unit:'', dp:1, def:1.2},
    {key:'air_freq', label:'Air shelf freq', min:6000, max:16000, step:100, unit:'Hz', dp:0, def:10500},
    {key:'air_gain', label:'Air shelf gain', min:-6, max:6, step:0.1, unit:'dB', dp:1, def:2.5},
  ]},
  {group:'Saturation', fields:[
    {key:'sat_drive_db', label:'Drive', min:0, max:20, step:0.5, unit:'dB', dp:1, def:8},
    {key:'sat_mix', label:'Mix', min:0, max:100, step:1, unit:'%', dp:0, def:12},
  ]},
  {group:'Reverb send', fields:[
    {key:'verb_decay', label:'Decay (RT60)', min:0.2, max:4, step:0.1, unit:'s', dp:1, def:1.8},
    {key:'verb_predelay', label:'Pre-delay', min:0, max:100, step:1, unit:'ms', dp:0, def:35},
    {key:'verb_lowcut', label:'Low cut', min:50, max:500, step:10, unit:'Hz', dp:0, def:200},
    {key:'verb_highcut', label:'High cut', min:2000, max:12000, step:100, unit:'Hz', dp:0, def:6000},
    {key:'verb_send_db', label:'Send level', min:-24, max:0, step:0.5, unit:'dB', dp:1, def:-3},
  ]},
  {group:'Delay send', fields:[
    {key:'delay_time_ms', label:'Time', min:50, max:800, step:5, unit:'ms', dp:0, def:350},
    {key:'delay_feedback', label:'Feedback', min:0, max:80, step:1, unit:'%', dp:0, def:13},
    {key:'delay_lowcut', label:'Low cut', min:50, max:1000, step:10, unit:'Hz', dp:0, def:300},
    {key:'delay_highcut', label:'High cut', min:1000, max:8000, step:100, unit:'Hz', dp:0, def:4000},
    {key:'delay_send_db', label:'Send level', min:-40, max:0, step:0.5, unit:'dB', dp:1, def:-20},
  ]},
];
const PRO_CHAIN_IDS = [];
const PRO_CHAIN_VALUES = {};

function fmtProVal(v, f){
  const n=Number(v).toFixed(f.dp);
  return (f.unit==='dB' && v>=0 ? '+':'') + n + (f.unit?' '+f.unit:'');
}
function renderProChainRack(){
  const box=$('proChainRack'); box.innerHTML='';
  PRO_CHAIN_FIELDS.forEach(grp=>{
    const h=document.createElement('div'); h.className='fxgroup'; h.textContent=grp.group;
    box.appendChild(h);
    grp.fields.forEach(f=>{
      PRO_CHAIN_VALUES[f.key]=f.def;
      const id='pc_'+f.key;
      PRO_CHAIN_IDS.push(id);
      const row=document.createElement('div'); row.className='fxrow';
      row.innerHTML=
        '<div style="width:34px"></div>'+
        '<div class="fxname">'+f.label+'</div>'+
        '<input type="range" id="'+id+'" min="'+f.min+'" max="'+f.max+'" step="'+f.step+'" value="'+f.def+'" disabled>'+
        '<div class="val" id="'+id+'v"></div>';
      box.appendChild(row);
      const el=row.querySelector('input'), out=row.querySelector('.val');
      const upd=()=>{ PRO_CHAIN_VALUES[f.key]=+el.value; out.textContent=fmtProVal(el.value,f); liveFxIfPlaying(); };
      el.addEventListener('input',upd); upd();
    });
  });
}
renderProChainRack();

$('proReset').addEventListener('click', async ()=>{
  let defaults=null;
  try{ const r=await fetch('/pro_chain_defaults'); if(r.ok) defaults=await r.json(); }catch(e){}
  PRO_CHAIN_FIELDS.forEach(grp=>grp.fields.forEach(f=>{
    const v=(defaults && defaults[f.key]!=null) ? defaults[f.key] : f.def;
    const el=$('pc_'+f.key); if(!el) return;
    el.value=v; PRO_CHAIN_VALUES[f.key]=+v;
    $('pc_'+f.key+'v').textContent=fmtProVal(v,f);
  }));
  liveFxIfPlaying();
});

function refreshTogs(){
  document.querySelectorAll('.tog').forEach(t=>{
    const k=t.dataset.fx; t.classList.toggle('on', !!fxOn[k]);
    const nm=document.querySelector('.fxname[data-name="'+k+'"]');
    if(nm) nm.classList.toggle('on', !!fxOn[k]);
  });
  // enable/disable sliders per group
  setEnabled('pro', PRO_CHAIN_IDS); setEnabled('denoise',['denoise']);
  ['pstr','pkey','pscale'].forEach(i=>$(i).disabled=!fxOn.pitch);
}
function setEnabled(fx, ids){ ids.forEach(i=>{const el=$(i); if(el) el.disabled=!fxOn[fx];}); }
document.querySelectorAll('.tog').forEach(t=>t.addEventListener('click',()=>{
  const k=t.dataset.fx; fxOn[k]=!fxOn[k]; refreshTogs();
  liveFxIfPlaying();
}));

// live-apply FX to the take while it's playing back (safe no-op otherwise)
function liveFxIfPlaying(){ if(typeof applyLiveFx==='function') applyLiveFx(); }

// ---- slider value labels ----
function bindVal(id, fmt){ const el=$(id),out=$(id+'v'); if(!el||!out)return;
  const upd=()=>{ out.textContent=fmt(el.value); liveFxIfPlaying(); };
  el.addEventListener('input',upd); upd(); }
bindVal('pstr',v=>v+'%'); bindVal('denoise',v=>v+'%');
bindVal('vgain',v=>(v>=0?'+':'')+v+' dB'); bindVal('mgain',v=>(v>=0?'+':'')+v+' dB');
refreshTogs();

// ---- STEP 1: search / load track ----
$('load').addEventListener('click', ()=>loadTrack());
$('url').addEventListener('keydown',e=>{if(e.key==='Enter')loadTrack();});

// tab switching (search <-> paste link)
$('tabSearch').addEventListener('click',()=>switchTab('search'));
$('tabUrl').addEventListener('click',()=>switchTab('url'));
function switchTab(mode){
  const s=mode==='search';
  $('tabSearch').classList.toggle('on',s); $('tabUrl').classList.toggle('on',!s);
  $('searchMode').style.display=s?'':'none'; $('urlMode').style.display=s?'none':'';
}

// ---- search ----
$('searchBtn').addEventListener('click',doSearch);
$('q').addEventListener('keydown',e=>{if(e.key==='Enter')doSearch();});
async function doSearch(){
  const q=$('q').value.trim();
  if(!q){ $('searchStatus').textContent='Type a song to search for.'; return; }
  $('loaderr').textContent=''; $('searchStatus').textContent='Searching YouTube…';
  $('results').innerHTML=''; $('searchBtn').disabled=true;
  let r;
  try{ r=await fetch('/search?q='+encodeURIComponent(q)); }
  catch(e){ $('searchBtn').disabled=false; $('searchStatus').textContent='Could not reach the server.'; return; }
  $('searchBtn').disabled=false;
  const d=await r.json();
  if(!r.ok){ $('searchStatus').textContent=d.error||'Search failed.'; return; }
  const res=d.results||[];
  if(!res.length){ $('searchStatus').textContent='No results. Try different words.'; return; }
  $('searchStatus').textContent=res.length+' results — click one to load it.';
  renderResults(res);
}
function renderResults(res){
  const box=$('results'); box.innerHTML='';
  res.forEach(v=>{
    const card=document.createElement('div'); card.className='rcard';
    const dur=(v.duration!=null)? fmt(v.duration) : '';
    card.innerHTML=
      '<div style="position:relative">'+
        '<img class="rthumb" loading="lazy" src="'+v.thumb+'" alt="">'+
        (dur?'<span class="rdur">'+dur+'</span>':'')+
      '</div>'+
      '<div class="rmeta">'+
        '<div class="rtitle">'+escapeHtml(v.title)+'</div>'+
        (v.channel?'<div class="rchan">'+escapeHtml(v.channel)+'</div>':'')+
      '</div>';
    card.addEventListener('click',()=>{
      document.querySelectorAll('.rcard').forEach(c=>c.classList.remove('loading'));
      card.classList.add('loading');
      loadTrack('https://www.youtube.com/watch?v='+v.id);
    });
    box.appendChild(card);
  });
}
function escapeHtml(s){ return (s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

async function loadTrack(explicitUrl){
  const url=(explicitUrl||$('url').value).trim();
  if(!url){$('loaderr').textContent='Paste a YouTube URL first.';return;}
  $('loaderr').textContent=''; $('load').disabled=true; $('searchBtn').disabled=true;
  $('loadbar').style.display='block'; $('loadstatus').textContent='Downloading backing track…';
  let r;
  try{ r=await fetch('/prepare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})}); }
  catch(e){ return failLoad('Could not reach the local server.'); }
  const d=await r.json();
  if(!r.ok) return failLoad(d.error||'Load failed.');
  SID=d.id;
  poll=setInterval(checkLoad,700);
}
function failLoad(m){ $('load').disabled=false; $('searchBtn').disabled=false;
  $('loaderr').textContent=m; $('loadbar').style.display='none';
  document.querySelectorAll('.rcard.loading').forEach(c=>c.classList.remove('loading')); }

async function checkLoad(){
  let j; try{ j=await(await fetch('/status?id='+SID)).json(); }catch(e){return;}
  if(typeof j.pct==='number') $('loadfill').style.width=j.pct+'%';
  $('loadstatus').textContent={queued:'Queued…',downloading:'Downloading backing track… '+(j.pct||0).toFixed(0)+'%',
    preparing:'Preparing audio…',ready:'Ready'}[j.stage]||'Working…';
  if(j.status==='ready'){ clearInterval(poll); poll=null; onReady(j); }
  else if(j.status==='error'){ clearInterval(poll); poll=null; failLoad(j.error||'Load failed.'); }
}

let VIDEO_ID=null;
function onReady(j){
  $('load').disabled=false; $('searchBtn').disabled=false;
  document.querySelectorAll('.rcard.loading').forEach(c=>c.classList.remove('loading'));
  $('loadstatus').textContent='Loaded: '+(j.title||'track');
  dur=j.duration||0;
  VIDEO_ID=j.video_id||null;
  window._title=j.title||'Karaoke';
  // Try the embedded video for lyrics. If the uploader blocked embedding,
  // we detect it and swap in a lyrics fallback panel (nothing else breaks).
  if(VIDEO_ID){ mountVideo(VIDEO_ID); }
  else { showVideoFallback('No video preview for this link.'); }
  // load clean backing audio (this is what plays + gets recorded)
  backingAudio=new Audio('/backing?id='+SID);
  backingAudio.preload='auto';
  wireBacking(backingAudio);
  decodeBackingBuffer(0);   // decode to an AudioBuffer for sample-accurate take playback
  ['stageCard','fxCard','mixCard'].forEach(c=>$(c).classList.add('active'));
  // enable transport controls
  $('rec').disabled=false; $('render').disabled=false; $('play').disabled=false;
  $('stopmusic').disabled=false;
  $('seek').disabled=false; $('keyup').disabled=false; $('keydn').disabled=false;
  $('keyhint').textContent='original';
  primeMic();
  loadPhoneSyncInfo();
}
function fmt(s){ s=Math.max(0,s|0); return String((s/60)|0).padStart(2,'0')+':'+String(s%60).padStart(2,'0'); }

// ---- video as MASTER CLOCK (YouTube IFrame API) ----------------------------
// The video drives timing. We can control it (play/pause/seek) and read its
// current time, so the clean backing audio can follow it in lockstep — the
// lyrics on screen always match where you're singing.
let ytPlayer=null, ytReady=false, ytApiLoading=false, pendingVid=null;

function loadYTApi(){
  if(window.YT && window.YT.Player){ ytApiLoading=false; if(pendingVid){const v=pendingVid;pendingVid=null;mountVideo(v);} return; }
  if(ytApiLoading) return; ytApiLoading=true;
  const s=document.createElement('script');
  s.src='https://www.youtube.com/iframe_api'; document.head.appendChild(s);
}
window.onYouTubeIframeAPIReady=function(){
  ytApiLoading=false;
  if(pendingVid){ const v=pendingVid; pendingVid=null; mountVideo(v); }
};

function mountVideo(vid){
  if(!(window.YT && window.YT.Player)){ pendingVid=vid; loadYTApi();
    setTimeout(()=>{ if(!ytReady && pendingVid===vid){ pendingVid=null; showVideoFallback('Video player did not load.'); } },6000);
    return;
  }
  $('stage').innerHTML='<div id="ytplayer"></div>'+
    '<button class="novideo" id="novideo" title="No video? Use the lyrics fallback">no video? ↳</button>';
  $('novideo').addEventListener('click',()=>showVideoFallback('Switched to lyrics view.'));
  ytReady=false;
  ytPlayer=new YT.Player('ytplayer',{
    videoId:vid, host:'https://www.youtube.com',
    playerVars:{ mute:1, rel:0, modestbranding:1, playsinline:1,
                 iv_load_policy:3, controls:1, origin:window.location.origin },
    events:{
      onReady:(e)=>{ try{e.target.mute();}catch(_){} ytReady=true; startSyncTicker(); },
      onError:(e)=>showVideoFallback('This uploader blocked embedding (code '+e.data+').'),
      onStateChange:onVideoState,
    },
  });
}

// Video player states: -1 unstarted,0 ended,1 playing,2 paused,3 buffering,5 cued
function onVideoState(e){
  if(!backingAudio) return;
  const st=e.data;
  if(st===YT.PlayerState.PLAYING){
    // match audio to video and play
    syncAudioToVideo(true);
    $('play').textContent='❚❚ Pause music'; $('play').classList.add('amber');
  } else if(st===YT.PlayerState.PAUSED){
    backingAudio.pause();
    if(!recording){ $('play').textContent='▶ Play music'; $('play').classList.remove('amber'); }
  } else if(st===YT.PlayerState.ENDED){
    backingAudio.pause(); if(recording) stopRec(); resetMusicBtn();
  }
}

// ---- sync engine: keep backing audio locked to the video's time ------------
let syncTimer=null;
function videoTime(){ try{ return ytReady? ytPlayer.getCurrentTime():0; }catch(_){ return 0; } }
function videoPlaying(){ try{ return ytReady && ytPlayer.getPlayerState()===YT.PlayerState.PLAYING; }catch(_){ return false; } }

function syncAudioToVideo(startPlaying){
  if(!backingAudio) return;
  const vt=videoTime();
  if(Math.abs((backingAudio.currentTime||0)-vt)>0.12){
    try{ backingAudio.currentTime=vt; }catch(_){}
  }
  if(startPlaying && backingAudio.paused){ backingAudio.play().catch(()=>{}); }
}

function startSyncTicker(){
  if(syncTimer) clearInterval(syncTimer);
  // continuously correct audio drift against the video (the master clock)
  syncTimer=setInterval(()=>{
    if(!backingAudio || !ytReady) return;
    if(seeking) return;
    // during recording/preview the audio is driven explicitly — don't fight it
    if(recording || previewing){
      const vt0=videoTime(); const d0=backingAudio.duration||dur||0;
      $('time').textContent=fmt(vt0)+' / '+fmt(d0);
      if(d0) $('seek').value=Math.round(vt0/d0*1000);
      return;
    }
    const playing=videoPlaying();
    const vt=videoTime();
    // update UI clock/seek from the master (video) time
    const d=backingAudio.duration||dur||0;
    $('time').textContent=fmt(vt)+' / '+fmt(d);
    if(d) $('seek').value=Math.round(vt/d*1000);
    const vc=$('vfclock'); if(vc) vc.textContent=fmt(vt);
    if(playing){
      // drift-correct: if audio has slipped from video by >120ms, snap it
      if(Math.abs((backingAudio.currentTime||0)-vt)>0.12){
        try{ backingAudio.currentTime=vt; }catch(_){}
      }
      if(backingAudio.paused && !recording) backingAudio.play().catch(()=>{});
    } else {
      if(!backingAudio.paused && !recording) backingAudio.pause();
    }
  },250);
}

function showVideoFallback(reason){
  const yurl='https://www.youtube.com/watch?v='+(VIDEO_ID||'');
  $('stage').innerHTML=
    '<div class="vfallback">'+
      '<div class="vf-note">'+(reason||'Video preview unavailable')+'</div>'+
      '<div class="vf-title">'+(window._title||'Karaoke')+'</div>'+
      '<div class="vf-clock" id="vfclock">00:00</div>'+
      '<a class="vf-link" href="'+yurl+'" target="_blank" rel="noopener">Open lyrics video in new tab ↗</a>'+
      '<button class="vf-link" id="tryvideo" style="border:0;cursor:pointer">Try showing video again ↺</button>'+
      '<div class="vf-hint">The clean backing track still plays &amp; records here. '+
        'Keep the lyrics tab visible next to this window.</div>'+
    '</div>';
  const tv=$('tryvideo'); if(tv) tv.addEventListener('click',()=>{ if(VIDEO_ID) mountVideo(VIDEO_ID); });
}

// ---- mic priming: persistent WebAudio graph -------------------------------
// mic ─┬─▶ monitorGain ─▶ destination   (hear your own voice, adjustable)
//      └─▶ analyser                     (level meter + live pitch detection)
// Music level is controlled directly via backingAudio.volume (simple + robust,
// no createMediaElementSource routing which is fragile across element swaps).
let micSource=null, monitorGain=null, micAnalyser=null;
let musicMonitorVol=1.0;   // 0..1 applied to backingAudio.volume while singing

async function primeMic(){
  if(micSource) return;   // already primed
  try{
    mediaStream=await navigator.mediaDevices.getUserMedia({audio:{
      echoCancellation:false, noiseSuppression:false, autoGainControl:false,
      channelCount:1, sampleRate:48000,
    }});
  }catch(e){ $('loaderr').textContent='Microphone blocked. Allow mic access and reload.'; return; }
  audioCtx=new (window.AudioContext||window.webkitAudioContext)({sampleRate:48000, latencyHint:'interactive'});

  // seed the vocal-timing calibration from the browser's reported latency
  // (base + output). It's an estimate; the slider lets you fine-tune by ear.
  try{
    const est=Math.round(((audioCtx.baseLatency||0)+(audioCtx.outputLatency||0))*1000)+80;
    if(est>0 && est<400){ vocalDelayMs=est; $('vocalDelay').value=est;
      $('vocalDelayV').textContent=est+'ms'; }
  }catch(_){}

  micSource=audioCtx.createMediaStreamSource(mediaStream);
  micAnalyser=audioCtx.createAnalyser(); micAnalyser.fftSize=2048;
  micSource.connect(micAnalyser);

  // voice monitor (starts muted; slider controls it)
  monitorGain=audioCtx.createGain(); monitorGain.gain.value=0;
  micSource.connect(monitorGain); monitorGain.connect(audioCtx.destination);

  startMeterAndPitch();
}
function applyMusicMonitorVol(){ if(backingAudio) backingAudio.volume=Math.max(0,Math.min(1,musicMonitorVol)); }

// ---- level meter + live pitch detection -----------------------------------
// Autocorrelation pitch detector (robust for voice). Runs every animation
// frame off the mic analyser; drives both the input meter and the tuner.
const A4=440;
function freqToNote(f){
  if(!f||f<=0) return null;
  const midi=Math.round(12*Math.log2(f/A4)+69);
  const name=NOTE[(midi%12+12)%12];
  const ideal=A4*Math.pow(2,(midi-69)/12);
  const cents=Math.round(1200*Math.log2(f/ideal));
  return {midi, name, octave:Math.floor(midi/12)-1, cents};
}
function autoCorrelate(buf, sr){
  let rms=0; for(let i=0;i<buf.length;i++) rms+=buf[i]*buf[i];
  rms=Math.sqrt(rms/buf.length);
  if(rms<0.008) return -1;   // too quiet to be a note
  let r1=0, r2=buf.length-1, thres=0.2;
  for(let i=0;i<buf.length/2;i++){ if(Math.abs(buf[i])<thres){r1=i;break;} }
  for(let i=1;i<buf.length/2;i++){ if(Math.abs(buf[buf.length-i])<thres){r2=buf.length-i;break;} }
  buf=buf.slice(r1,r2); const n=buf.length;
  const c=new Array(n).fill(0);
  for(let lag=0;lag<n;lag++){ for(let i=0;i<n-lag;i++) c[lag]+=buf[i]*buf[i+lag]; }
  let d=0; while(d<n-1 && c[d]>c[d+1]) d++;
  let maxv=-1, maxp=-1;
  for(let i=d;i<n;i++){ if(c[i]>maxv){maxv=c[i]; maxp=i;} }
  let T0=maxp;
  // parabolic interpolation for sub-sample accuracy
  if(T0>0 && T0<n-1){
    const x1=c[T0-1], x2=c[T0], x3=c[T0+1];
    const a=(x1+x3-2*x2)/2, b=(x3-x1)/2;
    if(a) T0=T0-b/(2*a);
  }
  return sr/T0;
}

let pitchBuf=null;
function startMeterAndPitch(){
  pitchBuf=new Float32Array(micAnalyser.fftSize);
  const tbuf=new Uint8Array(micAnalyser.fftSize);
  (function loop(){
    // level meter
    micAnalyser.getByteTimeDomainData(tbuf);
    let peak=0; for(let i=0;i<tbuf.length;i++){const v=Math.abs(tbuf[i]-128)/128; if(v>peak)peak=v;}
    $('mic').style.width=Math.min(100,peak*140)+'%';
    // live pitch (only meaningful while singing; updates the tuner)
    if(recording && tunerOn){
      micAnalyser.getFloatTimeDomainData(pitchBuf);
      const f=autoCorrelate(pitchBuf, audioCtx.sampleRate);
      updateTuner(f>0? freqToNote(f) : null, f);
    }
    rafMeter=requestAnimationFrame(loop);
  })();
}

// ---- tuner (live singing feedback) ----------------------------------------
let tunerOn=true;
$('tunerToggle').addEventListener('change',()=>{ tunerOn=$('tunerToggle').checked;
  if(!tunerOn) showTuner(false); else if(recording) showTuner(true); });
function showTuner(on){ $('tuner').style.display=on?'':'none'; if(!on){ $('tunerNote').textContent='—';
  $('tunerNote').className='tuner-note'; $('tunerNeedle').style.left='50%'; $('tunerMsg').textContent='sing a note…'; } }
let tunerSmooth=null;
function updateTuner(note, freq){
  if(!note){ $('tunerNote').textContent='—'; $('tunerNote').className='tuner-note';
    $('tunerMsg').textContent='listening…'; $('tunerNeedle').style.background='#fff'; return; }
  // smooth the cents a bit so the needle doesn't jitter
  tunerSmooth = tunerSmooth==null? note.cents : Math.round(tunerSmooth*0.5+note.cents*0.5);
  const cents=tunerSmooth;
  $('tunerNote').textContent=note.name+note.octave;
  const inTune=Math.abs(cents)<=12;
  $('tunerNote').className='tuner-note '+(inTune?'good':'off');
  // needle: map -50..+50 cents to 5%..95%
  const pos=Math.max(5,Math.min(95, 50 + cents));
  $('tunerNeedle').style.left=pos+'%';
  $('tunerNeedle').style.background=inTune?'var(--ok)':'var(--amber)';
  $('tunerMsg').textContent = inTune? 'on pitch ✓'
    : (cents>0? 'a little sharp — ease down' : 'a little flat — lift up');
}

// ---- live monitor volume sliders ------------------------------------------
$('musVol').addEventListener('input',()=>{ musicMonitorVol=(+$('musVol').value)/100;
  $('musVolV').textContent=$('musVol').value+'%'; applyMusicMonitorVol(); });
$('monVol').addEventListener('input',()=>{ const v=(+$('monVol').value)/100;
  $('monVolV').textContent = v===0?'off':Math.round(v*100)+'%';
  if(monitorGain) monitorGain.gain.value=v*1.2; });

// vocal-timing calibration: how much later your voice appears in the recording
// (mic input latency). We pull the take earlier by this much to line it up.
let vocalDelayMs=140;
$('vocalDelay').addEventListener('input',()=>{
  vocalDelayMs=+$('vocalDelay').value;
  $('vocalDelayV').textContent=vocalDelayMs+'ms';
  $('delayHint').textContent = vocalDelayMs>0 ? 'pulls vocals '+vocalDelayMs+'ms earlier'
    : (vocalDelayMs<0 ? 'pushes vocals '+(-vocalDelayMs)+'ms later' : 'no shift');
  // if a take is already recorded, re-align it live so you can hear the change
  if(window._rawTakeBuffer){ realignCurrentTake(); }
});

// ---- record UI helper + after-stop choice ---------------------------------
let pendingContinueAt=null;   // set when user picks "continue from here"
function showRecUI(on){
  $('lamp').classList.toggle('on',on);
  $('rec').disabled=on; $('stop').disabled=!on; $('play').disabled=on;
  if(on) $('stopChoice').style.display='none';
}
function offerContinueOrKeep(){
  const at=window._recEndMusicPos||0;
  $('choiceMsg').textContent='Sang up to '+fmt(at)+'. Continue from here, keep it, or redo?';
  $('stopChoice').style.display='flex';
}
$('continueBtn').addEventListener('click',()=>{
  pendingContinueAt=window._recEndMusicPos||0;   // next take begins where this ended
  $('stopChoice').style.display='none';
  startRec();
});
$('keepBtn').addEventListener('click',()=>{ $('stopChoice').style.display='none'; });
$('redoBtn').addEventListener('click',()=>{
  keptBuffer=null; window._lastVocalBuffer=null; window._lastVocalWav=null; window._rawTakeBuffer=null;
  pendingContinueAt=null; $('stopChoice').style.display='none';
  $('playtake').disabled=true; $('takePlayer').style.display='none'; $('playtake').style.display='';
  $('renderstatus').textContent='Cleared. Record again from the start.';
  startRec();
});

// ==== TRANSPORT STATE =======================================================
// backingAudio = the clean music (this is what records + exports).
// We keep a decoded copy of the previously-KEPT vocal (keptBuffer) so punch-in
// can stitch head+tail in the browser before upload.
let musicKey=0;                 // -6..+6 semitones (current soundtrack key)
let keptBuffer=null;            // AudioBuffer of the kept vocal (full length so far)
let seeking=false;

const NOTE=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
function keyLabel(k){ return k===0?'original':(k>0?'+'+k:''+k)+' semitone'+(Math.abs(k)>1?'s':''); }

// ---- free play / pause (commands the VIDEO; audio follows) -----------------
// If the video is master, we drive it and the sync ticker pulls audio along.
// If there's no video (fallback), we drive the audio directly.
$('play').addEventListener('click',toggleMusic);
async function toggleMusic(){
  if(previewing) stopPreview();
  if(ytReady){
    if(videoPlaying()){ ytPlayer.pauseVideo(); }
    else { ytPlayer.playVideo(); }   // onVideoState handles audio + button
    return;
  }
  // no-video fallback: audio is master
  if(!backingAudio) return;
  if(backingAudio.paused){ await backingAudio.play().catch(()=>{});
    $('play').textContent='❚❚ Pause music'; $('play').classList.add('amber'); }
  else { backingAudio.pause();
    $('play').textContent='▶ Play music'; $('play').classList.remove('amber'); }
}
function resetMusicBtn(){ $('play').textContent='▶ Play music'; $('play').classList.remove('amber'); }

// ---- stop music: halt + rewind to the start (video master, audio follows) --
$('stopmusic').addEventListener('click',stopMusic);
function stopMusic(){
  if(previewing) stopPreview();
  if(ytReady){ try{ ytPlayer.pauseVideo(); ytPlayer.seekTo(0,true); }catch(_){} }
  if(backingAudio){ try{ backingAudio.pause(); backingAudio.currentTime=0; }catch(_){} }
  resetMusicBtn();
  const d=backingAudio?.duration||dur||0;
  $('time').textContent='00:00 / '+fmt(d); $('seek').value=0;
}

// ---- seek bar (seeks the VIDEO; audio snaps to it) -------------------------
$('seek').addEventListener('input',()=>{ seeking=true;
  const d=backingAudio?.duration||dur||0;
  $('time').textContent=fmt($('seek').value/1000*d)+' / '+fmt(d);
});
$('seek').addEventListener('change',()=>{
  const d=backingAudio?.duration||dur||0;
  const t=$('seek').value/1000*d;
  if(ytReady){ try{ ytPlayer.seekTo(t,true); }catch(_){} }
  if(backingAudio){ try{ backingAudio.currentTime=t; }catch(_){} }
  seeking=false;
});

// ---- key transpose (live) --------------------------------------------------
$('keyup').addEventListener('click',()=>changeKey(+1));
$('keydn').addEventListener('click',()=>changeKey(-1));
async function changeKey(delta){
  const nk=Math.max(-6,Math.min(6,musicKey+delta));
  if(nk===musicKey) return;
  musicKey=nk; $('keyval').textContent=(musicKey>0?'+':'')+musicKey;
  $('keyhint').textContent=keyLabel(musicKey);
  // swap the backing source to the transposed render, keeping position + play state.
  // Position comes from the video (master clock) so audio realigns to the lyrics.
  const wasPlaying = ytReady ? videoPlaying() : (backingAudio && !backingAudio.paused);
  const pos = ytReady ? videoTime() : (backingAudio? backingAudio.currentTime : 0);
  $('keyhint').textContent='rendering '+keyLabel(musicKey)+'…';
  const src='/backing?id='+SID+'&key='+musicKey;
  await swapBacking(src,pos,wasPlaying);
  backingBuffer=null; decodeBackingBuffer(musicKey);   // re-decode buffer for take playback in new key
  $('keyhint').textContent=keyLabel(musicKey);
}
function swapBacking(src,pos,play){
  return new Promise(res=>{
    const old=backingAudio;
    const a=new Audio(src); a.preload='auto';
    a.addEventListener('canplay',async()=>{
      try{ a.currentTime=pos; }catch(_){}
      wireBacking(a);
      if(old){ old.pause(); }
      backingAudio=a;
      if(play){ await a.play().catch(()=>{}); }
      res();
    },{once:true});
    a.addEventListener('error',()=>{ res(); },{once:true});
  });
}
function wireBacking(a){
  try{ a.volume=Math.max(0,Math.min(1,musicMonitorVol)); }catch(_){}
  // When there's a video, the sync ticker drives the UI clock/seek from the
  // video (master). Without a video, the audio drives the UI directly.
  a.addEventListener('timeupdate',()=>{
    if(ytReady) return;               // video ticker owns the UI in that case
    const d=a.duration||dur||0;
    $('time').textContent=fmt(a.currentTime)+' / '+fmt(d);
    const vc=$('vfclock'); if(vc) vc.textContent=fmt(a.currentTime);
    if(!seeking && d) $('seek').value=Math.round(a.currentTime/d*1000);
  });
  a.addEventListener('ended',()=>{ if(recording) stopRec(); resetMusicBtn(); });
}

// ---- phone pairing / remote camera sync -------------------------------------
// Pairs a phone (opened on the same WiFi) so clicking Record also remote-
// triggers the phone's camera. A synthetic "chirp" tone played through the
// speakers at the exact moment recording starts lets the server later
// auto-align the phone's video to this app's finished audio mix — see
// audio_fx.py's align_and_mux_video for the matched-filter side of this.
let phonePaired=false, phoneStatus='unpaired';
let chirpParams={f0:800,f1:3200,durMs:120};
window._chirpMusicPos=null;      // song-time when the chirp actually played
window._combineTriggered=false;

async function loadPhoneSyncInfo(){
  try{
    const j=await(await fetch('/phone_sync_info?id='+SID)).json();
    if(j.chirp) chirpParams=j.chirp;
    const box=$('phoneUrlBox'), txt=$('phoneUrlText'), status=$('phoneStatusText');
    if(j.phone_url){
      txt.textContent=j.phone_url;
      box.style.display='';
      status.textContent='Pair your phone:';
    }else{
      status.textContent="Couldn't detect a LAN address — make sure you're on WiFi.";
    }
  }catch(_){
    $('phoneStatusText').textContent='Phone pairing unavailable.';
  }
}
if($('phoneCopyBtn')) $('phoneCopyBtn').addEventListener('click',()=>{
  navigator.clipboard?.writeText($('phoneUrlText').textContent).catch(()=>{});
});

function playSyncChirp(){
  const {f0,f1,durMs}=chirpParams, dur=durMs/1000, t0=audioCtx.currentTime;
  const osc=audioCtx.createOscillator(), gain=audioCtx.createGain();
  osc.type='sine';
  osc.frequency.setValueAtTime(f0,t0);
  osc.frequency.linearRampToValueAtTime(f1,t0+dur);
  gain.gain.setValueAtTime(0,t0);
  gain.gain.linearRampToValueAtTime(0.6,t0+dur*0.1);
  gain.gain.setValueAtTime(0.6,t0+dur*0.9);
  gain.gain.linearRampToValueAtTime(0,t0+dur);
  osc.connect(gain).connect(audioCtx.destination);   // to speakers, NOT into the mic graph
  osc.start(t0); osc.stop(t0+dur+0.02);
}

function checkPhoneStatus(){
  fetch('/status?id='+SID).then(r=>r.json()).then(j=>{
    phonePaired=!!j.phone_paired;
    phoneStatus=j.phone_status||'unpaired';
    const status=$('phoneStatusText');
    if(!status) return;
    if(!phonePaired){
      if($('phoneUrlBox').style.display!=='none') status.textContent='Pair your phone:';
    }else{
      const labels={paired:'📱 Phone connected — ready',recording:'📱 Recording…',
        stopping:'📱 Stopping…',uploaded:'📱 Video received ✓'};
      status.textContent=labels[phoneStatus]||('📱 '+phoneStatus);
    }
  }).catch(()=>{});
}
setInterval(checkPhoneStatus,800);

// ---- record (with optional punch-in) ---------------------------------------
$('rec').addEventListener('click',startRec);
$('stop').addEventListener('click',stopRec);
document.addEventListener('keydown',e=>{
  if(/(INPUT|SELECT|TEXTAREA)/.test(e.target.tagName)) return;
  if(e.code==='Space'){ e.preventDefault(); recording?stopRec():(!$('rec').disabled&&startRec()); }
  if(e.key==='p'||e.key==='P'){ e.preventDefault(); if(!$('play').disabled) toggleMusic(); }
});

let punchAt=0;   // song-time where the current take began (0 unless punch-in)

function pickMime(){
  for(const m of ['audio/webm;codecs=opus','audio/webm','audio/mp4','audio/ogg']){
    if(MediaRecorder.isTypeSupported(m)) return m;
  } return '';
}

// timing anchors used to align the recorded vocal with the music (fix lag):
//   recStartCtxTime = audioCtx.currentTime when MediaRecorder actually fires
//   musicPosAtRecStart = song time the backing was at, at that same instant
let recStartCtxTime=0, musicPosAtRecStart=0;

async function startRec(){
  if(previewing) stopPreview();
  if(!mediaStream){ await primeMic(); if(!mediaStream) return; }
  if(audioCtx.state==='suspended') await audioCtx.resume();

  const punch = pendingContinueAt!=null && keptBuffer;
  // punch-in begins at the chosen continue point (or the playhead); fresh take at 0
  punchAt = punch ? (pendingContinueAt!=null ? pendingContinueAt
                     : (ytReady? videoTime() : (backingAudio.currentTime||0))) : 0;
  pendingContinueAt=null;

  recChunks=[];
  recorder=new MediaRecorder(mediaStream,{mimeType:pickMime(),audioBitsPerSecond:512000});
  recorder.ondataavailable=e=>{ if(e.data.size) recChunks.push(e.data); };
  recorder.onstop=finishRec;
  window._chirpMusicPos=null;
  window._combineTriggered=false;

  const startAt = punch ? punchAt : 0;
  applyMusicMonitorVol();
  try{ backingAudio.currentTime=startAt; }catch(_){}
  if(ytReady){
    try{ ytPlayer.seekTo(startAt,true); ytPlayer.playVideo(); }catch(_){}
    await new Promise(r=>setTimeout(r,220));   // let the video actually spin up
    try{ backingAudio.currentTime=videoTime(); }catch(_){}
  }
  await backingAudio.play().catch(()=>{});

  // Start the recorder, then capture the PRECISE alignment anchors the instant
  // it begins. onstart fires when capture truly starts — best reference we have.
  const phoneThisTake=phonePaired && phoneStatus!=='recording' && !punch;  // v1: fresh takes only
  window._phoneThisTake=phoneThisTake;
  recorder.onstart=()=>{
    recStartCtxTime=audioCtx.currentTime;
    musicPosAtRecStart=backingAudio.currentTime||startAt;
    if(phoneThisTake){
      fetch('/phone/cmd',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({id:SID,cmd:'start'})}).catch(()=>{});
      // give the phone's MediaRecorder time to actually spin up before the
      // chirp fires — network trigger timing is sloppy, so this buffer just
      // needs to make it LIKELY the phone is already capturing; the chirp
      // itself is what gets matched precisely afterward, not this timing.
      setTimeout(()=>{
        if(!recording) return;   // stopped already (very short take) — skip the chirp
        playSyncChirp();
        window._chirpMusicPos=backingAudio.currentTime||startAt;
      },450);
    }
  };
  recorder.start();

  $('play').textContent='❚❚ Pause music'; $('play').classList.add('amber');
  recording=true;
  showRecUI(true);
  if(tunerOn) showTuner(true);
}

function stopRec(){ if(!recording)return;
  // remember how long we actually sang (song time) so we can trim precisely
  window._recEndMusicPos = backingAudio.currentTime || 0;
  recorder.stop(); backingAudio.pause();
  if(ytReady){ try{ ytPlayer.pauseVideo(); }catch(_){} }
  if(window._phoneThisTake){
    fetch('/phone/cmd',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({id:SID,cmd:'stop'})}).catch(()=>{});
  }
  recording=false; showRecUI(false); showTuner(false);
  $('play').disabled=false; resetMusicBtn();
}

async function finishRec(){
  const blob=new Blob(recChunks,{type:recorder.mimeType||'audio/webm'});
  const arr=await blob.arrayBuffer();
  const ctx=new (window.AudioContext||window.webkitAudioContext)({sampleRate:48000});
  const rawTake=await ctx.decodeAudioData(arr);

  // stash the RAW take + the alignment context so the vocal-timing slider can
  // re-align live without re-recording.
  window._rawTakeBuffer=rawTake;
  window._rawTakeMeta={ intendedStart:punchAt, musicPosAtRecStart, keptAtRecord:keptBuffer };

  applyAlignmentAndFinish(ctx);
  offerContinueOrKeep();
  if($('takePlayer').style.display==='block'){ buildTakeAudio(); }
  else { $('playtake').style.display=''; }
}

// Align the raw take using the current vocal-timing calibration, then commit it
// as the playable/renderable vocal. Re-runnable when the slider changes.
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
  $('playtake').disabled=false;
}

// called live when the vocal-timing slider moves
function realignCurrentTake(){
  const wasOpen=$('takePlayer').style.display==='block';
  const wasPlaying = tp && tp.playing;
  const pos = tp? takePos() : 0;
  applyAlignmentAndFinish();
  if(wasOpen){ buildTakeAudio(); tp.pos=pos;
    if(wasPlaying){ startTakePlayback(pos); } }
}

// Align a take so its sample 0 corresponds to the intended song start time.
//
// `offset` = musicPosAtRecStart - intendedStart, i.e. how far into the song the
// music already was when capture began. If offset>0 the recorder started LATE,
// so the take's sample 0 sits EARLIER (in song time) than intended — we must
// PREPEND `offset` seconds of silence to push the vocal to its correct place.
// If offset<0 the recorder started early — TRIM that leading slice.
function alignTake(ctx, take, offset){
  const sr=take.sampleRate, ch=take.getChannelData(0);
  const shift=Math.round(offset*sr);
  let out;
  if(shift>0){
    out=new Float32Array(ch.length+shift);         // pad leading silence
    out.set(ch, shift);
  } else if(shift<0){
    out=ch.subarray(Math.min(-shift,ch.length));   // trim leading
  } else { out=ch; }
  const copy=(out.slice? out.slice(0): Float32Array.from(out));
  const buf=ctx.createBuffer(1,Math.max(1,copy.length),sr);
  buf.copyToChannel(copy,0);
  return buf;
}

// equal-power crossfade stitch, mirrors the server-side stitch_vocals()
function stitchBuffers(ctx,head,tail,punchSec){
  const sr=head.sampleRate;
  const H=head.getChannelData(0), T=tail.getChannelData(0);
  const punch=Math.min(Math.round(punchSec*sr),H.length);
  const xf=Math.max(1,Math.round(0.04*sr));
  const outLen=punch + T.length;         // tail plays from punch onward
  const out=new Float32Array(Math.max(outLen, punch));
  // head 0..punch-xf verbatim
  for(let i=0;i<punch-xf && i<H.length;i++) out[i]=H[i];
  // crossfade region
  for(let i=0;i<xf;i++){
    const hi=punch-xf+i;
    const fo=Math.cos(i/xf*Math.PI/2), fi=Math.sin(i/xf*Math.PI/2);
    const hv=(hi>=0&&hi<H.length)?H[hi]:0;
    const tv=(i<T.length)?T[i]:0;
    out[hi]=hv*fo+tv*fi;
  }
  // rest of tail after crossfade
  for(let i=xf;i<T.length;i++){ out[punch-xf+i]=T[i]; }
  const buf=ctx.createBuffer(1,out.length,sr);
  buf.copyToChannel(out,0);
  return buf;
}

// ---- preview: hear your kept take over the backing track -------------------
// ---- reusable mini audio player with play/pause/stop/seek/time -------------
// Buffer-based player: no <audio> element. Position comes from opts.getPos()/
// getDur(); play/pause/stop/seek call opts hooks. Used for take playback where
// vocal+music are AudioBufferSources kept in perfect sample-sync.
// Min/max peaks per column, computed ONCE per buffer (scanning raw samples
// every animation frame during playback would jank on longer takes).
function computeWavePeaks(buffer, cols){
  const data=buffer.getChannelData(0);
  const perCol=data.length/cols;
  const peaks=new Float32Array(cols*2);
  for(let x=0;x<cols;x++){
    const start=Math.floor(x*perCol), end=Math.max(start+1,Math.floor((x+1)*perCol));
    let min=1, max=-1;
    for(let i=start;i<end && i<data.length;i++){ const v=data[i]; if(v<min)min=v; if(v>max)max=v; }
    if(min>max){ min=0; max=0; }
    peaks[x*2]=min; peaks[x*2+1]=max;
  }
  return peaks;
}

// Cheap per-frame redraw from cached peaks, plus an optional playhead line.
function drawWaveform(canvas, peaks, playheadFrac){
  const dpr=window.devicePixelRatio||1;
  const w=canvas.clientWidth||canvas.width, h=canvas.clientHeight||64;
  const cw=Math.max(1,Math.round(w*dpr)), ch=Math.round(h*dpr);
  if(canvas.width!==cw||canvas.height!==ch){ canvas.width=cw; canvas.height=ch; }
  const ctx=canvas.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  if(!peaks || !peaks.length) return;
  const cols=peaks.length/2, mid=h/2, colW=w/cols;
  ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--amber2').trim()||'#ff9b3d';
  ctx.globalAlpha=.85;
  for(let x=0;x<cols;x++){
    const min=peaks[x*2], max=peaks[x*2+1];
    const y1=mid+min*mid, y2=mid+max*mid;
    ctx.fillRect(x*colW, Math.min(y1,y2), Math.max(1,colW), Math.max(1,Math.abs(y2-y1)));
  }
  ctx.globalAlpha=1;
  if(playheadFrac!=null){
    ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--ink').trim()||'#e8ebf0';
    ctx.fillRect(playheadFrac*w,0,1.5,h);
  }
}

function buildBufferPlayer(container, opts){
  container.innerHTML=
    (opts.label? '<div class="plabel">'+opts.label+'</div>':'')+
    (opts.buffer? '<canvas class="pwave"></canvas>':'')+
    '<div class="ctrls">'+
      '<button class="pbtn play" data-a="play" title="Play/Pause">▶</button>'+
      '<button class="pbtn" data-a="stop" title="Stop (back to start)">■</button>'+
      '<input class="pseek" type="range" min="0" max="1000" value="0" step="1">'+
      '<span class="ptime">00:00 / 00:00</span>'+
    '</div>';
  const btnPlay=container.querySelector('[data-a=play]');
  const btnStop=container.querySelector('[data-a=stop]');
  const seek=container.querySelector('.pseek');
  const time=container.querySelector('.ptime');
  const wave=container.querySelector('.pwave');
  let dragging=false, playing=false;

  const wavePeaks = (wave && opts.buffer) ? computeWavePeaks(opts.buffer, 400) : null;

  if(wave){
    drawWaveform(wave, wavePeaks, 0);
    window.addEventListener('resize', ()=>drawWaveform(wave, wavePeaks, playheadFrac()));
    wave.addEventListener('click',(e)=>{
      const d=opts.getDur()||0; if(!d) return;
      const rect=wave.getBoundingClientRect();
      const t=Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width))*d;
      opts.onSeek&&opts.onSeek(t); paint();
    });
  }

  function playheadFrac(){
    const d=opts.getDur()||0; if(!d) return 0;
    return Math.min(opts.getPos()||0, d)/d;
  }

  function paint(){
    const d=opts.getDur()||0, t=Math.min(opts.getPos()||0, d);
    time.textContent=fmt(t)+' / '+fmt(d);
    if(!dragging && d) seek.value=Math.round(t/d*1000);
    playing = (typeof tp!=='undefined' && tp)? tp.playing : playing;
    btnPlay.textContent=playing?'❚❚':'▶'; btnPlay.classList.toggle('play',!playing);
    if(wave) drawWaveform(wave, wavePeaks, playheadFrac());
  }
  btnPlay.addEventListener('click',()=>{
    if(tp && tp.playing){ opts.onPause&&opts.onPause(); }
    else { opts.onPlay&&opts.onPlay(); }
    paint();
  });
  btnStop.addEventListener('click',()=>{ opts.onStop&&opts.onStop(); paint(); });
  seek.addEventListener('input',()=>{ dragging=true;
    const d=opts.getDur()||0; time.textContent=fmt(seek.value/1000*d)+' / '+fmt(d); });
  seek.addEventListener('change',()=>{
    const d=opts.getDur()||0; const t=seek.value/1000*d;
    opts.onSeek&&opts.onSeek(t); dragging=false; paint(); });
  paint();
  return {paint};
}

// container: element to render into. audio: an HTMLAudioElement (seekable).
// onPlay/onStop: optional hooks (e.g. to drive the muted lyrics video alongside).
function buildPlayer(container, audio, opts={}){
  container.innerHTML=
    (opts.label? '<div class="plabel">'+opts.label+'</div>':'')+
    '<div class="ctrls">'+
      '<button class="pbtn play" data-a="play" title="Play/Pause">▶</button>'+
      '<button class="pbtn" data-a="stop" title="Stop (back to start)">■</button>'+
      '<input class="pseek" type="range" min="0" max="1000" value="0" step="1">'+
      '<span class="ptime">00:00 / 00:00</span>'+
    '</div>';
  const btnPlay=container.querySelector('[data-a=play]');
  const btnStop=container.querySelector('[data-a=stop]');
  const seek=container.querySelector('.pseek');
  const time=container.querySelector('.ptime');
  let dragging=false;

  function paint(){
    const d=audio.duration||0, t=audio.currentTime||0;
    time.textContent=fmt(t)+' / '+fmt(d);
    if(!dragging && d) seek.value=Math.round(t/d*1000);
    btnPlay.textContent=audio.paused?'▶':'❚❚';
    btnPlay.classList.toggle('play',audio.paused);
  }
  btnPlay.addEventListener('click',async()=>{
    if(audio.paused){ if(opts.onPlay)opts.onPlay(audio.currentTime); await audio.play().catch(()=>{}); }
    else { audio.pause(); if(opts.onPause)opts.onPause(); }
    paint();
  });
  btnStop.addEventListener('click',()=>{
    audio.pause(); try{audio.currentTime=0;}catch(_){}
    if(opts.onStop)opts.onStop(); paint();
  });
  seek.addEventListener('input',()=>{ dragging=true;
    const d=audio.duration||0; time.textContent=fmt(seek.value/1000*d)+' / '+fmt(d); });
  seek.addEventListener('change',()=>{
    const d=audio.duration||0; const t=seek.value/1000*d;
    try{audio.currentTime=t;}catch(_){}
    if(opts.onSeek)opts.onSeek(t); dragging=false; paint();
  });
  audio.addEventListener('timeupdate',paint);
  audio.addEventListener('play',paint);
  audio.addEventListener('pause',paint);
  audio.addEventListener('ended',()=>{ if(opts.onEnded)opts.onEnded(); paint(); });
  audio.addEventListener('loadedmetadata',paint);
  paint();
  return {paint, seekEl:seek, playBtn:btnPlay};
}

// ==== TAKE PLAYBACK — sample-accurate, drift-free ==========================
// Both the vocal take AND the backing music play as AudioBufferSourceNodes in
// ONE AudioContext, started at the SAME audioCtx.currentTime. Buffer sources
// are sample-accurate, so vocal and music can NEVER drift — this eliminates the
// "track way ahead of vocals" lag that element-based playback caused.
let previewing=false;   // takeGraph etc. declared early up top
$('playtake').addEventListener('click',revealTakePlayer);

let backingBuffer=null, backingBufKey=null, tp=null; // tp = live take-playback state

async function decodeBackingBuffer(key){
  try{
    const ac=audioCtx || new (window.AudioContext||window.webkitAudioContext)({sampleRate:48000});
    if(!audioCtx) audioCtx=ac;
    const r=await fetch('/backing?id='+SID+(key?'&key='+key:''));
    const ab=await r.arrayBuffer();
    backingBuffer=await ac.decodeAudioData(ab); backingBufKey=key||0;
  }catch(e){ backingBuffer=null; }
}

function makeImpulse(seconds, decay){
  const sr=audioCtx.sampleRate, n=Math.floor(sr*seconds);
  const buf=audioCtx.createBuffer(2,n,sr);
  for(let c=0;c<2;c++){ const d=buf.getChannelData(c);
    for(let i=0;i<n;i++){ d[i]=(Math.random()*2-1)*Math.pow(1-i/n,decay); } }
  return buf;
}

function revealTakePlayer(){
  if(!window._lastVocalBuffer || recording) return;
  $('playtake').style.display='none';
  $('takePlayer').style.display='block';
  buildTakeAudio();
}

// Build the take player UI. Playback position is tracked by us (offset + ctx time).
function buildTakeAudio(){
  if(testMode){ testMode=false; const p=$('testPanel'); if(p) p.remove(); }
  stopTakePlayback();
  const vocalBuf=window._lastVocalBuffer;
  const total=Math.max(vocalBuf?vocalBuf.duration:0, dur||0);
  tp={ playing:false, startCtx:0, startPos:0, pos:0, dur:total, raf:null, nodes:null };

  takePlayerUI=buildBufferPlayer($('takePlayer'), {
    label:'Your take — vocals locked to the music',
    buffer:vocalBuf,
    getPos:()=>takePos(), getDur:()=>tp.dur,
    onPlay:(t)=>startTakePlayback(t),
    onPause:()=>pauseTakePlayback(),
    onStop:()=>{ stopTakePlayback(); tp.pos=0; },
    onSeek:(t)=>{ const was=tp.playing; pauseTakePlayback(); tp.pos=t; if(was) startTakePlayback(t); },
  });
  // balance mixer (mirrors the export Vocal/Music sliders — one source of truth)
  // + a Test button to enter effects test mode.
  const extra=document.createElement('div'); extra.className='takemix';
  extra.innerHTML=
    '<div class="monrow"><span class="monlbl">Vocal</span>'+
      '<input type="range" id="tpVocal" min="-12" max="12" value="'+(+$('vgain').value)+'">'+
      '<span class="monval" id="tpVocalV">'+fmtDb($('vgain').value)+'</span></div>'+
    '<div class="monrow"><span class="monlbl">Music</span>'+
      '<input type="range" id="tpMusic" min="-18" max="6" value="'+(+$('mgain').value)+'">'+
      '<span class="monval" id="tpMusicV">'+fmtDb($('mgain').value)+'</span></div>'+
    '<button class="btn" id="testModeBtn" title="Loop a section and audition each effect live">🎛 Test effects</button>';
  $('takePlayer').appendChild(extra);
  // re-record controls: from the scrubbed point, or sing-along continuation
  const rec=document.createElement('div'); rec.className='takemix'; rec.style.marginTop='8px';
  rec.innerHTML=
    '<span class="monlbl">Re-record</span>'+
    '<button class="btn" id="recFromHere" title="Keep the song up to the playhead, re-sing from there">● From playhead</button>'+
    '<button class="btn amber" id="singAlong" title="Play your take and sing along; it auto-records when the recorded part runs out">▶ Sing-along + continue</button>'+
    '<span class="tp-note" id="recFromNote"></span>';
  $('takePlayer').appendChild(rec);
  // link the mini-mixer to the real export sliders (keeps them in sync both ways)
  $('tpVocal').addEventListener('input',()=>{ $('vgain').value=$('tpVocal').value;
    $('tpVocalV').textContent=fmtDb($('tpVocal').value); $('vgainv').textContent=fmtDb($('vgain').value); applyLiveFx(); });
  $('tpMusic').addEventListener('input',()=>{ $('mgain').value=$('tpMusic').value;
    $('tpMusicV').textContent=fmtDb($('tpMusic').value); $('mgainv').textContent=fmtDb($('mgain').value); applyLiveFx(); });
  $('testModeBtn').addEventListener('click',toggleTestMode);
  // record from the current scrub position (keeps 0→pos of the existing take)
  $('recFromHere').addEventListener('click',()=>{
    const at=takePos()||0;
    stopTakePlayback();
    pendingContinueAt=at;                 // startRec uses this as the punch point
    $('takePlayer').style.display='none'; $('playtake').style.display='';
    startRec();
  });
  // sing-along + auto-continue at the gap
  $('singAlong').addEventListener('click',()=>startSingAlong());
}
function fmtDb(v){ return (v>=0?'+':'')+v+' dB'; }

// ==== SING-ALONG + AUTO-CONTINUE ============================================
// Play the recorded take (with music) from the current position so you can sing
// along; when playback reaches the END of the recorded vocal, seamlessly hand
// off to LIVE recording so you continue the song from exactly there.
let singAlongWatch=null;
function startSingAlong(){
  if(!window._lastVocalBuffer) return;
  const vocalEnd=window._lastVocalBuffer.duration;      // where the recorded part ends
  const from=Math.min(takePos()||0, Math.max(0,vocalEnd-0.25));
  $('recFromNote').textContent='singing along… will record at '+fmt(vocalEnd);
  stopTakePlayback();
  tp.pos=from; startTakePlayback(from,false);
  // watch playback; when it crosses the end of the recorded vocal, hand off to rec
  if(singAlongWatch) cancelAnimationFrame(singAlongWatch);
  (function watch(){
    if(!tp.playing){ return; }               // user paused/stopped → abort handoff
    if(takePos()>=vocalEnd-0.08){
      // reached the gap → switch to live recording, continuing from here
      const handoff=vocalEnd;
      stopTakePlayback();
      pendingContinueAt=handoff;             // stitch new recording onto the take
      $('takePlayer').style.display='none'; $('playtake').style.display='';
      $('renderstatus').textContent='Recording your continuation from '+fmt(handoff)+'…';
      startRec();
      return;
    }
    singAlongWatch=requestAnimationFrame(watch);
  })();
}

// ==== EFFECTS TEST MODE =====================================================
// Loop a chosen section of the take and audition FX live. Reverb/EQ/comp/volume
// are exact; pitch-correction is an APPROXIMATE live preview (a constant shift
// toward the key) — the exact per-note correction happens at final render.
let testMode=false, loopA=0, loopB=0;
function toggleTestMode(){ testMode? exitTestMode() : enterTestMode(); }

function enterTestMode(){
  testMode=true;
  const total=tp.dur||0;
  // default loop = first 8s (or whole take if shorter)
  loopA=0; loopB=Math.min(8, total||8);
  $('testModeBtn').classList.add('on'); $('testModeBtn').textContent='✕ Close test';
  const p=document.createElement('div'); p.className='testpanel'; p.id='testPanel';
  p.innerHTML=
    '<h3>Effects test — loop &amp; audition</h3>'+
    '<div class="tp-sub">Drag on the bar to pick a section. It loops so you can toggle effects and hear each change instantly.</div>'+
    '<div class="loopbar" id="loopbar">'+
      '<div class="loopregion" id="loopregion"></div>'+
      '<div class="loopplay" id="loopplay"></div>'+
    '</div>'+
    '<div class="looprow">'+
      '<button class="tp-btn on" id="loopPlayBtn">▶ Loop</button>'+
      '<span class="tp-note" id="loopInfo"></span>'+
    '</div>'+
    '<div class="looprow" id="fxQuick"></div>'+
    '<div class="tp-note" style="margin-top:8px">EQ / compression / de-ess / reverb send / volume preview approximately here. '+
      'Saturation, the delay send, and pitch fix only apply exactly when you render.</div>';
  $('takePlayer').appendChild(p);
  // quick FX toggles that mirror the rack
  const quick=[['pro','Pro chain'],['pitch','Pitch fix'],['denoise','De-noise']];
  const fxq=p.querySelector('#fxQuick');
  quick.forEach(([key,label])=>{
    const b=document.createElement('button'); b.className='tp-btn'+(fxOn[key]?' on':''); b.textContent=label;
    b.addEventListener('click',()=>{ fxOn[key]=!fxOn[key]; b.classList.toggle('on',fxOn[key]);
      refreshTogs(); applyLiveFx(); restartLoopIfPlaying(); });
    fxq.appendChild(b);
  });
  wireLoopBar();
  paintLoop();
  // jump playback into the loop
  tp.pos=loopA; startTakePlayback(loopA, true);
  $('loopPlayBtn').addEventListener('click',()=>{
    if(tp.playing){ pauseTakePlayback(); $('loopPlayBtn').textContent='▶ Loop'; $('loopPlayBtn').classList.remove('on'); }
    else { startTakePlayback(loopA,true); $('loopPlayBtn').textContent='❚❚ Pause'; $('loopPlayBtn').classList.add('on'); }
  });
}
function exitTestMode(){
  testMode=false;
  $('testModeBtn').classList.remove('on'); $('testModeBtn').textContent='🎛 Test effects';
  const p=$('testPanel'); if(p) p.remove();
  stopTakePlayback(); tp.pos=0;
}
function restartLoopIfPlaying(){ if(testMode && tp.playing){ startTakePlayback(loopA,true); } }

function wireLoopBar(){
  const bar=$('loopbar'); let dragStart=null;
  const posOf=(e)=>{ const r=bar.getBoundingClientRect();
    return Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*(tp.dur||1); };
  bar.addEventListener('mousedown',e=>{ dragStart=posOf(e); loopA=dragStart; loopB=dragStart; paintLoop(); });
  window.addEventListener('mousemove',e=>{ if(dragStart==null) return;
    const cur=posOf(e); loopA=Math.min(dragStart,cur); loopB=Math.max(dragStart,cur); paintLoop(); });
  window.addEventListener('mouseup',()=>{ if(dragStart==null) return; dragStart=null;
    if(loopB-loopA<0.5) loopB=Math.min(tp.dur, loopA+2);   // min 2s loop
    paintLoop(); if(testMode) startTakePlayback(loopA,true); });
}
function paintLoop(){
  const d=tp.dur||1; const reg=$('loopregion'); if(!reg) return;
  reg.style.left=(loopA/d*100)+'%'; reg.style.width=((loopB-loopA)/d*100)+'%';
  const info=$('loopInfo'); if(info) info.textContent='looping '+fmt(loopA)+' – '+fmt(loopB);
}
function takePos(){
  if(!tp.playing) return tp.pos;
  let t=tp.startPos+(audioCtx.currentTime-tp.startCtx);
  if(tp.looping && loopB>loopA){ const len=loopB-loopA;
    if(t>=loopB) t=loopA+((t-loopA)%len); }
  return t;
}

function buildTakeGraphNodes(){
  // Approximate live preview of the pro vocal chain (exact version applies at
  // render time via ffmpeg — see audio_fx.build_pro_chain_filter_complex):
  //   src → hp → dip → nasal → tone1 → tone2 → air → deEss → comp → comp2 →
  //   dry/wet(reverb) → vocalGain
  // Saturation and the delay send aren't previewed live (WebAudio has no
  // built-in soft-clipper/parallel-mix primitive worth the complexity here).
  const hp=audioCtx.createBiquadFilter(); hp.type='highpass';
  const dip=audioCtx.createBiquadFilter(); dip.type='peaking';
  const nasal=audioCtx.createBiquadFilter(); nasal.type='peaking';
  const tone1=audioCtx.createBiquadFilter(); tone1.type='peaking';
  const tone2=audioCtx.createBiquadFilter(); tone2.type='peaking';
  const air=audioCtx.createBiquadFilter(); air.type='highshelf';
  const deEss=audioCtx.createBiquadFilter(); deEss.type='highshelf';
  const comp=audioCtx.createDynamicsCompressor();
  const comp2=audioCtx.createDynamicsCompressor();
  const dry=audioCtx.createGain(), wet=audioCtx.createGain();
  const conv=audioCtx.createConvolver(); conv.buffer=makeImpulse(2.2,2.0);
  const vocalGain=audioCtx.createGain();
  hp.connect(dip); dip.connect(nasal); nasal.connect(tone1); tone1.connect(tone2);
  tone2.connect(air); air.connect(deEss); deEss.connect(comp); comp.connect(comp2);
  comp2.connect(dry); comp2.connect(conv); conv.connect(wet);
  dry.connect(vocalGain); wet.connect(vocalGain); vocalGain.connect(audioCtx.destination);
  const musicGainNode=audioCtx.createGain(); musicGainNode.connect(audioCtx.destination);
  takeGraph={hp,dip,nasal,tone1,tone2,air,deEss,comp,comp2,dry,wet,vocalGain,musicGainNode};
  applyLiveFx();
  return {hp, vocalGain, musicGainNode};
}

async function startTakePlayback(fromT, loopIt){
  if(!window._lastVocalBuffer) return;
  if(audioCtx.state==='suspended') await audioCtx.resume();
  if(!backingBuffer){ await decodeBackingBuffer(musicKey); }
  stopSources();
  const g=buildTakeGraphNodes();
  const looping = !!loopIt && testMode && loopB>loopA;
  const at=looping ? loopA : ((fromT!=null?fromT:tp.pos)||0);

  const vSrc=audioCtx.createBufferSource(); vSrc.buffer=window._lastVocalBuffer;
  vSrc.connect(g.hp);
  let mSrc=null;
  if(backingBuffer){ mSrc=audioCtx.createBufferSource(); mSrc.buffer=backingBuffer;
    mSrc.connect(g.musicGainNode); }

  // rough live pitch preview: shift the whole vocal toward the key centre.
  // (Approximate — the exact per-note correction is done at render.)
  if(fxOn.pitch){ const cents=roughPitchCents(); try{ vSrc.detune.value=cents; }catch(_){} }

  if(looping){
    vSrc.loop=true; vSrc.loopStart=loopA; vSrc.loopEnd=loopB;
    if(mSrc){ mSrc.loop=true; mSrc.loopStart=loopA; mSrc.loopEnd=Math.min(loopB,mSrc.buffer.duration); }
  }

  const startCtx=audioCtx.currentTime+0.06;
  try{ vSrc.start(startCtx, Math.min(at, vSrc.buffer.duration)); }catch(_){}
  if(mSrc){ try{ mSrc.start(startCtx, Math.min(at, mSrc.buffer.duration)); }catch(_){} }

  tp.nodes={vSrc,mSrc}; tp.startCtx=startCtx; tp.startPos=at; tp.playing=true; tp.looping=looping;
  previewing=true; $('rec').disabled=true; $('play').disabled=true;
  applyMusicMonitorVol_graph();

  if(ytReady && !looping){ try{ ytPlayer.seekTo(at,true); ytPlayer.playVideo(); }catch(_){} }

  vSrc.onended=()=>{ if(tp.playing && !tp.looping && takePos()>=tp.dur-0.1){ stopTakePlayback(); tp.pos=0; takePlayerUI&&takePlayerUI.paint(); } };
  tickTake();
}
// approximate live pitch shift (cents) toward the chosen key — preview only
function roughPitchCents(){
  const strength=(+$('pstr').value)/100;
  // a gentle nudge; real correction is per-note at render. Cap the preview shift.
  return 0;   // 0 = no constant detune; per-note preview isn't possible live.
               // We keep pitch preview honest: it shows the key/strength UI is
               // active, but audible correction only appears in the final render.
}
function pauseTakePlayback(){
  if(!tp||!tp.playing) return;
  tp.pos=takePos(); tp.playing=false;
  stopSources();
  if(ytReady){ try{ ytPlayer.pauseVideo(); }catch(_){} }
  cancelAnimationFrame(tp.raf);
  takePlayerUI&&takePlayerUI.paint();
}
function stopTakePlayback(){
  if(tp){ tp.playing=false; if(tp.raf) cancelAnimationFrame(tp.raf); }
  stopSources();
  if(ytReady){ try{ ytPlayer.pauseVideo(); }catch(_){} }
  previewing=false; $('rec').disabled=false; $('play').disabled=false; resetMusicBtn();
  takePlayerUI&&takePlayerUI.paint();
}
function stopSources(){
  if(tp&&tp.nodes){ try{tp.nodes.vSrc&&tp.nodes.vSrc.stop();}catch(_){}
    try{tp.nodes.mSrc&&tp.nodes.mSrc.stop();}catch(_){} tp.nodes=null; }
}
function tickTake(){ if(!tp||!tp.playing) return; takePlayerUI&&takePlayerUI.paint();
  if(testMode){ const lp=$('loopplay'); if(lp && tp.dur){ lp.style.left=(takePos()/tp.dur*100)+'%'; } }
  tp.raf=requestAnimationFrame(tickTake); }
function stopPreview(){ stopTakePlayback(); tp&&(tp.pos=0); }

function applyMusicMonitorVol_graph(){
  // music level during playback = the export Music (mgain) balance in dB,
  // scaled by the monitor slider so you can still ride it in your headphones.
  if(takeGraph&&takeGraph.musicGainNode){
    const mdb=+$('mgain').value;
    takeGraph.musicGainNode.gain.value=Math.pow(10,mdb/20)*Math.max(0,Math.min(1,musicMonitorVol));
  }
}
// Reflect FX-rack + mix settings into the live take graph.
function clampParam(param, min, max, v){ param.value = Math.max(min, Math.min(max, v)); }

// Reflect the pro-chain sliders (+ mix gain) into the live take graph. This is
// an approximation for audition — the exact chain (incl. saturation and the
// delay send) is only applied by ffmpeg at render time.
function applyLiveFx(){
  if(!takeGraph) return;
  const pv=PRO_CHAIN_VALUES, on=!!fxOn.pro;
  const nyq=(audioCtx.sampleRate||48000)/2 - 1;

  if(takeGraph.hp) clampParam(takeGraph.hp.frequency, 10, nyq, on?pv.hpf_freq:10);
  if(takeGraph.dip){ clampParam(takeGraph.dip.frequency,10,nyq,pv.dip_freq); takeGraph.dip.Q.value=pv.dip_q;
    takeGraph.dip.gain.value = on?pv.dip_gain:0; }
  if(takeGraph.nasal){ clampParam(takeGraph.nasal.frequency,10,nyq,pv.nasal_freq); takeGraph.nasal.Q.value=pv.nasal_q;
    takeGraph.nasal.gain.value = on?pv.nasal_gain:0; }
  if(takeGraph.tone1){ clampParam(takeGraph.tone1.frequency,10,nyq,pv.tone1_freq); takeGraph.tone1.Q.value=pv.tone1_q;
    takeGraph.tone1.gain.value = on?pv.tone1_gain:0; }
  if(takeGraph.tone2){ clampParam(takeGraph.tone2.frequency,10,nyq,pv.tone2_freq); takeGraph.tone2.Q.value=pv.tone2_q;
    takeGraph.tone2.gain.value = on?pv.tone2_gain:0; }
  if(takeGraph.air){ clampParam(takeGraph.air.frequency,10,nyq,pv.air_freq);
    takeGraph.air.gain.value = on?pv.air_gain:0; }
  if(takeGraph.deEss){ clampParam(takeGraph.deEss.frequency,10,nyq,pv.deess_freq);
    takeGraph.deEss.gain.value = on ? -(pv.deess_intensity/100)*8 : 0; }
  if(takeGraph.comp){
    clampParam(takeGraph.comp.threshold,-100,0, on?pv.fast_threshold_db:0);
    takeGraph.comp.knee.value=6; clampParam(takeGraph.comp.ratio,1,20, on?pv.fast_ratio:1);
    clampParam(takeGraph.comp.attack,0,1, (on?pv.fast_attack:0)/1000);
    clampParam(takeGraph.comp.release,0,1, (on?pv.fast_release:10)/1000);
  }
  if(takeGraph.comp2){
    clampParam(takeGraph.comp2.threshold,-100,0, on?pv.smooth_threshold_db:0);
    takeGraph.comp2.knee.value=8; clampParam(takeGraph.comp2.ratio,1,20, on?pv.smooth_ratio:1);
    clampParam(takeGraph.comp2.attack,0,1, (on?pv.smooth_attack:0)/1000);
    clampParam(takeGraph.comp2.release,0,1, (on?pv.smooth_release:10)/1000);
  }
  // reverb send: regenerate the impulse if the decay time changed meaningfully
  if(takeGraph.conv){
    const targetDecay=on?pv.verb_decay:0.01;
    if(Math.abs((takeGraph._verbDecay||0)-targetDecay)>0.05){
      takeGraph.conv.buffer=makeImpulse(Math.max(0.05,targetDecay),2.0);
      takeGraph._verbDecay=targetDecay;
    }
  }
  const wetGain = on ? Math.pow(10, pv.verb_send_db/20) : 0;
  if(takeGraph.wet) takeGraph.wet.gain.value=wetGain;
  if(takeGraph.dry) takeGraph.dry.gain.value=1;
  const vdb=+$('vgain').value; if(takeGraph.vocalGain) takeGraph.vocalGain.gain.value=Math.pow(10,vdb/20);
  applyMusicMonitorVol_graph();
}

function encodeWav(audioBuffer){
  const sr=audioBuffer.sampleRate, ch=audioBuffer.getChannelData(0), n=ch.length;
  const buf=new ArrayBuffer(44+n*2), dv=new DataView(buf);
  const wr=(o,s)=>{for(let i=0;i<s.length;i++)dv.setUint8(o+i,s.charCodeAt(i));};
  wr(0,'RIFF'); dv.setUint32(4,36+n*2,true); wr(8,'WAVE'); wr(12,'fmt ');
  dv.setUint32(16,16,true); dv.setUint16(20,1,true); dv.setUint16(22,1,true);
  dv.setUint32(24,sr,true); dv.setUint32(28,sr*2,true); dv.setUint16(32,2,true);
  dv.setUint16(34,16,true); wr(36,'data'); dv.setUint32(40,n*2,true);
  let o=44; for(let i=0;i<n;i++){let s=Math.max(-1,Math.min(1,ch[i])); dv.setInt16(o,s<0?s*0x8000:s*0x7fff,true); o+=2;}
  return new Blob([buf],{type:'audio/wav'});
}

// ---- render ----
$('render').addEventListener('click',doRender);
function collectFx(){
  return {
    pitch: fxOn.pitch?{enabled:true,key:$('pkey').value,scale:$('pscale').value,strength:(+$('pstr').value)/100}:{enabled:false},
    pro: Object.assign({enabled:!!fxOn.pro}, PRO_CHAIN_VALUES),
    denoise: fxOn.denoise?(+$('denoise').value)/100:0,
  };
}
async function doRender(){
  if(!window._lastVocalWav){ $('rendererr').textContent='Record a vocal take first.'; return; }
  $('rendererr').textContent=''; $('result').innerHTML=''; $('render').disabled=true;
  $('renderbar').style.display='block'; $('renderfill').style.width='30%';
  $('renderstatus').textContent='Uploading take…';
  const b64=await blobToB64(window._lastVocalWav);
  const body={id:SID, vocal_wav_b64:b64, fx:collectFx(), mix:{
    vocal_gain_db:+$('vgain').value, music_gain_db:+$('mgain').value,
    music_key:musicKey, format:$('ofmt').value, loudnorm:true },
    chirp_music_pos_sec:window._chirpMusicPos};
  let r; try{ r=await fetch('/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); }
  catch(e){ return failRender('Could not reach server.'); }
  const d=await r.json(); if(!r.ok) return failRender(d.error||'Render failed.');
  $('renderfill').style.width='55%';
  poll=setInterval(checkRender,800);
}
function failRender(m){ $('render').disabled=false; $('rendererr').textContent=m; $('renderbar').style.display='none'; }
async function checkRender(){
  let j; try{ j=await(await fetch('/status?id='+SID)).json(); }catch(e){return;}
  $('renderstatus').textContent={processing_vocal:'Applying voice FX…',mixing:'Mixing final song…',done:'Done'}[j.stage]||'Rendering…';
  if(j.stage==='mixing') $('renderfill').style.width='80%';
  if(j.status==='done'){ clearInterval(poll);poll=null; $('renderfill').style.width='100%';
    $('render').disabled=false; $('renderstatus').textContent='Rendered.';
    // Full in-app playback of the finished mix: play/pause/stop/seek + download.
    // cache-bust so a re-render loads the new file, not a stale cached one.
    $('result').innerHTML=
      '<div class="player" id="finalPlayer"></div>'+
      '<a class="dl" href="/final?id='+SID+'" download>⬇ Download “'+(j.display_name||'song')+'”</a>';
    const fa=new Audio('/final?id='+SID+'&t='+Date.now());
    fa.preload='metadata';
    if(window._finalAudio){ try{window._finalAudio.pause();}catch(_){}}
    window._finalAudio=fa;
    buildPlayer($('finalPlayer'), fa, {label:'Your finished song'});
    if(window._chirpMusicPos!=null) tryCombineVideo(); }
  else if(j.status==='error'){ clearInterval(poll);poll=null; failRender(j.error||'Render failed.'); }
}

// ---- phone video combine (align phone's video to the finished mix) --------
function tryCombineVideo(){
  if(window._combineTriggered) return;
  fetch('/combine_video',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:SID})}).then(r=>{
      if(r.ok){ window._combineTriggered=true; videoPoll=setInterval(checkVideoCombine,1000); }
      // 409 = phone hasn't uploaded yet; retry shortly
      else setTimeout(tryCombineVideo,2000);
    }).catch(()=>setTimeout(tryCombineVideo,2000));
}
let videoPoll=null;
async function checkVideoCombine(){
  let j; try{ j=await(await fetch('/status?id='+SID)).json(); }catch(e){return;}
  if(j.video_status==='done'){
    clearInterval(videoPoll); videoPoll=null;
    $('result').insertAdjacentHTML('beforeend',
      '<a class="dl" href="/final_video?id='+SID+'" download>⬇ Download video (with phone camera)</a>');
  }else if(j.video_status==='error'){
    clearInterval(videoPoll); videoPoll=null;
    $('result').insertAdjacentHTML('beforeend',
      '<div class="err">Couldn\'t sync phone video: '+(j.video_error||'unknown error')+'</div>');
  }
}
function blobToB64(blob){ return new Promise(res=>{const fr=new FileReader();
  fr.onload=()=>res(fr.result.split(',')[1]); fr.readAsDataURL(blob);}); }

// Reopen a previously-prepared session directly: /?sid=XXXX
// (runs LAST, after every declaration exists, to avoid temporal-dead-zone errors)
(function autoload(){
  const sid=new URLSearchParams(location.search).get('sid');
  if(!sid) return;
  SID=sid;
  fetch('/status?id='+sid).then(r=>r.json()).then(j=>{
    if(j && (j.status==='ready'||j.stage==='ready')) onReady(j);
  }).catch(()=>{});
})();
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
    lan = _lan_ip()
    if lan:
        print(f"  Phone-camera pairing (same WiFi): http://{lan}:{PORT}/phone\n")
    print(f"  Party mode: open http://{HOST}:{PORT}/party on this machine (TV screen)")
    if lan:
        print(f"              guests join at http://{lan}:{PORT}/join (once you start a room)\n")
    print(f"  Sessions saved in: {SESS_DIR}")
    print("  Ctrl+C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
