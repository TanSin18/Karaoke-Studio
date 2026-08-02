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
import re
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import audio_fx

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "index.html")


def _read_index():
    with open(INDEX_PATH, "rb") as fh:
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

# NOTE: bind to "localhost", not "127.0.0.1". YouTube's embedded player rejects
# embeds whose origin is a raw IP (Error 153 / "Video unavailable") but accepts
# "localhost". Same machine-only binding, but the video actually loads.
HOST, PORT = "localhost", int(os.environ.get("KARAOKE_PORT", 8770))

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
              punch_sec=None, source_take_ids=None, tid=None):
    tid = tid or uuid.uuid4().hex[:8]
    meta = _read_meta(jid)
    takes = meta.get("takes") or []
    take = {
        "id": tid, "file": file, "duration": duration, "kind": kind,
        "fx_snapshot": fx_snapshot, "punch_sec": punch_sec,
        "source_take_ids": source_take_ids, "deleted": False,
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
        set_job(jid, status="ready", stage="ready", pct=100,
                title="Karaoke", duration=_probe_duration(backing), recovered=True)
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

def do_prepare(jid, url):
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

    set_job(jid, status="ready", stage="ready", pct=100,
            video_id=vid, title=title or "Karaoke", duration=_probe_duration(backing))


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


def _run_render_locked(jid, vocal_path, fx, mix_opts, lock):
    """Thread target for /render: holds `lock` for the whole render so a
    second request for the same session can't start until this one is done
    (the lock itself was already acquired by the HTTP handler)."""
    try:
        do_render(jid, vocal_path, fx, mix_opts)
    finally:
        lock.release()


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
            jid = uuid.uuid4().hex[:12]
            set_job(jid, status="queued")
            threading.Thread(target=do_prepare, args=(jid, url), daemon=True).start()
            self._json(200, {"id": jid})
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

        self._json(404, {"error": "not found"})



def main():
    if not os.path.isfile(INDEX_PATH):
        raise SystemExit(
            f"Fatal: index.html not found at {INDEX_PATH}\n"
            "It must sit next to studio.py — check your install/bundle."
        )
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\n  🎤 Karaoke Studio  →  http://{HOST}:{PORT}\n")
    print(f"  Sessions saved in: {SESS_DIR}")
    print("  Ctrl+C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
