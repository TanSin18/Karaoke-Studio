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
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import audio_fx

HERE = os.path.dirname(os.path.abspath(__file__))

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
HOST, PORT = "localhost", 8770

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
            try:
                r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                                    "format=duration", "-of", "csv=p=0", backing],
                                   capture_output=True, text=True)
                dur = float(r.stdout.strip())
            except Exception:
                pass
        set_job(jid, status="ready", stage="ready", pct=100,
                title=meta.get("title") or "Karaoke",
                video_id=meta.get("video_id"),
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

    # duration for the UI
    dur = None
    try:
        r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                            "format=duration", "-of", "csv=p=0", backing],
                           capture_output=True, text=True)
        dur = float(r.stdout.strip())
    except Exception:
        pass

    set_job(jid, status="ready", stage="ready", pct=100,
            video_id=vid, title=title or "Karaoke", duration=dur)
    _write_meta(jid, video_id=vid, title=title or "Karaoke", duration=dur,
                url=url, created_at=time.time())


def list_recent_sessions(limit=8):
    """Sessions with a downloaded backing track, newest first, for the
    'resume a recent song' panel. Survives server restarts (reads meta.json)."""
    out = []
    try:
        entries = os.listdir(SESS_DIR)
    except OSError:
        return out
    for jid in entries:
        sdir = os.path.join(SESS_DIR, jid)
        backing = os.path.join(sdir, "backing.wav")
        if not os.path.isfile(backing):
            continue
        meta = _read_meta(jid)
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

def do_render(jid, vocal_path, fx, mix_opts):
    set_job(jid, status="rendering", stage="processing_vocal", error=None)
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

    try:
        processed_vocal = os.path.join(sdir, "vocal_fx.wav")
        audio_fx.process_vocal(vocal_path, processed_vocal, fx, tmp)

        set_job(jid, stage="mixing")
        out_fmt = mix_opts.get("format", "wav")
        final = os.path.join(sdir, f"final.{out_fmt}")
        audio_fx.mixdown(
            processed_vocal, backing, final,
            vocal_gain_db=float(mix_opts.get("vocal_gain_db", 0)),
            music_gain_db=float(mix_opts.get("music_gain_db", -3)),
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
            b = INDEX_HTML.encode()
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

        if p.path == "/sessions":
            self._json(200, {"sessions": list_recent_sessions()})
            return

        if p.path == "/status":
            jid = (parse_qs(p.query).get("id") or [""])[0]
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

        if p.path == "/render":
            body = self._read_body()
            jid = body.get("id")
            j = job_or_disk(jid)
            if not j:
                self._json(404, {"error": "unknown session"})
                return
            # vocal audio arrives as base64 wav
            b64 = body.get("vocal_wav_b64", "")
            if not b64:
                self._json(400, {"error": "No vocal recording received."})
                return
            sdir = os.path.join(SESS_DIR, jid)
            os.makedirs(sdir, exist_ok=True)
            vocal_path = os.path.join(sdir, "vocal_raw.wav")
            try:
                with open(vocal_path, "wb") as fh:
                    fh.write(base64.b64decode(b64))
            except Exception:
                self._json(400, {"error": "Could not decode recording."})
                return
            fx = body.get("fx", {})
            mix_opts = body.get("mix", {})
            set_job(jid, status="rendering")
            threading.Thread(target=do_render,
                             args=(jid, vocal_path, fx, mix_opts),
                             daemon=True).start()
            self._json(200, {"ok": True})
            return

        self._json(404, {"error": "not found"})


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

    <!-- resume a previously-prepared song, no re-download needed -->
    <div id="recentWrap" style="display:none;margin-bottom:16px">
      <div class="metlabel">RESUME A RECENT SONG</div>
      <div class="results" id="recentResults"></div>
    </div>

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
          <!-- reverb -->
          <div class="fxrow">
            <div class="tog" data-fx="reverb"></div>
            <div class="fxname" data-name="reverb">Reverb</div>
            <input type="range" id="reverb" min="0" max="100" value="35" disabled>
            <div class="val" id="reverbv">35%</div>
          </div>
          <!-- echo -->
          <div class="fxrow">
            <div class="tog" data-fx="echo"></div>
            <div class="fxname" data-name="echo">Echo</div>
            <input type="range" id="echo" min="0" max="100" value="20" disabled>
            <div class="val" id="echov">20%</div>
          </div>
          <!-- eq low/mid/high -->
          <div class="fxrow">
            <div class="tog" data-fx="eq"></div>
            <div class="fxname" data-name="eq">EQ · low</div>
            <input type="range" id="eqlow" min="-12" max="12" value="2" disabled>
            <div class="val" id="eqlowv">+2 dB</div>
          </div>
          <div class="fxrow">
            <div style="width:34px"></div>
            <div class="fxname">EQ · mid</div>
            <input type="range" id="eqmid" min="-12" max="12" value="1" disabled>
            <div class="val" id="eqmidv">+1 dB</div>
          </div>
          <div class="fxrow">
            <div style="width:34px"></div>
            <div class="fxname">EQ · high</div>
            <input type="range" id="eqhigh" min="-12" max="12" value="3" disabled>
            <div class="val" id="eqhighv">+3 dB</div>
          </div>
          <!-- compression -->
          <div class="fxrow">
            <div class="tog" data-fx="compress"></div>
            <div class="fxname" data-name="compress">Compress</div>
            <input type="range" id="compress" min="0" max="100" value="55" disabled>
            <div class="val" id="compressv">55%</div>
          </div>
          <!-- denoise + deess -->
          <div class="fxrow">
            <div class="tog" data-fx="denoise"></div>
            <div class="fxname" data-name="denoise">De-noise</div>
            <input type="range" id="denoise" min="0" max="100" value="40" disabled>
            <div class="val" id="denoisev">40%</div>
          </div>
          <div class="fxrow">
            <div class="tog" data-fx="deess"></div>
            <div class="fxname" data-name="deess">De-ess</div>
            <div style="flex:1;color:var(--dim);font-family:var(--mono);font-size:11px">tame harsh S sounds</div>
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
const fxOn = {pitch:true, reverb:false, echo:false, eq:false, compress:true, denoise:false, deess:true};
function refreshTogs(){
  document.querySelectorAll('.tog').forEach(t=>{
    const k=t.dataset.fx; t.classList.toggle('on', !!fxOn[k]);
    const nm=document.querySelector('.fxname[data-name="'+k+'"]');
    if(nm) nm.classList.toggle('on', !!fxOn[k]);
  });
  // enable/disable sliders per group
  setEnabled('reverb',['reverb']); setEnabled('echo',['echo']);
  setEnabled('compress',['compress']); setEnabled('denoise',['denoise']);
  setEnabled('eq',['eqlow','eqmid','eqhigh']);
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
bindVal('pstr',v=>v+'%'); bindVal('reverb',v=>v+'%'); bindVal('echo',v=>v+'%');
bindVal('compress',v=>v+'%'); bindVal('denoise',v=>v+'%');
bindVal('eqlow',v=>(v>=0?'+':'')+v+' dB'); bindVal('eqmid',v=>(v>=0?'+':'')+v+' dB');
bindVal('eqhigh',v=>(v>=0?'+':'')+v+' dB');
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

// ---- resume a recent song (prepared before, backing track already on disk) --
async function loadRecentSessions(){
  let r; try{ r=await fetch('/sessions'); }catch(e){ return; }
  if(!r.ok) return;
  let d; try{ d=await r.json(); }catch(e){ return; }
  const sessions=d.sessions||[];
  if(!sessions.length) return;
  renderRecentSessions(sessions);
  $('recentWrap').style.display='block';
}
function renderRecentSessions(sessions){
  const box=$('recentResults'); box.innerHTML='';
  sessions.forEach(s=>{
    const card=document.createElement('div'); card.className='rcard';
    const dur=(s.duration!=null)? fmt(s.duration) : '';
    card.innerHTML=
      '<div style="position:relative">'+
        (s.thumb? '<img class="rthumb" loading="lazy" src="'+s.thumb+'" alt="">' : '<div class="rthumb"></div>')+
        (dur?'<span class="rdur">'+dur+'</span>':'')+
      '</div>'+
      '<div class="rmeta">'+
        '<div class="rtitle">'+escapeHtml(s.title)+'</div>'+
      '</div>';
    card.addEventListener('click',()=>{
      document.querySelectorAll('.rcard').forEach(c=>c.classList.remove('loading'));
      card.classList.add('loading');
      resumeSession(s.id);
    });
    box.appendChild(card);
  });
}
function resumeSession(sid){
  SID=sid;
  $('loaderr').textContent='';
  $('loadbar').style.display='block'; $('loadfill').style.width='100%';
  $('loadstatus').textContent='Resuming session…';
  fetch('/status?id='+sid).then(r=>r.json()).then(j=>{
    if(j && (j.status==='ready'||j.stage==='ready')){ onReady(j); }
    else {
      document.querySelectorAll('.rcard.loading').forEach(c=>c.classList.remove('loading'));
      $('loadbar').style.display='none';
      $('loaderr').textContent='That session is no longer available.';
    }
  }).catch(()=>{
    document.querySelectorAll('.rcard.loading').forEach(c=>c.classList.remove('loading'));
    $('loadbar').style.display='none';
    $('loaderr').textContent='Could not reach the local server.';
  });
}
loadRecentSessions();

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
  recorder.onstart=()=>{
    recStartCtxTime=audioCtx.currentTime;
    musicPosAtRecStart=backingAudio.currentTime||startAt;
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
function buildBufferPlayer(container, opts){
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
  let dragging=false, playing=false;
  function paint(){
    const d=opts.getDur()||0, t=Math.min(opts.getPos()||0, d);
    time.textContent=fmt(t)+' / '+fmt(d);
    if(!dragging && d) seek.value=Math.round(t/d*1000);
    playing = (typeof tp!=='undefined' && tp)? tp.playing : playing;
    btnPlay.textContent=playing?'❚❚':'▶'; btnPlay.classList.toggle('play',!playing);
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
    '<div class="tp-note" style="margin-top:8px">Reverb / EQ / compression / volume preview exactly. '+
      'Pitch fix previews approximately here — the precise version is applied when you render.</div>';
  $('takePlayer').appendChild(p);
  // quick FX toggles that mirror the rack
  const quick=[['reverb','Reverb'],['eq','EQ'],['compress','Compress'],['pitch','Pitch fix'],['deess','De-ess'],['denoise','De-noise']];
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
  // Studio-smooth vocal chain:
  //   src → hp → deHarsh → eqLow → eqMid → eqHigh → deEss → comp → dry/wet(reverb) → vocalGain
  // comp smooths the "throw" (loud pushes); deEss/deHarsh tame harsh S & 3kHz bite.
  const hp=audioCtx.createBiquadFilter(); hp.type='highpass'; hp.frequency.value=80; hp.Q.value=0.7;
  const deHarsh=audioCtx.createBiquadFilter(); deHarsh.type='peaking'; deHarsh.frequency.value=3000; deHarsh.Q.value=1.4; deHarsh.gain.value=-2.5;
  const eqLow=audioCtx.createBiquadFilter(); eqLow.type='lowshelf'; eqLow.frequency.value=200;
  const eqMid=audioCtx.createBiquadFilter(); eqMid.type='peaking'; eqMid.frequency.value=1500; eqMid.Q.value=1;
  const eqHigh=audioCtx.createBiquadFilter(); eqHigh.type='highshelf'; eqHigh.frequency.value=6000;
  const deEss=audioCtx.createBiquadFilter(); deEss.type='highshelf'; deEss.frequency.value=7000; deEss.gain.value=-3;
  const comp=audioCtx.createDynamicsCompressor();
  comp.threshold.value=-22; comp.knee.value=26; comp.ratio.value=3.2; comp.attack.value=0.006; comp.release.value=0.18;
  const dry=audioCtx.createGain(), wet=audioCtx.createGain();
  const conv=audioCtx.createConvolver(); conv.buffer=makeImpulse(2.2,2.0);
  const vocalGain=audioCtx.createGain();
  hp.connect(deHarsh); deHarsh.connect(eqLow); eqLow.connect(eqMid); eqMid.connect(eqHigh);
  eqHigh.connect(deEss); deEss.connect(comp);
  comp.connect(dry); comp.connect(conv); conv.connect(wet);
  dry.connect(vocalGain); wet.connect(vocalGain); vocalGain.connect(audioCtx.destination);
  const musicGainNode=audioCtx.createGain(); musicGainNode.connect(audioCtx.destination);
  takeGraph={hp,deHarsh,eqLow,eqMid,eqHigh,deEss,comp,dry,wet,vocalGain,musicGainNode};
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
function applyLiveFx(){
  if(!takeGraph) return;
  const eq=fxOn.eq?{low:+$('eqlow').value,mid:+$('eqmid').value,high:+$('eqhigh').value}:{low:0,mid:0,high:0};
  if(takeGraph.eqLow) takeGraph.eqLow.gain.value=eq.low;
  if(takeGraph.eqMid) takeGraph.eqMid.gain.value=eq.mid;
  if(takeGraph.eqHigh) takeGraph.eqHigh.gain.value=eq.high;
  const rev=fxOn.reverb?(+$('reverb').value)/100:0;
  if(takeGraph.wet) takeGraph.wet.gain.value=rev*0.9;
  if(takeGraph.dry) takeGraph.dry.gain.value=1-rev*0.4;
  const vdb=+$('vgain').value; if(takeGraph.vocalGain) takeGraph.vocalGain.gain.value=Math.pow(10,vdb/20);
  // smoothing chain: compression (throw), de-ess, de-harsh — scale by toggle+amount
  if(takeGraph.comp){
    const c=fxOn.compress?(+$('compress').value)/100:0;
    // more compression → lower threshold + higher ratio → smoother pushes
    takeGraph.comp.threshold.value = -12 - c*24;      // -12..-36 dB
    takeGraph.comp.ratio.value     = 1.5 + c*4;       // 1.5..5.5:1
    takeGraph.comp.knee.value      = 24;
  }
  if(takeGraph.deEss) takeGraph.deEss.gain.value = fxOn.deess ? -6 : -2;   // always a touch, more when on
  if(takeGraph.deHarsh) takeGraph.deHarsh.gain.value = -2.5;               // gentle de-harsh always on
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
    reverb: fxOn.reverb?(+$('reverb').value)/100:0,
    echo: fxOn.echo?(+$('echo').value)/100:0,
    eq: fxOn.eq?{low:+$('eqlow').value,mid:+$('eqmid').value,high:+$('eqhigh').value}:{},
    compress: fxOn.compress?(+$('compress').value)/100:0,
    denoise: fxOn.denoise?(+$('denoise').value)/100:0,
    deess: !!fxOn.deess,
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
    music_key:musicKey, format:$('ofmt').value, loudnorm:true }};
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
    buildPlayer($('finalPlayer'), fa, {label:'Your finished song'}); }
  else if(j.status==='error'){ clearInterval(poll);poll=null; failRender(j.error||'Render failed.'); }
}
function blobToB64(blob){ return new Promise(res=>{const fr=new FileReader();
  fr.onload=()=>res(fr.result.split(',')[1]); fr.readAsDataURL(blob);}); }

// Reopen a previously-prepared session directly: /?sid=XXXX
// (runs LAST, after every declaration exists, to avoid temporal-dead-zone errors)
(function autoload(){
  const sid=new URLSearchParams(location.search).get('sid');
  if(sid) resumeSession(sid);
})();
</script>
"""


def main():
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
