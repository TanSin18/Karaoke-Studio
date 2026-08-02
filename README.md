# 🎤 Karaoke Studio

A local karaoke recording studio that runs in your browser. Search or paste a
YouTube song, sing along to the video's lyrics, record your vocals, apply studio
effects, and export a finished mix — all on your own machine.

![status](https://img.shields.io/badge/runs-locally-4ade80)
![python](https://img.shields.io/badge/python-3.9%2B-6ea8fe)

---

## What it does

- **Find a song** — search YouTube in-app or paste a link.
- **Lyrics video, in sync** — the video plays as the master clock; the clean
  backing audio follows it locked, so the lyrics always match where you're singing.
- **Record your vocals** with a live **pitch tuner** (shows the note you're hitting)
  and an adjustable **voice monitor** (hear yourself in your headphones).
- **Transpose the key** of the backing track (−6 … +6 semitones) to fit your range.
- **Studio-smooth vocal chain** — compression (evens out loud pushes), de-ess,
  de-harsh, EQ, reverb/echo, and Auto-Tune-style **pitch correction**.
- **Test mode** — loop a section and audition each effect live before committing.
- **Punch-in / continuation** — re-sing from any point, or sing along to your take
  and have it auto-record the rest.
- **Export** a finished mix as WAV (24-bit), FLAC, or MP3 320k.

Everything runs on **your computer**. Nothing is uploaded to any server.

---

## Requirements

You need three command-line tools installed and on your `PATH`:

| Tool      | What it's for                        |
|-----------|--------------------------------------|
| `python3` | runs the local server (3.9+)         |
| `yt-dlp`  | downloads the backing audio          |
| `ffmpeg`  | audio conversion, effects, mixdown   |

Optional but recommended: `rubberband` and `sox` (higher-quality pitch/transpose).

Plus a few Python packages: `numpy`, `scipy`, `librosa`, `soundfile`.

### macOS (Homebrew)

```bash
brew install python yt-dlp ffmpeg rubberband sox
pip3 install numpy scipy librosa soundfile
```

### Linux (Debian/Ubuntu)

```bash
sudo apt install python3 python3-pip ffmpeg rubberband-cli sox
pip3 install yt-dlp numpy scipy librosa soundfile
```

### Windows

Install [Python](https://python.org), [ffmpeg](https://ffmpeg.org/download.html),
and [yt-dlp](https://github.com/yt-dlp/yt-dlp/releases) (put them on your PATH),
then:

```bash
pip install numpy scipy librosa soundfile
```

---

## Run it

```bash
git clone https://github.com/TanSin18/Karaoke-Studio.git
cd Karaoke-Studio
python3 studio.py
```

Then open **http://localhost:8770** in your browser.

> **On macOS** you can also double-click **`Karaoke Studio.command`** in Finder —
> it starts the server and opens your browser in one step. (First time: right-click
> → Open to get past Gatekeeper.)

### Important: use `localhost`, not `127.0.0.1`

Open **`http://localhost:8770`** exactly. YouTube's embedded player rejects
`127.0.0.1` (shows "Video unavailable") but accepts `localhost`.

### Wear headphones

The clean backing track is what gets recorded and exported. If you use speakers,
your mic re-records the music and the final mix gets muddy. Headphones keep your
mic capturing only your voice.

---

## How it works

```
Browser (your UI)                     Python backend (studio.py)
─────────────────                     ──────────────────────────
search / paste URL  ───────────────▶  yt-dlp downloads clean audio
video plays (lyrics, master clock)
mic records vocals (WebAudio)
live pitch tuner + monitor
                    ◀───────────────  serves backing track (+ transposed keys)
adjust effects live (WebAudio graph)
                    ─── take + FX ──▶  ffmpeg + librosa: pitch-correct,
                                       smooth, mix → final song
download final  ◀──────────────────   WAV / FLAC / MP3
```

- `studio.py` — the local web server + orchestration (pure Python stdlib HTTP).
- `audio_fx.py` — the offline effects engine: pitch correction (librosa),
  the ffmpeg filter chain, transpose, stitching, and mixdown.
- `index.html` — the full studio UI (also embedded into `studio.py` at runtime).

---

## Troubleshooting

**"Download failed" / videos won't load** — YouTube changes things often and
`yt-dlp` needs updating. Run `brew upgrade yt-dlp` (or `pip install -U yt-dlp`).
This fixes it ~90% of the time.

**"Video unavailable" in the player** — make sure you opened `localhost:8770`,
not `127.0.0.1`. Some uploaders also disable embedding; the app falls back to a
lyrics view with an "open in new tab" link in that case.

**No sound / mic not working** — allow microphone access when the browser asks,
and check your input device in the OS sound settings.

---

## Notes & limitations

- This is a **local, single-user** tool. It is **not** designed to be hosted as a
  public website — downloading YouTube audio from a shared server gets rate-limited
  and blocked, and would raise terms-of-service issues. Run it on your own machine.
- Only download and record content you have the rights to. Respect YouTube's Terms
  of Service and copyright law in your country.
- Pitch correction previews approximately live; the precise, high-quality version
  is applied when you export.

---

## License

MIT — see [LICENSE](LICENSE).
