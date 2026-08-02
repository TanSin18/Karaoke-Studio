#!/bin/bash
# Double-click this file in Finder to launch Karaoke Studio.
# Starts the local server (if not already running) and opens your browser.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

PORT=8770
URL="http://localhost:$PORT"

echo ""
echo "  🎤  Karaoke Studio"
echo "  ------------------"

# 1. Make sure the tools we need are present.
missing=""
for tool in python3 yt-dlp ffmpeg; do
  command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ]; then
  echo "  Missing required tools:$missing"
  echo "  Install with:  brew install$missing"
  echo ""
  read -n 1 -s -r -p "  Press any key to close."
  exit 1
fi

# 2. If something is already serving on the port, just open the browser.
if lsof -ti tcp:$PORT >/dev/null 2>&1; then
  echo "  Already running — opening $URL"
  open "$URL"
  exit 0
fi

# 3. Start the server in the background, wait until it answers, open browser.
echo "  Starting server on $URL ..."
python3 "$DIR/studio.py" >/tmp/karaoke-studio.log 2>&1 &
SERVER_PID=$!

# wait up to ~10s for the server to come up
for i in $(seq 1 40); do
  if curl -s -o /dev/null "$URL"; then
    break
  fi
  sleep 0.25
done

if curl -s -o /dev/null "$URL"; then
  echo "  Ready. Opening your browser."
  open "$URL"
  echo ""
  echo "  The server is running in this window."
  echo "  KEEP THIS WINDOW OPEN while you use the app."
  echo "  Close it (or press Ctrl+C) to stop the studio."
  echo ""
  # keep the launcher attached to the server so closing the window stops it
  wait $SERVER_PID
else
  echo "  Server did not start. See /tmp/karaoke-studio.log for details."
  echo ""
  read -n 1 -s -r -p "  Press any key to close."
  exit 1
fi
