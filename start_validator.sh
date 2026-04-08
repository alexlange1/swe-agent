#!/bin/bash
set -euo pipefail
cd /home/const/subnet66/tau
source /home/const/subnet66/.venv/bin/activate

PIDFILE=/home/const/subnet66/tau/validator.pid
LOGFILE=/home/const/subnet66/tau/validator.log

# Prevent double-start
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Validator already running (pid $(cat "$PIDFILE"))" >&2
  exit 1
fi

export GITHUB_TOKEN=$(doppler secrets get GITHUB_TOKEN -p arbos -c dev --plain)
export OPENROUTER_API_KEY=$(doppler secrets get OPENROUTER_API_KEY -p arbos -c dev --plain)
export CURSOR_API_KEY=$(doppler secrets get CURSOR_API_KEY -p arbos -c dev --plain)
export R2_ACCESS_KEY_ID=$(doppler secrets get R2_ACCESS_KEY_ID -p arbos -c dev --plain)
export R2_SECRET_ACCESS_KEY=$(doppler secrets get R2_SECRET_ACCESS_KEY -p arbos -c dev --plain)
export R2_BUCKET_NAME=$(doppler secrets get R2_BUCKET_NAME -p arbos -c dev --plain)
export R2_BUCKET_URL=$(doppler secrets get R2_BUCKET_URL -p arbos -c dev --plain)
export R2_URL=$(doppler secrets get R2_URL -p arbos -c dev --plain)
export R2_PUBLIC_URL=$(doppler secrets get R2_PUBLIC_URL -p arbos -c dev --plain)
export TMC_API_KEY=$(doppler secrets get TMC_API_KEY -p arbos -c dev --plain)
export PYTHONUNBUFFERED=1

# Kill orphaned containers from prior runs
docker ps -q --filter "name=swe-eval-" | xargs -r docker kill 2>/dev/null || true
docker ps -aq --filter "name=swe-eval-" | xargs -r docker rm -f 2>/dev/null || true

echo "$(date -Iseconds) Starting validator..." >> "$LOGFILE"

trap 'rm -f "$PIDFILE"' EXIT
echo $$ > "$PIDFILE"

exec tau validate \
  --wallet-name sn66_owner \
  --wallet-hotkey default \
  --max-concurrency 48 \
  --task-pool-target 60 \
  --duel-rounds 15 \
  --win-margin 3 \
  >> "$LOGFILE" 2>&1
