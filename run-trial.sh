#!/usr/bin/env bash
# Run one practice trial locally without Docker: start the kit's env server, run this repo's bot
# against it, SIGTERM the env so it exports state.json, then run the kit's scorer.
# Usage: ./run-trial.sh <trial-index 0|1|2> [env-port]
set -u
KIT="$HOME/workspace/makefun/kits/free-model-aggregator-01/free-model-aggregator-01-kit"
REPO="$HOME/workspace/makefun/free-model-aggregator-01"
T="${1:-0}"
PORT="${2:-18080}"
TRIAL="$KIT/practice/trial-$T"
STATE="/tmp/fma-state-$T"
SCORES="/tmp/fma-scores-$T"
rm -rf "$STATE" "$SCORES"
mkdir -p "$STATE" "$SCORES"

MAKEFUN_ENV_DATA_DIR="$TRIAL/input/env" MAKEFUN_STATE_DIR="$STATE" MAKEFUN_ENV_PORT="$PORT" \
  node "$KIT/program/env/server.mjs" > "$STATE/env.log" 2>&1 &
ENV_PID=$!
# Wait for the env's ready file.
for _ in $(seq 1 50); do
  [ -f "$STATE/ready" ] && break
  sleep 0.1
done
if [ ! -f "$STATE/ready" ]; then echo "env did not become ready"; cat "$STATE/env.log"; kill $ENV_PID 2>/dev/null; exit 1; fi

MAKEFUN_ENV_URL="http://127.0.0.1:$PORT" \
  MAKEFUN_PROMPT_FILE="$TRIAL/input/prompt.txt" \
  MAKEFUN_TRIAL_SEED="local-trial-$T" \
  timeout 170 node "$REPO/main.ts" > "$STATE/bot.log" 2>&1
BOT_EXIT=$?
echo "bot exit=$BOT_EXIT"
kill -TERM $ENV_PID 2>/dev/null
wait $ENV_PID 2>/dev/null
if [ ! -f "$STATE/export/state.json" ]; then echo "no state export"; cat "$STATE/env.log"; exit 1; fi
MAKEFUN_OUTPUT_DIR="$STATE/export" MAKEFUN_REFERENCE_DIR="$TRIAL/reference" MAKEFUN_SCORES_DIR="$SCORES" \
  node "$KIT/program/scorer/score.mjs"
echo "--- scores:"; cat "$SCORES/scores.json"
