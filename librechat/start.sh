#!/usr/bin/env bash
# Start the Journal gateway (8008) + MongoDB + LibreChat (http://localhost:3080).
# `make chat-up` / `make chat-down` wrap this and stop.sh.
# Usage: librechat/start.sh            (from the cartridges repo)
#        LIBRECHAT_DIR=... librechat/start.sh
#        LIBRECHAT_FOREGROUND=1 librechat/start.sh   (backend in the foreground; Playwright webServer)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
LIBRECHAT_DIR="${LIBRECHAT_DIR:-$HOME/Github/LibreChat}"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"

[ -d "$LIBRECHAT_DIR" ] || { echo "LibreChat not found at $LIBRECHAT_DIR (git clone https://github.com/danny-avila/LibreChat)"; exit 1; }

# 1. .env: copy the example and fill secrets on first run.
if [ ! -f "$LIBRECHAT_DIR/.env" ]; then
  sed -e "s|^CREDS_KEY=$|CREDS_KEY=$(openssl rand -hex 32)|" \
      -e "s|^CREDS_IV=$|CREDS_IV=$(openssl rand -hex 16)|" \
      -e "s|^JWT_SECRET=$|JWT_SECRET=$(openssl rand -hex 32)|" \
      -e "s|^JWT_REFRESH_SECRET=$|JWT_REFRESH_SECRET=$(openssl rand -hex 32)|" \
      "$HERE/.env.example" > "$LIBRECHAT_DIR/.env"
  echo "wrote $LIBRECHAT_DIR/.env with fresh secrets"
fi

# 2. librechat.yaml: symlink the repo-tracked config into the LibreChat root.
ln -sfn "$HERE/librechat.yaml" "$LIBRECHAT_DIR/librechat.yaml"

# 3. Gateway on 127.0.0.1:8008 (Modal GPU containers start on the first request).
if lsof -nP -iTCP:8008 -sTCP:LISTEN >/dev/null; then
  echo "gateway already listening on 8008"
else
  # Redirect the whole background list, or its wrapper shell keeps the caller's
  # stdout open (a `make chat-up | ...` pipe would then never end).
  (cd "$REPO" && exec uv run cartridges gateway_serve --port 8008) > "$LOG_DIR/gateway.log" 2>&1 < /dev/null &
  echo $! > "$LOG_DIR/gateway.pid"
  for _ in $(seq 1 60); do lsof -nP -iTCP:8008 -sTCP:LISTEN >/dev/null && break; sleep 1; done
  echo "gateway started (pid $(cat "$LOG_DIR/gateway.pid"), log $LOG_DIR/gateway.log)"
fi

# 4. MongoDB on 127.0.0.1:27017 (persistent data in librechat/.mongo-data).
if lsof -nP -iTCP:27017 -sTCP:LISTEN >/dev/null; then
  echo "mongod already listening on 27017"
else
  (cd "$REPO" && exec node "$HERE/mongo.js") > "$LOG_DIR/mongo.log" 2>&1 < /dev/null &
  echo $! > "$LOG_DIR/mongo.pid"
  for _ in $(seq 1 60); do lsof -nP -iTCP:27017 -sTCP:LISTEN >/dev/null && break; sleep 1; done
  echo "mongod started (pid $(cat "$LOG_DIR/mongo.pid"), log $LOG_DIR/mongo.log)"
fi

# 5. Dependencies + frontend build (skipped when already built).
cd "$LIBRECHAT_DIR"
[ -d node_modules ] || npm ci
[ -f client/dist/index.html ] || npm run frontend

# 6. Backend on 3080.
if lsof -nP -iTCP:3080 -sTCP:LISTEN >/dev/null; then
  echo "something already listening on 3080; not starting the backend"
else
  if [ "${LIBRECHAT_FOREGROUND:-}" = "1" ]; then
    # Playwright's webServer owns the process: run the backend in the foreground.
    exec npm run backend
  fi
  npm run backend > "$LOG_DIR/backend.log" 2>&1 < /dev/null &
  echo $! > "$LOG_DIR/backend.pid"
  for _ in $(seq 1 90); do curl -sf http://localhost:3080/health >/dev/null && break; sleep 1; done
  echo "backend started (pid $(cat "$LOG_DIR/backend.pid"), log $LOG_DIR/backend.log)"
fi

cat <<EOF

LibreChat: http://localhost:3080
Gateway:   http://127.0.0.1:8008/v1

Stop everything (incl. idle Modal GPU containers): $HERE/stop.sh   (or: make chat-down)
EOF
