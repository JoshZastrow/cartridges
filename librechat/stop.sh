#!/usr/bin/env bash
# Stop LibreChat, MongoDB and the gateway, then stop this app's idle Modal GPU
# containers (Teacher + CartridgeServer) so they do not bill through the
# scaledown window. Mongo data (librechat/.mongo-data) is kept; `make chat-clean`
# removes it.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
LOG_DIR="$HERE/logs"

stop_port() {  # name port pidfile
  local pids
  pids="$( { [ -f "$3" ] && cat "$3"; lsof -nP -iTCP:"$2" -sTCP:LISTEN -t; } 2>/dev/null | sort -u)"
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null && echo "stopped $1 ($(echo $pids | tr '\n' ' '))"
  else
    echo "$1 not running"
  fi
  rm -f "$3"
}

stop_port librechat 3080  "$LOG_DIR/backend.pid"
stop_port gateway   8008  "$LOG_DIR/gateway.pid"
stop_port mongod    27017 "$LOG_DIR/mongo.pid"

cd "$REPO"
set -a; [ -f .env.local ] && . ./.env.local; set +a
uv run python -c "
from cartridges.cartridge import teacher
for app in ('cartridges-teacher', 'cartridges-cartridge-serve'):
    print('modal containers stopped for', app, teacher.stop(app, recheck_s=0))
"
