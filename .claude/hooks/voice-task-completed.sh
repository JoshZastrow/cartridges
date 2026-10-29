#!/bin/bash
# TaskCompleted gate for the voice epic (#7): a task may close only when the
# voice package tests and the untouched root cartridges tests both pass.
# Exit 2 blocks completion and sends stderr back to the teammate.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 0
if [ -d voice ]; then
  if ! uv run --package voice pytest -q 2>&1 | tail -20; then
    echo "voice tests failed; task stays open" >&2
    exit 2
  fi
fi
if ! uv run pytest -q 2>&1 | tail -20; then
  echo "root cartridges tests failed; the isolation rule is broken" >&2
  exit 2
fi
exit 0
