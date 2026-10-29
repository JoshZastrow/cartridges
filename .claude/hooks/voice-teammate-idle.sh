#!/bin/bash
# TeammateIdle gate for the voice epic (#7): an impl-N teammate may not idle
# while issue #N is still In Progress on the Cartridges board.
# Exit 2 keeps the teammate working and sends stderr as feedback.
set -u
name=$(jq -r '.teammate_name // empty' 2>/dev/null)
n=${name#impl-}
if [ -z "$n" ] || [ "$n" = "$name" ]; then
  exit 0  # not an impl-N teammate
fi
status=$(gh project item-list 1 --owner JoshZastrow --format json \
  --jq ".items[] | select(.content.number == $n) | .status" 2>/dev/null)
if [ "$status" = "In Progress" ]; then
  echo "issue #$n is still In Progress: post the summary comment and set Status to Review before idling" >&2
  exit 2
fi
exit 0
