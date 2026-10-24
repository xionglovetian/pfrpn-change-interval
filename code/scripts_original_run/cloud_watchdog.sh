#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
LOG=$ROOT/logs/watchdog.log
CHAIN_LOG=$ROOT/logs/chain.log
RESTARTS=0
echo "=== WATCHDOG START $(date) ===" >> "$LOG"
while true; do
  sleep 240
  if grep -q "CHAIN DONE" "$CHAIN_LOG" 2>/dev/null; then
    echo "watchdog: CHAIN DONE detected, exiting $(date)" >> "$LOG"
    exit 0
  fi
  if ! pgrep -f "cloud_chain\.sh" > /dev/null 2>&1; then
    RESTARTS=$((RESTARTS+1))
    if [ "$RESTARTS" -gt 20 ]; then
      echo "watchdog: too many restarts ($RESTARTS), giving up $(date)" >> "$LOG"
      exit 1
    fi
    echo "watchdog: chain not alive, restart #$RESTARTS $(date)" >> "$LOG"
    cd /root/autodl-tmp || exit 1
    setsid nohup bash /root/autodl-tmp/cloud_chain.sh >> "$ROOT/logs/chain.out" 2>&1 < /dev/null &
    sleep 90
  fi
done