#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
LOG=$ROOT/logs/watchdog_hn.log
RESTARTS=0
echo "=== HN WATCHDOG START $(date) ===" >> "$LOG"
while true; do
  sleep 240
  if grep -q "HN CHAIN DONE" "$ROOT/logs/chain_hn.log" 2>/dev/null; then
    echo "watchdog_hn: HN CHAIN DONE, exiting $(date)" >> "$LOG"
    exit 0
  fi
  if ! pgrep -f "cloud_chain_hn\.sh" > /dev/null 2>&1; then
    RESTARTS=$((RESTARTS+1))
    if [ "$RESTARTS" -gt 20 ]; then
      echo "watchdog_hn: too many restarts, giving up $(date)" >> "$LOG"; exit 1
    fi
    echo "watchdog_hn: restart #$RESTARTS $(date)" >> "$LOG"
    cd /root/autodl-tmp || exit 1
    setsid nohup bash /root/autodl-tmp/cloud_chain_hn.sh >> "$ROOT/logs/chain_hn.out" 2>&1 < /dev/null &
    sleep 90
  fi
done