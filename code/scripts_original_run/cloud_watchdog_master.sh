#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
LOG=$ROOT/logs/watchdog_master.log
RESTARTS=0
echo "=== MASTER WATCHDOG START $(date) ===" >> "$LOG"
while true; do
  sleep 300
  if grep -q "MASTER (phases 1-4) DONE" "$ROOT/logs/master.log" 2>/dev/null; then
    echo "watchdog_master: all phases done, exiting $(date)" >> "$LOG"; exit 0
  fi
  if ! pgrep -f "cloud_[m]aster\.sh" > /dev/null 2>&1; then
    RESTARTS=$((RESTARTS+1))
    if [ "$RESTARTS" -gt 30 ]; then
      echo "watchdog_master: too many restarts, giving up $(date)" >> "$LOG"; exit 1
    fi
    echo "watchdog_master: master not alive, restart #$RESTARTS $(date)" >> "$LOG"
    cd /root/autodl-tmp || exit 1
    setsid nohup bash /root/autodl-tmp/cloud_master.sh >> "$ROOT/logs/master.out" 2>&1 < /dev/null &
    sleep 90
  fi
done