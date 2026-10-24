#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
LOG=$ROOT/logs/watchdog_late.log
echo "=== LATE WATCHDOG (v2) START $(date) ===" >> "$LOG"
while true; do
  sleep 300
  if grep -q "OD CHAIN DONE" "$ROOT/logs/chain_od.log" 2>/dev/null; then od_done=1; else od_done=0; fi
  if grep -q "DSC CHAIN DONE" "$ROOT/logs/chain_dsc.log" 2>/dev/null; then dsc_done=1; else dsc_done=0; fi
  if [ "$od_done" = "1" ] && [ "$dsc_done" = "1" ]; then
    echo "watchdog_late: both OD and DSC done, exiting $(date)" >> "$LOG"; exit 0
  fi
  if [ "$od_done" = "0" ] && ! pgrep -f "cloud_chain_od\.sh" >/dev/null 2>&1; then
    echo "watchdog_late: OD chain not alive, restart $(date)" >> "$LOG"
    cd /root/autodl-tmp && setsid nohup bash /root/autodl-tmp/cloud_chain_od.sh >> "$ROOT/logs/chain_od.out" 2>&1 < /dev/null &
    sleep 60
  fi
  if [ "$dsc_done" = "0" ] && ! pgrep -f "cloud_chain_dsc\.sh" >/dev/null 2>&1; then
    echo "watchdog_late: DSC chain not alive, restart $(date)" >> "$LOG"
    cd /root/autodl-tmp && setsid nohup bash /root/autodl-tmp/cloud_chain_dsc.sh >> "$ROOT/logs/chain_dsc.out" 2>&1 < /dev/null &
    sleep 60
  fi
done