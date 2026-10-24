#!/bin/bash
set -u
LOG=/root/autodl-tmp/stall_watch.log
echo "=== STALL WATCH START $(date) ===" >> $LOG
declare -A LAST
while true; do
  sleep 300
  for f in $(find /root/autodl-tmp/PF-RPN/work_dirs/m2_pf_rpn/loeo /root/autodl-tmp/PF-RPN/work_dirs/m2_pf_rpn/loro -name train.log 2>/dev/null); do
    now=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    prev=${LAST[$f]:-0}
    if [ "$now" != "0" ] && [ "$now" == "$prev" ]; then
      age=$(( $(date +%s) - now ))
      if [ $age -gt 900 ]; then
        echo "$(date) STALL ${age}s : $f" >> $LOG
      fi
    fi
    LAST[$f]=$now
  done
  if [ -f /root/autodl-tmp/loro/LORO_DONE ]; then
    echo "$(date) ALL DONE, watch exits" >> $LOG; exit 0
  fi
done
