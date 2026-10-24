#!/bin/bash
set -u
LOG=/root/autodl-tmp/stall_watch.log
echo "=== WATCH v2 START $(date) ===" >> $LOG
while true; do
  sleep 300
  if [ -f /root/autodl-tmp/loro/LORO_DONE ]; then echo "$(date) ALL DONE" >> $LOG; exit 0; fi
  alive=$(ps -eo cmd | grep -c "[t]ools/train.py")
  if [ "$alive" -eq 0 ]; then continue; fi
  while IFS= read -r f; do
    age=$(( $(date +%s) - $(stat -c %Y "$f" 2>/dev/null || date +%s) ))
    if [ "$age" -gt 1200 ]; then echo "$(date) REAL-STALL ${age}s : $f" >> $LOG; fi
  done < <(find /root/autodl-tmp/PF-RPN/work_dirs/m2_pf_rpn/loeo /root/autodl-tmp/PF-RPN/work_dirs/m2_pf_rpn/loro -name train.log 2>/dev/null)
done
