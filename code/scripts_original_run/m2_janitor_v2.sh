#!/bin/bash
LOG=/root/autodl-tmp/m2_janitor_v2.log
echo "=== V2 JANITOR START $(date) ===" >> $LOG
while true; do
  find /root/autodl-tmp/m2_roi_cache_v2 -name '*.features.npz' -mmin +2 -delete 2>/dev/null
  free=$(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')
  echo "sweep $(date) free=$free" >> $LOG
  [ -f /root/autodl-tmp/m2_roi_cache_v2/TEST_DONE ] && break
  sleep 120
done
find /root/autodl-tmp/m2_roi_cache_v2 -name '*.features.npz' -delete 2>/dev/null
echo "=== V2 JANITOR DONE $(date) ===" >> $LOG