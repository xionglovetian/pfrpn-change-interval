#!/bin/bash
# Module 2 does not consume the fused M1 features.npz (it uses roi.npz), and the
# fused copy is ~45% of the export size. Drop it (older than 2 min, so a file
# currently being written is never touched) to keep the disk bounded.
LOG=/root/autodl-tmp/m2_janitor.log
echo "=== JANITOR START $(date) ===" >> $LOG
while true; do
  find /root/autodl-tmp/m2_roi_cache -name '*.features.npz' -mmin +2 -delete 2>/dev/null
  if grep -q "M2 ROI EXPORT CHAIN DONE" /root/autodl-tmp/m2_export_roi.log 2>/dev/null; then
    find /root/autodl-tmp/m2_roi_cache -name '*.features.npz' -delete 2>/dev/null
    echo "=== JANITOR FINAL SWEEP DONE $(date) ===" >> $LOG
    break
  fi
  echo "sweep $(date) free=$(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')" >> $LOG
  sleep 120
done