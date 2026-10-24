#!/bin/bash
ROOT=/root/autodl-tmp/Sen12Landslides
D=$ROOT/data/data_harmonized/s2
URL=https://hf-mirror.com/datasets/paulhoehn/Sen12Landslides/resolve/main/data_harmonized/s2/s2_part20.tar.gz
LOG=$ROOT/logs/repair20.log
: > "$LOG"
for attempt in $(seq 1 40); do
  echo "attempt $attempt $(date)" >> "$LOG"
  curl -L -C - --retry 20 --retry-delay 5 --retry-all-errors \
       --connect-timeout 30 --speed-time 60 --speed-limit 20000 \
       -o "$D/s2_part20.tar.gz" "$URL" >> "$LOG" 2>&1
  rc=$?
  if gzip -t "$D/s2_part20.tar.gz" 2>/dev/null; then
    echo "GZIP_OK $(date)" >> "$LOG"; break
  fi
  echo "bad (curl rc=$rc), retrying $(date)" >> "$LOG"
  sleep 5
done
gzip -t "$D/s2_part20.tar.gz" 2>/dev/null && echo "PART20_REPAIRED $(date)" >> "$LOG" || echo "PART20_FAILED $(date)" >> "$LOG"