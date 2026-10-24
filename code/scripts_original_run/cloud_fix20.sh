#!/bin/bash
ROOT=/root/autodl-tmp/Sen12Landslides
D=$ROOT/data/data_harmonized/s2
DEST=$D/s2_part20.tar.gz
URL=https://hf-mirror.com/datasets/paulhoehn/Sen12Landslides/resolve/main/data_harmonized/s2/s2_part20.tar.gz
LOG=$ROOT/logs/fix20.log
: > "$LOG"
rm -f "$DEST"
for i in $(seq 1 30); do
  echo "round $i $(date)" >> "$LOG"
  wget -c --tries=5 --timeout=120 --waitretry=10 -O "$DEST" "$URL" >> "$LOG" 2>&1
  if gzip -t "$DEST" 2>/dev/null; then
    echo "GZIP_OK round $i $(date)" >> "$LOG"
    echo "PART20_REPAIRED $(date)" >> "$LOG"
    exit 0
  fi
  echo "incomplete, size=$(stat -c %s $DEST 2>/dev/null) $(date)" >> "$LOG"
  sleep 10
done
echo "PART20_FAILED $(date)" >> "$LOG"
exit 1