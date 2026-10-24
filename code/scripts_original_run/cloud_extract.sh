#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
LOG=$ROOT/logs
echo "WAIT_EXTRACT_START $(date)" >> "$LOG/extract.log"
# wait for downloader to finish
for i in $(seq 1 900); do
  if grep -q "DOWNLOAD_QUEUE_DONE" "$LOG/downloader.log" 2>/dev/null; then break; fi
  sleep 20
done
if ! grep -q "DOWNLOAD_QUEUE_DONE" "$LOG/downloader.log" 2>/dev/null; then
  echo "EXTRACT_TIMEOUT_WAITING $(date)" >> "$LOG/extract.log"; exit 1
fi
echo "DOWNLOADS_DONE $(date)" >> "$LOG/extract.log"
echo "=== download results ===" >> "$LOG/extract.log"
echo "OK count: $(wc -l < $LOG/dl_ok.txt 2>/dev/null || echo 0)" >> "$LOG/extract.log"
echo "FAIL count: $(wc -l < $LOG/dl_fail.txt 2>/dev/null || echo 0)" >> "$LOG/extract.log"
cat "$LOG/dl_fail.txt" >> "$LOG/extract.log" 2>/dev/null || true

# restructure: ROOT/data_harmonized -> ROOT/data/data_harmonized
mkdir -p "$ROOT/data"
if [ -d "$ROOT/data_harmonized" ]; then
  mv "$ROOT/data_harmonized" "$ROOT/data/" || true
fi
D="$ROOT/data/data_harmonized"
for mod in s2 s1asc; do
  echo "=== extract $mod $(date) ===" >> "$LOG/extract.log"
  for f in "$D/$mod"/*.tar.gz; do
    [ -e "$f" ] || continue
    tar -xzf "$f" -C "$D/$mod" && rm -f "$f" && echo "extracted $(basename $f)" >> "$LOG/extract.log" \
      || echo "EXTRACT_FAIL $f" >> "$LOG/extract.log"
  done
  echo "$mod nc count: $(ls $D/$mod/*.nc 2>/dev/null | wc -l)" >> "$LOG/extract.log"
done
echo "EXTRACT_DONE $(date)" >> "$LOG/extract.log"
df -h /root/autodl-tmp >> "$LOG/extract.log"