#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
BASE=https://hf-mirror.com/datasets/paulhoehn/Sen12Landslides/resolve/main
D=$ROOT/data/data_harmonized
LOG=$ROOT/logs/dsc_download.log
mkdir -p "$D/s1dsc"
: > "$LOG"
echo "=== S1DSC DOWNLOAD START $(date) ===" >> "$LOG"

: > "$ROOT/logs/dsc_list.txt"
for i in $(seq -w 1 13); do
  echo "$BASE/data_harmonized/s1dsc/s1dsc_part${i}.tar.gz $D/s1dsc/s1dsc_part${i}.tar.gz" >> "$ROOT/logs/dsc_list.txt"
done

dl() {
  local url="$1" dest="$2"
  for attempt in 1 2 3 4 5; do
    wget -c --tries=10 --timeout=120 --waitretry=5 -q -O "$dest" "$url" \
      && gzip -t "$dest" 2>/dev/null && { echo "OK $(basename $dest)" >> "$LOG"; return 0; }
    echo "retry $(basename $dest) attempt=$attempt $(date)" >> "$LOG"
    sleep 10
  done
  echo "FAIL $(basename $dest)" >> "$LOG"
}
export -f dl
export LOG
cat "$ROOT/logs/dsc_list.txt" | xargs -P 8 -L 1 bash -c 'dl "$0" "$1"'
echo "=== DOWNLOAD DONE $(date) ===" >> "$LOG"

bad=0
for f in "$D/s1dsc"/*.tar.gz; do
  [ -e "$f" ] || continue
  gzip -t "$f" 2>/dev/null || { echo "BAD $(basename $f)" >> "$LOG"; bad=$((bad+1)); }
done
if [ "$bad" -ne 0 ]; then echo "=== VERIFY FAILED ($bad) $(date) ===" >> "$LOG"; exit 1; fi

for f in "$D/s1dsc"/*.tar.gz; do
  [ -e "$f" ] || continue
  tar -xzf "$f" -C "$D/s1dsc" && rm -f "$f" && echo "EXTRACTED $(basename $f)" >> "$LOG"
done
echo "s1dsc nc: $(ls $D/s1dsc/*.nc 2>/dev/null | wc -l)" >> "$LOG"
echo "=== S1DSC DONE $(date) ===" >> "$LOG"
df -h /root/autodl-tmp >> "$LOG"