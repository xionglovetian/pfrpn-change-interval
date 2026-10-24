#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
BASE=https://hf-mirror.com/datasets/paulhoehn/Sen12Landslides/resolve/main
D=$ROOT/data/data_harmonized
LOG=$ROOT/logs
mkdir -p "$D/s2" "$D/s1asc" "$LOG"

echo "=== DOWNLOAD_FINAL START $(date) ===" >> "$LOG/download_final.log"

# 1. drop unreliable partially-extracted nc files from the aborted extraction
rm -f "$D/s2"/*.nc "$D/s1asc"/*.nc
echo "cleaned partial nc: $(date)" >> "$LOG/download_final.log"

# 2. build resume list (all 28 s2 + 14 s1asc)
: > "$LOG/final_list.txt"
for i in $(seq -w 1 28); do
  echo "$BASE/data_harmonized/s2/s2_part${i}.tar.gz $D/s2/s2_part${i}.tar.gz" >> "$LOG/final_list.txt"
done
for i in $(seq -w 1 14); do
  echo "$BASE/data_harmonized/s1asc/s1asc_part${i}.tar.gz $D/s1asc/s1asc_part${i}.tar.gz" >> "$LOG/final_list.txt"
done

dl() {
  local url="$1" dest="$2"
  wget -c --tries=15 --timeout=120 --waitretry=5 -q -O "$dest" "$url" \
    && echo "DL_OK $(basename $dest)" >> "$LOG/download_final.log" \
    || echo "DL_FAIL $(basename $dest)" >> "$LOG/download_final.log"
}
export -f dl
export LOG
cat "$LOG/final_list.txt" | xargs -P 16 -L 1 bash -c 'dl "$0" "$1"'
echo "=== DOWNLOAD PASS1 DONE $(date) ===" >> "$LOG/download_final.log"

# 3. integrity check + one repair pass
repair() {
  local url="$1" dest="$2"
  if [ ! -s "$dest" ] || ! gzip -t "$dest" 2>/dev/null; then
    echo "REPAIR $(basename $dest)" >> "$LOG/download_final.log"
    rm -f "$dest"
    wget -c --tries=15 --timeout=120 -q -O "$dest" "$url" \
      && echo "REPAIR_OK $(basename $dest)" >> "$LOG/download_final.log" \
      || echo "REPAIR_FAIL $(basename $dest)" >> "$LOG/download_final.log"
  fi
}
export -f repair
cat "$LOG/final_list.txt" | xargs -P 16 -L 1 bash -c 'repair "$0" "$1"'
echo "=== REPAIR DONE $(date) ===" >> "$LOG/download_final.log"

# 4. verify all, then extract + delete
bad=0
for mod in s2 s1asc; do
  for f in "$D/$mod"/*.tar.gz; do
    [ -e "$f" ] || { echo "MISSING $mod"; bad=$((bad+1)); continue; }
    if ! gzip -t "$f" 2>/dev/null; then echo "BAD $(basename $f)" >> "$LOG/download_final.log"; bad=$((bad+1)); fi
  done
done
echo "bad_count=$bad $(date)" >> "$LOG/download_final.log"
if [ "$bad" -ne 0 ]; then
  echo "=== VERIFY FAILED, not extracting $(date) ===" >> "$LOG/download_final.log"; exit 1
fi

for mod in s2 s1asc; do
  echo "=== EXTRACT $mod $(date) ===" >> "$LOG/download_final.log"
  for f in "$D/$mod"/*.tar.gz; do
    [ -e "$f" ] || continue
    if tar -xzf "$f" -C "$D/$mod"; then
      rm -f "$f"
      echo "EXTRACTED $(basename $f)" >> "$LOG/download_final.log"
    else
      echo "EXTRACT_FAIL $(basename $f)" >> "$LOG/download_final.log"; exit 1
    fi
  done
  echo "$mod nc: $(ls $D/$mod/*.nc 2>/dev/null | wc -l)" >> "$LOG/download_final.log"
done
echo "=== DOWNLOAD_FINAL DONE $(date) ===" >> "$LOG/download_final.log"
df -h /root/autodl-tmp >> "$LOG/download_final.log"