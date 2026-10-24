#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
BASE=https://hf-mirror.com/datasets/paulhoehn/Sen12Landslides/resolve/main
mkdir -p "$ROOT/data_harmonized/s2" "$ROOT/data_harmonized/s1asc" "$ROOT/data_harmonized/s1dsc" "$ROOT/tasks/S12LS-LD/harmonized" "$ROOT/logs"

# tasks
for f in splits_aligned.json norm_aligned.json config.json; do
  curl -sL -C - --retry 5 --retry-delay 3 -o "$ROOT/tasks/S12LS-LD/harmonized/$f" "$BASE/tasks/S12LS-LD/harmonized/$f"
done
curl -sL -C - --retry 5 --retry-delay 3 -o "$ROOT/inventories.zip" "$BASE/inventories.zip"

dl() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    # verify not a leftover partial by trying resume
    :
  fi
  wget -c --tries=10 --timeout=90 --waitretry=5 -q -O "$dest" "$url" \
    && echo "OK $(basename $dest)" >> "$ROOT/logs/dl_ok.txt" \
    || { echo "FAIL $url" >> "$ROOT/logs/dl_fail.txt"; }
}
export -f dl

# build lists
: > "$ROOT/logs/s2_list.txt"
for i in $(seq -w 1 28); do
  echo "$BASE/data_harmonized/s2/s2_part${i}.tar.gz $ROOT/data_harmonized/s2/s2_part${i}.tar.gz"
done > "$ROOT/logs/s2_list.txt"

: > "$ROOT/logs/s1asc_list.txt"
for i in $(seq -w 1 14); do
  echo "$BASE/data_harmonized/s1asc/s1asc_part${i}.tar.gz $ROOT/data_harmonized/s1asc/s1asc_part${i}.tar.gz"
done > "$ROOT/logs/s1asc_list.txt"

echo "START s2 $(date)" >> "$ROOT/logs/dl.log"
cat "$ROOT/logs/s2_list.txt" | xargs -P 4 -L 1 bash -c 'dl "$0" "$1"'
echo "DONE s2 $(date)" >> "$ROOT/logs/dl.log"

echo "START s1asc $(date)" >> "$ROOT/logs/dl.log"
cat "$ROOT/logs/s1asc_list.txt" | xargs -P 4 -L 1 bash -c 'dl "$0" "$1"'
echo "DONE s1asc $(date)" >> "$ROOT/logs/dl.log"

echo "ALL_DOWNLOADS_DONE $(date)" >> "$ROOT/logs/dl.log"