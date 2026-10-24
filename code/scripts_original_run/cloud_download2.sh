#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
LOG=$ROOT/logs
mkdir -p "$LOG"
detach_wait() { while pgrep -x wget >/dev/null; do sleep 2; done; }
detach_wait

dl() {
  local url="$1" dest="$2"
  wget -c --tries=10 --timeout=90 --waitretry=5 -q -O "$dest" "$url" \
    && echo "OK $(basename $dest) $(date +%T)" >> "$LOG/dl_ok.txt" \
    || echo "FAIL $url $(date +%T)" >> "$LOG/dl_fail.txt"
}
export -f dl
export LOG

echo "PARALLEL_RESTART $(date)" >> "$LOG/downloader.log"
cat "$LOG/all_list.txt" | xargs -P 8 -L 1 bash -c 'dl "$0" "$1"'
echo "DOWNLOAD_QUEUE_DONE $(date)" >> "$LOG/downloader.log"