#!/bin/bash
set -u
echo "chain: waiting for post_extra_run $(date)"
for _ in $(seq 1 1440); do
  pgrep -f "post_extra_run" >/dev/null 2>&1 || break
  sleep 30
done
echo "chain: launching loeo $(date)"
bash /root/autodl-tmp/loeo_run.sh
