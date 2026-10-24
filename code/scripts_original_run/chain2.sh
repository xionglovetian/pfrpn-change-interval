#!/bin/bash
set -u
MARK=/root/autodl-tmp/baselines/analysis/size_stratified.md
echo "chain2: waiting for post_extra outputs $(date)"
for _ in $(seq 1 720); do
  [ -f "$MARK" ] && break
  sleep 30
done
if [ -f "$MARK" ]; then
  echo "chain2: post_extra done, launching LOEO $(date)"
  bash /root/autodl-tmp/loeo_run.sh
else
  echo "chain2: TIMEOUT waiting for $MARK $(date)"
fi
