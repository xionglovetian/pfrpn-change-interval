#!/bin/bash
set -u
MARK=/root/autodl-tmp/loeo/y8_loeo_hokkaido_metrics.json
echo "chain3: waiting for LOEO last fold $(date)"
for _ in $(seq 1 1200); do
  [ -f "$MARK" ] && break
  sleep 60
done
if [ -f "$MARK" ]; then
  echo "chain3: LOEO done, launching LORO $(date)"
  bash /root/autodl-tmp/loro_run.sh
else
  echo "chain3: TIMEOUT waiting for $MARK $(date)"
fi
