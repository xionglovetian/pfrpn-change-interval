#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
TLOG=$ROOT/logs/train.log
ELOG=$ROOT/logs/eval.log

echo "=== EVAL WAIT START $(date) ===" >> "$ELOG"
for i in $(seq 1 2880); do
  grep -q "A0-A9 DONE" "$TLOG" 2>/dev/null && break
  if grep -q "GATE FAILED\|PIPELINE FAILED" "$TLOG" 2>/dev/null; then
    echo "TRAINING ABORTED, no eval" >> "$ELOG"; exit 1
  fi
  sleep 30
done
grep -q "A0-A9 DONE" "$TLOG" || { echo "EVAL WAIT TIMEOUT" >> "$ELOG"; exit 1; }
echo "=== EVAL START $(date) ===" >> "$ELOG"
cd "$PROJ"
export PYTHONPATH="$PROJ"
$PY tools/eval_cloud_ablations.py >> "$ELOG" 2>&1
echo "=== EVAL DONE $(date) ===" >> "$ELOG"
df -h /root/autodl-tmp >> "$ELOG"