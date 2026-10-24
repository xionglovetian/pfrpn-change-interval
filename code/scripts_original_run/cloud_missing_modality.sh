#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
ELOG=$ROOT/logs/eval.log
MLOG=$ROOT/logs/missing_modality.log

echo "=== MISSING-MODALITY WAIT $(date) ===" >> "$MLOG"
for i in $(seq 1 2880); do
  grep -q "EVAL DONE" "$ELOG" 2>/dev/null && break
  sleep 30
done
grep -q "EVAL DONE" "$ELOG" || { echo "TIMEOUT" >> "$MLOG"; exit 1; }
cd "$PROJ"
export PYTHONPATH="$PROJ"
echo "=== MISSING-MODALITY RUN $(date) ===" >> "$MLOG"
$PY tools/eval_cloud_ablations.py A6_add_cross_modal_router \
  --missing-modality --missing-split test >> "$MLOG" 2>&1
echo "=== MISSING-MODALITY DONE $(date) ===" >> "$MLOG"