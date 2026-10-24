#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
LOG=$ROOT/logs/post_report.log
: > "$LOG"
echo "=== POST REPORT WAIT $(date) ===" >> "$LOG"
for i in $(seq 1 4320); do
  grep -q "CHAIN DONE" "$ROOT/logs/chain.log" 2>/dev/null && break
  sleep 60
done
grep -q "CHAIN DONE" "$ROOT/logs/chain.log" || { echo "POST REPORT TIMEOUT $(date)" >> "$LOG"; exit 1; }
echo "=== GENERATE REPORT $(date) ===" >> "$LOG"
cd "$PROJ"
export PYTHONPATH="$PROJ"
$PY tools/report_cloud_results.py >> "$LOG" 2>&1
echo "=== POST REPORT DONE $(date) ===" >> "$LOG"
ls -la "$PROJ/docs/cloud_m1_results.md" "$PROJ/artifacts/cloud_metrics/tables" >> "$LOG" 2>&1