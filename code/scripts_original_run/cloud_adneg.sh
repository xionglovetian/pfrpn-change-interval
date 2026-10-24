#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
LOG=$ROOT/logs/adneg_convert.log
: > "$LOG"
cd "$PROJ"
echo "=== AD NEGATIVE CONVERT START $(date) ===" >> "$LOG"
$PY tools/data/prepare_sen12landslides.py "$ROOT" \
  --task S12LS-AD --version harmonized \
  --splits-file "$ROOT/tasks/ad_splits.json" \
  --out "$PROJ/data/landslide_multimodal_adneg" \
  --sar-source asc --skip-annotated >> "$LOG" 2>&1
echo "=== AD NEGATIVE CONVERT DONE $(date) ===" >> "$LOG"
ls "$PROJ/data/landslide_multimodal_adneg/manifests" >> "$LOG" 2>&1
du -sh "$PROJ/data/landslide_multimodal_adneg" >> "$LOG" 2>&1