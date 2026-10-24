#!/bin/bash
set -u
PROJ=/root/autodl-tmp/PF-RPN
M2=/root/autodl-tmp/landslide-framework/module2
PY=/root/miniconda3/bin/python3
OUT=/root/autodl-tmp/m2_roi_cache_v2
TRAIN_LOG=/root/autodl-tmp/m2_train_v3.log
cd $M2
export PYTHONPATH="$M2:$PROJ"
echo "=== M2 TRAIN v3 START $(date) (in-domain negatives) ===" >> $TRAIN_LOG
$PY -m src.train \
  --config configs/module2.yaml \
  --gt-dir $PROJ/data/landslide_multimodal_hn_lb/manifests \
  --data-root $PROJ/data/landslide_multimodal_hn_lb \
  --train-root $OUT/A4_train_pos --train-split train \
  --neg-root $OUT/A4_hokkaido --neg-split val --neg-events hokkaido \
  --cal-root $OUT/A4_newzealand --cal-split val --cal-events newzealand \
  --dev-events itogon --include-train-negatives --max-train-negatives 40000 \
  --epochs 10 --batch-size 32 --out-dir /root/autodl-tmp/m2_train_v3 >> $TRAIN_LOG 2>&1
rc=$?
echo "=== M2 TRAIN v3 rc=$rc DONE $(date) ===" >> $TRAIN_LOG