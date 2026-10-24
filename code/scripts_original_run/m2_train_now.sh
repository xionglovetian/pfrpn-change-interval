#!/bin/bash
set -u
PROJ=/root/autodl-tmp/PF-RPN
M2=/root/autodl-tmp/landslide-framework/module2
PY=/root/miniconda3/bin/python3
OUT=/root/autodl-tmp/m2_roi_cache
TRAIN_LOG=/root/autodl-tmp/m2_train.log
cd $M2
export PYTHONPATH="$M2:$PROJ"
echo "=== M2 TRAIN START $(date) (memory-lean, no test yet) ===" >> $TRAIN_LOG
$PY -m src.train \
  --config configs/module2.yaml \
  --gt-dir $PROJ/data/landslide_multimodal_hn_lb/manifests \
  --data-root $PROJ/data/landslide_multimodal_hn_lb \
  --train-root $OUT/A4_train_pos --train-split train \
  --neg-root $OUT/A4_hokkaido --neg-split val --neg-events hokkaido \
  --cal-root $OUT/A4_newzealand --cal-split val --cal-events newzealand \
  --dev-events itogon \
  --epochs 20 --batch-size 32 --out-dir /root/autodl-tmp/m2_train >> $TRAIN_LOG 2>&1
rc=$?
echo "=== M2 TRAIN rc=$rc DONE $(date) ===" >> $TRAIN_LOG