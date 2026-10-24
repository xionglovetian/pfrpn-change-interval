#!/bin/bash
set -u
PROJ=/root/autodl-tmp/PF-RPN
M2=/root/autodl-tmp/landslide-framework/module2
PY=/root/miniconda3/bin/python3
OUT=/root/autodl-tmp/m2_roi_cache_v2
TRAIN_LOG=/root/autodl-tmp/m2_train_v2.log
cd $M2
export PYTHONPATH="$M2:$PROJ"
echo "=== M2 TRAIN v2 START $(date) ===" >> $TRAIN_LOG
echo "  train=A4_train_pos(3067)  neg=A4_hokkaido(284,hokkaido)  cal=A4_newzealand(1096)  dev=itogon" >> $TRAIN_LOG
$PY -m src.train \
  --config configs/module2.yaml \
  --gt-dir $PROJ/data/landslide_multimodal_hn_lb/manifests \
  --data-root $PROJ/data/landslide_multimodal_hn_lb \
  --train-root $OUT/A4_train_pos --train-split train \
  --neg-root $OUT/A4_hokkaido --neg-split val --neg-events hokkaido \
  --cal-root $OUT/A4_newzealand --cal-split val --cal-events newzealand \
  --dev-events itogon \
  --epochs 20 --batch-size 32 --out-dir /root/autodl-tmp/m2_train_v2 >> $TRAIN_LOG 2>&1
rc=$?
echo "=== M2 TRAIN v2 rc=$rc DONE $(date) ===" >> $TRAIN_LOG