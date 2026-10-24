#!/bin/bash
set -u
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
OUT=/root/autodl-tmp/m2_roi_cache_v2
LOG=/root/autodl-tmp/m2_export_v2.log
cd $PROJ
export PYTHONPATH=$PROJ
echo "=== TEST WATCHER START $(date) ===" >> $LOG

# wait until no train_pos export process remains
for i in $(seq 1 720); do
  n=$(ps -eo cmd | grep "[e]xport_proposals" | grep -c "A4_train_pos" || true)
  [ "$n" -eq 0 ] && break
  sleep 30
done
echo "=== train_pos finished $(date), restarting test ===" >> $LOG
sleep 20

for i in 0 1; do
  $PY tools/inference/export_proposals.py configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py \
    work_dirs/m2_pf_rpn/hn_lb/A4_add_change/epoch_8.pth \
    --out $OUT/A4_test --split test \
    --roi-per-modality --roi-size 6 --max-per-img 300 --score-thr 0.05 \
    --cfg-options train_dataloader.dataset.ann_file=manifests/te_s$i.json >> $LOG 2>&1 &
done
wait
echo "=== TEST REDONE $(date) ===" >> $LOG
find $OUT/A4_test -name "*.roi.npz" | wc -l >> $LOG
touch $OUT/TEST_DONE