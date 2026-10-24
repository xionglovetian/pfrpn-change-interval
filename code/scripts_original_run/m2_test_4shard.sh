#!/bin/bash
set -u
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
M=$PROJ/data/landslide_multimodal_hn_lb/manifests
OUT=/root/autodl-tmp/m2_roi_cache_v2
LOG=/root/autodl-tmp/m2_export_v3_test.log
cd $PROJ
export PYTHONPATH=$PROJ

# kill any previous test watcher
for p in $(pgrep -f m2_test_watch.sh); do kill -9 $p 2>/dev/null || true; done
echo "=== TEST-4SHARD WATCHER START $(date) ===" >> $LOG

$PY -c "
import json, pathlib
M = pathlib.Path('$M')
d = json.loads((M/'test.json').read_text(encoding='utf-8'))
for i in range(4):
    part = d[i::4]
    (M/('te4_s%d.json' % i)).write_text(json.dumps(part, ensure_ascii=False), encoding='utf-8')
    print('te4_s%d' % i, len(part), flush=True)
" >> $LOG 2>&1

# wait for train_pos to finish
for i in $(seq 1 240); do
  n=$(ps -eo cmd | grep "[e]xport_proposals" | grep -c "A4_train_pos" || true)
  [ "$n" -eq 0 ] && break
  sleep 15
done
echo "=== train_pos clear $(date) -> launching 4 test shards ===" >> $LOG
sleep 10
for i in 0 1 2 3; do
  $PY tools/inference/export_proposals.py configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py \
    work_dirs/m2_pf_rpn/hn_lb/A4_add_change/epoch_8.pth \
    --out $OUT/A4_test --split test \
    --roi-per-modality --roi-size 6 --max-per-img 300 --score-thr 0.05 \
    --cfg-options train_dataloader.dataset.ann_file=manifests/te4_s$i.json >> $LOG 2>&1 &
done
wait
echo "=== TEST 4-SHARD DONE $(date) ===" >> $LOG
find $OUT/A4_test -name "*.roi.npz" | wc -l >> $LOG
touch $OUT/TEST_DONE