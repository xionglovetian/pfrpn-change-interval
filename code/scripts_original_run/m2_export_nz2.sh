#!/bin/bash
set -u
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
M=$PROJ/data/landslide_multimodal_hn_lb/manifests
OUT=/root/autodl-tmp/m2_roi_cache_v2
LOG=/root/autodl-tmp/m2_export_v2.log
cd $PROJ
export PYTHONPATH=$PROJ
$PY -c "
import json, pathlib
M = pathlib.Path('$M')
d = json.loads((M/'event_newzealand.json').read_text(encoding='utf-8'))
for i in range(2):
    (M/('nz2_s%d.json' % i)).write_text(json.dumps(d[i::2], ensure_ascii=False), encoding='utf-8')
    print('nz2_s%d' % i, len(d[i::2]), flush=True)
" >> $LOG 2>&1
for i in 0 1; do
  $PY tools/inference/export_proposals.py configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py \
    work_dirs/m2_pf_rpn/hn_lb/A4_add_change/epoch_8.pth \
    --out $OUT/A4_newzealand --split val \
    --roi-per-modality --roi-size 6 --max-per-img 300 --score-thr 0.05 \
    --cfg-options train_dataloader.dataset.ann_file=manifests/nz2_s$i.json >> $LOG 2>&1 &
done
wait
echo "  [done] nz2 rc=$? $(date)" >> $LOG
touch $OUT/NZ_DONE