#!/bin/bash
set -u
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
M=/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb/manifests
OUT=/root/autodl-tmp/m2_roi_cache_v2
LOG=/root/autodl-tmp/m2_export_v2.log
CFG=configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py
CKPT=work_dirs/m2_pf_rpn/hn_lb/A4_add_change/epoch_8.pth
mkdir -p $OUT
cd $PROJ
export PYTHONPATH=$PROJ
echo "=== V2 EXPORT START $(date) ===" >> $LOG

$PY - <<PYEOF >> $LOG 2>&1
import json, pathlib
M = pathlib.Path("/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb/manifests")
def shard(src, n, tag):
    d = json.loads((M/src).read_text(encoding="utf-8"))
    for i in range(n):
        part = d[i::n]
        (M/f"{tag}_s{i}.json").write_text(json.dumps(part, ensure_ascii=False), encoding="utf-8")
        print(tag, i, len(part), flush=True)
shard("train_pos.json", 4, "tr")
shard("test.json", 2, "te")
shard("event_newzealand.json", 1, "nz")
shard("event_hokkaido.json", 1, "hk")
PYEOF

run () {
  name=$1; ann=$2; split=$3; out=$4
  $PY tools/inference/export_proposals.py $CFG $CKPT \
     --out $OUT/$out --split $split \
     --roi-per-modality --roi-size 6 --max-per-img 300 --score-thr 0.05 \
     --cfg-options train_dataloader.dataset.ann_file=$ann >> $LOG 2>&1
  echo "  [done] $name rc=$? $(date)" >> $LOG
}

for i in 0 1 2 3; do run "tr$i" manifests/tr_s$i.json train A4_train_pos & done
for i in 0 1; do run "te$i" manifests/te_s$i.json test A4_test & done
run nz manifests/nz_s0.json val A4_newzealand &
run hk manifests/hk_s0.json val A4_hokkaido &
wait
echo "=== V2 EXPORT ALL DONE $(date) ===" >> $LOG
touch $OUT/V2_DONE