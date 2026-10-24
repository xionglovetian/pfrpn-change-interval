#!/bin/bash
set -u
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
OUT=/root/autodl-tmp/m2_roi_cache
LOG=/root/autodl-tmp/m2_export_roi.log
CFG=configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py
CKPT=work_dirs/m2_pf_rpn/hn_lb/A4_add_change/epoch_8.pth
cd $PROJ
export PYTHONPATH=$PROJ
echo "=== M2 ROI EXPORT CHAIN START $(date) ===" >> $LOG

$PY - <<'PYEOF' >> $LOG 2>&1
import json, pathlib
src = pathlib.Path("data/landslide_multimodal_hn_lb/manifests")
for split in ("train", "val", "test"):
    d = json.loads((src / f"{split}.json").read_text(encoding="utf-8"))
    pos = [e for e in d if (e.get("labels") or {}).get("annotations")]
    (src / f"{split}_pos.json").write_text(json.dumps(pos, ensure_ascii=False), encoding="utf-8")
    evs = {}
    for e in d:
        evs.setdefault(e.get("event_id"), []).append(e)
    for ev, items in evs.items():
        (src / f"event_{ev}.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(split, "total", len(d), "pos", len(pos), "events", {k: len(v) for k, v in evs.items()}, flush=True)
PYEOF

run_export () {
  name=$1; ann=$2; split=$3
  echo "=== EXPORT $name START $(date) ===" >> $LOG
  $PY tools/inference/export_proposals.py $CFG $CKPT \
     --out $OUT/A4_$name --split $split \
     --roi-per-modality --roi-size 6 --max-per-img 300 --score-thr 0.05 \
     --cfg-options train_dataloader.dataset.ann_file=$ann >> $LOG 2>&1
  rc=$?
  echo "=== EXPORT $name rc=$rc $(date) ===" >> $LOG
  touch $OUT/A4_$name.DONE
}

run_export hokkaido   manifests/event_hokkaido.json   val
run_export newzealand manifests/event_newzealand.json val
run_export train_pos  manifests/train_pos.json        train
run_export test       manifests/test.json             test
echo "=== M2 ROI EXPORT CHAIN DONE $(date) ===" >> $LOG