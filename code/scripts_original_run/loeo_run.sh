#!/bin/bash
# loeo_run.sh - leave-one-event-out: train A4 + YOLOv8s per fold, eval on the held-out event.
set -u
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
OUT=/root/autodl-tmp/loeo
LOG=/root/autodl-tmp/loeo.log
HN=data/landslide_multimodal_hn_lb/
M=$PROJ/data/landslide_multimodal_hn_lb/manifests
BASE="train_dataloader.batch_size=8 optim_wrapper.optimizer.lr=0.00002"
FOLDS="italy kyrgyzstan1 newzealand chimanimani hiroshima indonesia hokkaido"
exec > >(tee -a "$LOG") 2>&1
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
mkdir -p "$OUT"

echo "=== LOEO START $(date) ==="
echo "--- (waits removed) ---"
echo "=== clean GPU, starting folds $(date) ==="

$PY /root/autodl-tmp/make_loeo_yolo_folds.py

for F in $FOLDS; do
  echo ""
  echo "########## FOLD $F  $(date) ##########"

  echo "[$F] A4 train"
  $PY tools/run_cloud_ablations.py A4_add_change \
      --work-root work_dirs/m2_pf_rpn/loeo/$F \
      --cfg-options $BASE train_dataloader.dataset.data_root=$HN \
        train_dataloader.dataset.ann_file=$M/loeo/${F}_train.json

  echo "[$F] A4 eval on held-out event"
  $PY tools/eval_m1_split.py \
      work_dirs/m2_pf_rpn/loeo/$F/A4_add_change/m2_pf_rpn_cloud_full.py \
      work_dirs/m2_pf_rpn/loeo/$F/A4_add_change/epoch_8.pth \
      --split loeo_$F --batch-size 8 \
      --out "$OUT/A4_loeo_${F}.json" > /dev/null 2>&1 || echo "[$F] A4 eval failed"

  echo "[$F] YOLOv8s train"
  $PY /root/autodl-tmp/train_yolo_loeo.py /root/autodl-tmp/yolov8s.pt y8_loeo_$F 320 100 16 \
      /root/autodl-tmp/yolo_ds/all/data_$F.yaml

  echo "[$F] YOLOv8s export + eval"
  $PY /root/autodl-tmp/run_yolo_world.py \
      --weights /root/autodl-tmp/yolo_runs/y8_loeo_$F/weights/best.pt \
      --manifest $M/loeo_$F.json --data-root $HN \
      --out /root/autodl-tmp/baselines/y8_loeo_$F \
      --conf 0.001 --input-size 320 --rgb-mode percentile --device 0 > /dev/null 2>&1 || echo "[$F] export failed"
  $PY tools/eval_proposal_metrics.py /root/autodl-tmp/baselines/y8_loeo_$F $M/loeo_$F.json \
      --out "$OUT/y8_loeo_${F}_metrics.json" > /dev/null 2>&1 || echo "[$F] eval failed"
done

echo ""
echo "=== LOEO summary ==="
$PY - <<'PYEOF'
import json, os, glob
OUT = "/root/autodl-tmp/loeo"
folds = ["italy","kyrgyzstan1","newzealand","chimanimani","hiroshima","indonesia","hokkaido"]
print("| holdout | A4 AR@900 | YOLOv8s AR@900 |")
print("|---|---:|---:|")
a4s, y8s = [], []
for f in folds:
    try:
        a = json.load(open(os.path.join(OUT, "A4_loeo_%s.json" % f)))["AR"]["AR@900"]
    except Exception: a = float("nan")
    try:
        b = json.load(open(os.path.join(OUT, "y8_loeo_%s_metrics.json" % f)))["AR"]["AR@900"]
    except Exception: b = float("nan")
    a4s.append(a); y8s.append(b)
    print("| %s | %.4f | %.4f |" % (f, a, b))
import statistics as st
def m(v): 
    v=[x for x in v if x==x]
    return (sum(v)/len(v), st.pstdev(v)) if v else (float('nan'),0)
print("| **mean?sd** | **%.4f?%.4f** | **%.4f?%.4f** |" % (*m(a4s), *m(y8s)))
PYEOF
echo "=== LOEO DONE $(date) ==="

