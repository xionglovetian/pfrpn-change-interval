#!/bin/bash
# loro_run.sh - leave-one-REGION-out: 3 folds (East Asia / Central Asia / SE Asia).
set -u
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
OUT=/root/autodl-tmp/loro
LOG=/root/autodl-tmp/loro.log
HN=data/landslide_multimodal_hn_lb/
M=$PROJ/data/landslide_multimodal_hn_lb/manifests
BASE="train_dataloader.batch_size=8 optim_wrapper.optimizer.lr=0.00002"
FOLDS="EastAsia CentralAsia SEAsia"
exec > >(tee -a "$LOG") 2>&1
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
mkdir -p "$OUT"

echo "=== LORO START $(date) ==="
$PY /root/autodl-tmp/make_loro_folds.py

for F in $FOLDS; do
  echo ""
  echo "########## REGION $F  $(date) ##########"
  echo "[$F] A4 train"
  $PY tools/run_cloud_ablations.py A4_add_change \
      --work-root work_dirs/m2_pf_rpn/loro/$F \
      --cfg-options $BASE train_dataloader.dataset.data_root=$HN \
        train_dataloader.dataset.ann_file=$M/loro/${F}_train.json train_dataloader.persistent_workers=False

  echo "[$F] A4 eval on held-out region"
  $PY tools/eval_m1_split.py \
      work_dirs/m2_pf_rpn/loro/$F/A4_add_change/m2_pf_rpn_cloud_full.py \
      work_dirs/m2_pf_rpn/loro/$F/A4_add_change/epoch_8.pth \
      --split loro_$F --batch-size 8 \
      --out "$OUT/A4_loro_${F}.json" > /dev/null 2>&1 || echo "[$F] A4 eval failed"

  echo "[$F] YOLOv8s train"
  $PY /root/autodl-tmp/train_yolo_loeo.py /root/autodl-tmp/yolov8s.pt y8_loro_$F 320 100 16 \
      /root/autodl-tmp/yolo_ds/all/data_loro_$F.yaml

  echo "[$F] YOLOv8s export + eval"
  $PY /root/autodl-tmp/run_yolo_world.py \
      --weights /root/autodl-tmp/yolo_runs/y8_loro_$F/weights/best.pt \
      --manifest $M/loro_$F.json --data-root $HN \
      --out /root/autodl-tmp/baselines/y8_loro_$F \
      --conf 0.001 --input-size 320 --rgb-mode percentile --device 0 > /dev/null 2>&1 || echo "[$F] export failed"
  $PY tools/eval_proposal_metrics.py /root/autodl-tmp/baselines/y8_loro_$F $M/loro_$F.json \
      --out "$OUT/y8_loro_${F}_metrics.json" > /dev/null 2>&1 || echo "[$F] eval failed"
done

echo ""
echo "=== LORO summary ==="
$PY - <<'PYEOF'
import json, os, statistics as st
OUT = "/root/autodl-tmp/loro"
folds = ["EastAsia","CentralAsia","SEAsia"]
print("| held-out region | A4 AR@900 | YOLOv8s AR@900 |")
print("|---|---:|---:|")
a4, y8 = [], []
for f in folds:
    try: a = json.load(open(f"{OUT}/A4_loro_{f}.json"))["AR"]["AR@900"]
    except Exception: a = float("nan")
    try: b = json.load(open(f"{OUT}/y8_loro_{f}_metrics.json"))["AR"]["AR@900"]
    except Exception: b = float("nan")
    a4.append(a); y8.append(b)
    print("| %s | %.4f | %.4f |" % (f, a, b))
def m(v):
    v=[x for x in v if x==x]
    return (sum(v)/len(v), st.pstdev(v)) if v else (float("nan"),0.0)
print("| **mean?sd** | **%.4f?%.4f** | **%.4f?%.4f** |" % (*m(a4), *m(y8)))
PYEOF
touch "$OUT/LORO_DONE"
echo "=== LORO DONE $(date) ==="

