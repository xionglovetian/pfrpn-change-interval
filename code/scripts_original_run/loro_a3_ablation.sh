#!/bin/bash
# loro_a3_ablation.sh -- cross-REGION mechanism control (companion to loeo_a3_ablation.sh).
# Trains A3 (Optical+SAR+DEM == A4 minus temporal change) on the 3 LORO folds so the
# "change modality dominates" claim also holds in the leave-one-REGION-out setting.
#
# Chains AFTER loeo_a3_ablation.sh: waits for its "LOEO-A3 ABLATION DONE" log line.
# (Does NOT modify the running script; no pgrep/pkill anywhere.)
#
# Instance deadline: 2026-10-14 12:17:12 CST. Internal hard stop: 2026-10-14 08:00.
set -u
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
OUT=/root/autodl-tmp/loro
LOG=/root/autodl-tmp/loro_a3.log
A3LOG=/root/autodl-tmp/loeo_a3.log
A3JSON=/root/autodl-tmp/loeo
HN=data/landslide_multimodal_hn_lb/
M=$PROJ/data/landslide_multimodal_hn_lb/manifests
BASE="train_dataloader.batch_size=8 optim_wrapper.optimizer.lr=0.00002"
FOLDS="EastAsia CentralAsia SEAsia"
DEADLINE=$(date -d '2026-10-14 08:00:00' +%s)

exec > >(tee -a "$LOG") 2>&1
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
mkdir -p "$OUT"

echo "=== LORO-A3 ABLATION START $(date) ==="
echo "    folds: $FOLDS"
echo "    hard stop: $(date -d @$DEADLINE '+%Y-%m-%d %H:%M:%S')"
echo "--- waiting for loeo_a3_ablation.sh to finish ---"
for _ in $(seq 1 1200); do
  if [ -f "$A3LOG" ] && grep -q "LOEO-A3 ABLATION DONE" "$A3LOG"; then
    echo "chain: LOEO-A3 DONE line found"; break
  fi
  # fallback: LOEO-A3 log stale >90min AND at least one fold produced => it is dead
  if [ -f "$A3LOG" ]; then
    now=$(date +%s); mt=$(stat -c %Y "$A3LOG")
    if [ $((now - mt)) -gt 5400 ] && ls "$A3JSON"/A3_loeo_*.json >/dev/null 2>&1; then
      echo "chain: LOEO-A3 log stale >90min with partial output -> proceeding"; break
    fi
  fi
  sleep 60
done

for F in $FOLDS; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "!! deadline guard hit before region $F -- stopping"; break
  fi
  echo ""
  echo "########## A3 LORO REGION $F  $(date) ##########"
  if [ -f "$OUT/A3_loro_${F}.json" ]; then
    echo "[$F] A3 result already exists, skipping"; continue
  fi

  echo "[$F] A3 train (no temporal change)"
  $PY tools/run_cloud_ablations.py A3_optical_sar_topography \
      --work-root work_dirs/m2_pf_rpn/loro/$F \
      --cfg-options $BASE train_dataloader.dataset.data_root=$HN \
        train_dataloader.dataset.ann_file=$M/loro/${F}_train.json \
        train_dataloader.persistent_workers=False

  echo "[$F] A3 eval on held-out region"
  $PY tools/eval_m1_split.py \
      work_dirs/m2_pf_rpn/loro/$F/A3_optical_sar_topography/m2_pf_rpn_cloud_full.py \
      work_dirs/m2_pf_rpn/loro/$F/A3_optical_sar_topography/epoch_8.pth \
      --split loro_$F --batch-size 8 \
      --out "$OUT/A3_loro_${F}.json" > /dev/null 2>&1 || echo "[$F] A3 eval FAILED"
  [ -f "$OUT/A3_loro_${F}.json" ] && touch "$OUT/A3_loro_${F}.done"
done

echo ""
echo "=== LORO-A3 summary (A3 no-change vs A4 +change, AR@900) ==="
$PY - <<'PYEOF'
import json
OUT = "/root/autodl-tmp/loro"
folds = ["EastAsia", "CentralAsia", "SEAsia"]
print("| held-out region | A3 (no change) | A4 (+change) | delta |")
print("|---|---:|---:|---:|")
d = []
for f in folds:
    try: a3 = json.load(open(f"{OUT}/A3_loro_{f}.json"))["AR"]["AR@900"]
    except Exception: a3 = float("nan")
    try: a4 = json.load(open(f"{OUT}/A4_loro_{f}.json"))["AR"]["AR@900"]
    except Exception: a4 = float("nan")
    if a3 == a3 and a4 == a4: d.append(a4 - a3)
    print("| %s | %.4f | %.4f | %+.4f |" % (f, a3, a4, a4 - a3))
if d:
    print("| **mean delta** | | | **%+.4f** |" % (sum(d)/len(d)))
PYEOF
echo "=== LORO-A3 ABLATION DONE $(date) ==="

