#!/bin/bash
# loeo_a3_ablation.sh -- cross-event MECHANISM CONTROL.
# Trains A3 (Optical+SAR+DEM == A4 with temporal change REMOVED) on LOEO folds so the
# "change modality dominates" claim is established in the cross-event setting, not
# only on the main event-level split.
#
# Instance deadline: 2026-10-14 12:17:12 CST  (corrected by user on 2026-10-12)
# Internal hard stop: 2026-10-14 08:00  -> ~4h buffer for packaging + pull-back.
#
# Red lines honoured: waits on a MARKER FILE (never pgrep/pkill); per-fold marker
# so the run is idempotent/resumable; deadline guard before every fold.
set -u
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
OUT=/root/autodl-tmp/loeo
LOG=/root/autodl-tmp/loeo_a3.log
LORO_DONE=/root/autodl-tmp/loro/LORO_DONE
LORO_LOG=/root/autodl-tmp/loro.log
HN=data/landslide_multimodal_hn_lb/
M=$PROJ/data/landslide_multimodal_hn_lb/manifests
BASE="train_dataloader.batch_size=8 optim_wrapper.optimizer.lr=0.00002"
# all 7 LOEO folds, ordered so a partial run still spans the AR@900 range
FOLDS="italy newzealand indonesia hiroshima hokkaido chimanimani kyrgyzstan1"
DEADLINE=$(date -d '2026-10-14 08:00:00' +%s)

exec > >(tee -a "$LOG") 2>&1
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"

echo "=== LOEO-A3 ABLATION START $(date) ==="
echo "    folds: $FOLDS"
echo "    hard stop: $(date -d @$DEADLINE '+%Y-%m-%d %H:%M:%S')"
echo "--- waiting for LORO to finish (marker: $LORO_DONE) ---"
for _ in $(seq 1 400); do
  [ -f "$LORO_DONE" ] && { echo "chain: LORO_DONE found"; break; }
  if [ -f "$LORO_LOG" ]; then
    now=$(date +%s); mt=$(stat -c %Y "$LORO_LOG")
    if [ $((now - mt)) -gt 1800 ]; then echo "chain: LORO log stale >30min -> proceeding"; break; fi
  fi
  sleep 60
done

for F in $FOLDS; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "!! deadline guard hit before fold $F -- stopping"; break
  fi
  echo ""
  echo "########## A3 LOEO FOLD $F  $(date) ##########"
  if [ -f "$OUT/A3_loeo_${F}.json" ]; then
    echo "[$F] A3 result already exists, skipping"; continue
  fi

  echo "[$F] A3 train (no temporal change)"
  $PY tools/run_cloud_ablations.py A3_optical_sar_topography \
      --work-root work_dirs/m2_pf_rpn/loeo/$F \
      --cfg-options $BASE train_dataloader.dataset.data_root=$HN \
        train_dataloader.dataset.ann_file=$M/loeo/${F}_train.json

  echo "[$F] A3 eval on held-out event"
  $PY tools/eval_m1_split.py \
      work_dirs/m2_pf_rpn/loeo/$F/A3_optical_sar_topography/m2_pf_rpn_cloud_full.py \
      work_dirs/m2_pf_rpn/loeo/$F/A3_optical_sar_topography/epoch_8.pth \
      --split loeo_$F --batch-size 8 \
      --out "$OUT/A3_loeo_${F}.json" > /dev/null 2>&1 || echo "[$F] A3 eval FAILED"
  [ -f "$OUT/A3_loeo_${F}.json" ] && touch "$OUT/A3_loeo_${F}.done"
done

echo ""
echo "=== LOEO-A3 summary (A3 free of change vs A4 with change, AR@900) ==="
$PY - <<'PYEOF'
import json
OUT = "/root/autodl-tmp/loeo"
folds = ["italy","newzealand","indonesia","hiroshima","hokkaido","chimanimani","kyrgyzstan1"]
print("| fold | A3 (no change) | A4 (+change) | delta |")
print("|---|---:|---:|---:|")
d = []
for f in folds:
    try: a3 = json.load(open(f"{OUT}/A3_loeo_{f}.json"))["AR"]["AR@900"]
    except Exception: a3 = float("nan")
    try: a4 = json.load(open(f"{OUT}/A4_loeo_{f}.json"))["AR"]["AR@900"]
    except Exception: a4 = float("nan")
    if a3 == a3 and a4 == a4: d.append(a4 - a3)
    print("| %s | %.4f | %.4f | %+.4f |" % (f, a3, a4, a4 - a3))
if d:
    print("| **mean delta** | | | **%+.4f** |" % (sum(d)/len(d)))
PYEOF
echo "=== LOEO-A3 ABLATION DONE $(date) ==="

