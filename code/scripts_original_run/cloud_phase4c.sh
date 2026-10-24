#!/bin/bash
# cloud_phase4c.sh - post-training threat diagnostics (roadmap 19.11):
#   1) pipeline audit with a saved report (leakage invariants)
#   2) change-modality shuffle sanity check on A4
#   3) router usage / entropy diagnostic on A6
# Waits for cloud_phase4b.sh so it never fights it for the GPU.
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
LOG=$ROOT/logs/phase4c.log
HN=data/landslide_multimodal_hn_lb/
CFG=configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py
BASE="train_dataloader.batch_size=8 optim_wrapper.optimizer.lr=0.00002"
exec > >(tee -a "$LOG") 2>&1
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
mkdir -p artifacts/diagnostics

echo "=== PHASE4C START $(date): waiting for phase4b ==="
for _ in $(seq 1 20160); do
  if grep -qa "PHASE4B DONE" "$ROOT/logs/phase4b.log" 2>/dev/null; then break; fi
  if ! pgrep -f "cloud_[p]hase4b\.sh" >/dev/null 2>&1; then
    sleep 180
    pgrep -f "cloud_[p]hase4b\.sh" >/dev/null 2>&1 || break
  fi
  sleep 60
done
echo "=== PHASE4C diagnostics $(date) ==="

echo "--- 1/3 pipeline audit ---"
$PY tools/audit_m1_pipeline.py $CFG > artifacts/diagnostics/audit_hn_lb.txt 2>&1
echo "audit exit=$?"
tail -6 artifacts/diagnostics/audit_hn_lb.txt

echo "--- 2/3 change-shuffle sanity on A4 (test) ---"
$PY tools/eval_m1_split.py $CFG \
  work_dirs/m2_pf_rpn/hn_lb/A4_add_change/epoch_8.pth \
  --split test --batch-size 8 --shuffle-modalities change \
  --out artifacts/diagnostics/A4_shuffle_change_test.json \
  --cfg-options $BASE train_dataloader.dataset.data_root=$HN
$PY tools/eval_m1_split.py $CFG \
  work_dirs/m2_pf_rpn/hn_lb/A4_add_change/epoch_8.pth \
  --split test --batch-size 8 \
  --out artifacts/diagnostics/A4_clean_test.json \
  --cfg-options $BASE train_dataloader.dataset.data_root=$HN

echo "--- 3/3 router usage diagnostic on A6 (val) ---"
$PY tools/diag_m1_router.py $CFG \
  work_dirs/m2_pf_rpn/hn_lb/A6_add_cross_modal_router/epoch_8.pth \
  --split val --max-images 96 --batch-size 8 \
  --out artifacts/diagnostics/router_A6.json \
  --cfg-options $BASE train_dataloader.dataset.data_root=$HN

$PY - <<'PYEOF'
import json, os
d = "artifacts/diagnostics"
def load(p, key):
    try:
        return json.load(open(os.path.join(d, p)))[key]["AR@900"]
    except Exception as e:
        return str(e)
print("A4 clean   test AR@900 =", load("A4_clean_test.json", "AR"))
print("A4 shuffle test AR@900 =", load("A4_shuffle_change_test.json", "AR"))
try:
    r = json.load(open(os.path.join(d, "router_A6.json")))
    print("router normalised entropy =", r["usage_normalized_entropy"],
          "| top expert x_uniform =", r["usage_max_over_uniform"])
except Exception as e:
    print("router diag:", e)
PYEOF
echo "=== PHASE4C DONE $(date) ==="
df -h /root/autodl-tmp | tail -1

