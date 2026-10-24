#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
LOG=$ROOT/logs/chain_od.log
exec > >(tee -a "$LOG") 2>&1

echo "=== OD CHAIN START (waiting for HN) $(date) ==="
for i in $(seq 1 5760); do
  grep -q "HN CHAIN DONE" "$ROOT/logs/chain_hn.log" 2>/dev/null && break
  sleep 60
done
grep -q "HN CHAIN DONE" "$ROOT/logs/chain_hn.log" || { echo "OD WAIT TIMEOUT $(date)"; exit 1; }
echo "=== HN DONE -> OPTICAL-MISSING PHASE $(date) ==="

cd "$PROJ"
export PYTHONPATH="$PROJ"

echo "=== OD TRAIN (allow_optical_dropout=True) $(date) ==="
$PY tools/run_cloud_ablations.py A3_optical_sar_topography A6_add_cross_modal_router \
  --work-root work_dirs/m2_pf_rpn/cloud_hn_od \
  --cfg-options train_dataloader.dataset.data_root=data/landslide_multimodal_hn/ \
                model.allow_optical_dropout=True
echo "=== OD TRAIN DONE $(date) ==="

echo "=== OD EVAL (incl. optical drops) $(date) ==="
$PY tools/eval_cloud_ablations.py A3_optical_sar_topography A6_add_cross_modal_router \
  --work-root work_dirs/m2_pf_rpn/cloud_hn_od \
  --metric-root artifacts/cloud_hn_od_metrics \
  --proposal-root artifacts/proposals/cloud_hn_od \
  --missing-modality --missing-split test --include-optical-drops \
  --cfg-options train_dataloader.dataset.data_root=data/landslide_multimodal_hn/ \
                model.allow_optical_dropout=True
echo "=== OD EVAL DONE $(date) ==="

$PY tools/report_cloud_results.py \
  --summary artifacts/cloud_hn_od_metrics/summary.json \
  --out-md docs/cloud_m1_results_od.md \
  --out-dir artifacts/cloud_hn_od_metrics/tables || true
echo "=== OD CHAIN DONE $(date) ==="