#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
LOG=$ROOT/logs/master.log
HN=data/landslide_multimodal_hn_lb/
POS=data/landslide_multimodal_lb_pos/
BASE="train_dataloader.batch_size=8 optim_wrapper.optimizer.lr=0.00002"
exec > >(tee -a "$LOG") 2>&1
cd "$PROJ"; export PYTHONPATH="$PROJ"

echo "=== MASTER START $(date) ==="

echo "=== PHASE 1/5: HN A0-A9 (positives + hard negatives) $(date) ==="
$PY tools/run_cloud_ablations.py \
  --work-root work_dirs/m2_pf_rpn/hn_lb \
  --cfg-options $BASE train_dataloader.dataset.data_root=$HN
echo "--- PHASE 1 train done $(date) ---"
$PY tools/eval_cloud_ablations.py \
  --work-root work_dirs/m2_pf_rpn/hn_lb \
  --metric-root artifacts/hn_lb_metrics \
  --proposal-root artifacts/proposals/hn_lb \
  --missing-modality --missing-split test \
  --cfg-options train_dataloader.dataset.data_root=$HN
$PY tools/report_cloud_results.py --summary artifacts/hn_lb_metrics/summary.json \
  --out-md docs/cloud_m1_results_hn_lb.md --out-dir artifacts/hn_lb_metrics/tables || true
echo "=== PHASE 1 DONE $(date) ==="

echo "=== PHASE 2/5: positive-only control A0/A4/A6 $(date) ==="
$PY tools/run_cloud_ablations.py A0_rgb A4_add_change A6_add_cross_modal_router \
  --work-root work_dirs/m2_pf_rpn/p_lb \
  --cfg-options $BASE train_dataloader.dataset.data_root=$POS
$PY tools/eval_cloud_ablations.py A0_rgb A4_add_change A6_add_cross_modal_router \
  --work-root work_dirs/m2_pf_rpn/p_lb \
  --metric-root artifacts/p_lb_metrics \
  --cfg-options train_dataloader.dataset.data_root=$POS
echo "=== PHASE 2 DONE $(date) ==="

echo "=== PHASE 3/5: optical-missing (A3/A6 + optical dropout) $(date) ==="
$PY tools/run_cloud_ablations.py A3_optical_sar_topography A6_add_cross_modal_router \
  --work-root work_dirs/m2_pf_rpn/od_lb \
  --cfg-options $BASE train_dataloader.dataset.data_root=$HN model.allow_optical_dropout=True
$PY tools/eval_cloud_ablations.py A3_optical_sar_topography A6_add_cross_modal_router \
  --work-root work_dirs/m2_pf_rpn/od_lb \
  --metric-root artifacts/od_lb_metrics \
  --missing-modality --missing-split test --include-optical-drops \
  --cfg-options train_dataloader.dataset.data_root=$HN model.allow_optical_dropout=True
$PY tools/report_cloud_results.py --summary artifacts/od_lb_metrics/summary.json \
  --out-md docs/cloud_m1_results_optical_missing.md --out-dir artifacts/od_lb_metrics/tables || true
echo "=== PHASE 3 DONE $(date) ==="

echo "=== PHASE 4/5: multi-seed A4/A5/A6/A7 $(date) ==="
for SEED in 123 777; do
  $PY tools/run_cloud_ablations.py A4_add_change A5_add_modality_token A6_add_cross_modal_router A7_no_router \
    --work-root work_dirs/m2_pf_rpn/hn_lb_seed$SEED \
    --cfg-options $BASE train_dataloader.dataset.data_root=$HN randomness.seed=$SEED
  $PY tools/eval_cloud_ablations.py A4_add_change A5_add_modality_token A6_add_cross_modal_router A7_no_router \
    --work-root work_dirs/m2_pf_rpn/hn_lb_seed$SEED \
    --metric-root artifacts/hn_lb_seed${SEED}_metrics \
    --cfg-options train_dataloader.dataset.data_root=$HN randomness.seed=$SEED
done
echo "=== PHASE 4 DONE $(date) ==="

echo "=== MASTER (phases 1-4) DONE $(date) ==="
df -h /root/autodl-tmp | tail -1