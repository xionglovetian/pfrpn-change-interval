#!/bin/bash
# cloud_phase4b.sh - extend Phase 4 with A3 and produce per-image dumps +
# paired statistics (RESEARCH_ROADMAP.md 19.5 / 19.7).  Waits for the running
# master to finish so it never fights it for the GPU.
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
LOG=$ROOT/logs/phase4b.log
HN=data/landslide_multimodal_hn_lb/
BASE="train_dataloader.batch_size=8 optim_wrapper.optimizer.lr=0.00002"
RUNS="A3_optical_sar_topography A4_add_change A5_add_modality_token A6_add_cross_modal_router A7_no_router"
SEED42="A0_rgb A3_optical_sar_topography A4_add_change A5_add_modality_token A6_add_cross_modal_router A7_no_router A9_no_change"
exec > >(tee -a "$LOG") 2>&1
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"

echo "=== PHASE4B START $(date): waiting for master phases 1-4 ==="
for _ in $(seq 1 20160); do          # up to 14 days, 60s ticks
  if grep -qa "MASTER (phases 1-4) DONE" "$ROOT/logs/master.log" 2>/dev/null; then
    break
  fi
  if ! pgrep -f "cloud_[m]aster\.sh" >/dev/null 2>&1; then
    sleep 180
    pgrep -f "cloud_[m]aster\.sh" >/dev/null 2>&1 || break
  fi
  sleep 60
done
echo "=== PHASE4B master finished (or gone) $(date) ==="

echo "=== STAGE A: A3 multi-seed training + eval (seeds 123/777) $(date) ==="
for SEED in 123 777; do
  $PY tools/run_cloud_ablations.py $RUNS \
    --work-root work_dirs/m2_pf_rpn/hn_lb_seed$SEED \
    --cfg-options $BASE train_dataloader.dataset.data_root=$HN randomness.seed=$SEED
  $PY tools/eval_cloud_ablations.py $RUNS \
    --work-root work_dirs/m2_pf_rpn/hn_lb_seed$SEED \
    --metric-root artifacts/hn_lb_seed${SEED}_metrics \
    --dump-per-image \
    --cfg-options $BASE train_dataloader.dataset.data_root=$HN randomness.seed=$SEED
  $PY tools/report_cloud_results.py --summary artifacts/hn_lb_seed${SEED}_metrics/summary.json \
    --out-md docs/cloud_m1_results_seed${SEED}.md \
    --out-dir artifacts/hn_lb_seed${SEED}_metrics/tables || true
  $PY tools/run_paired_stats.py \
    --dump-root artifacts/hn_lb_seed${SEED}_metrics --split test \
    --pairs A4_add_change:A3_optical_sar_topography \
            A5_add_modality_token:A4_add_change \
            A6_add_cross_modal_router:A5_add_modality_token \
            A7_no_router:A6_add_cross_modal_router \
            A9_no_change:A6_add_cross_modal_router \
    --out artifacts/hn_lb_seed${SEED}_metrics/paired_test.json \
    --out-md docs/paired_stats_seed${SEED}_test.md || true
done
echo "=== STAGE A done $(date) ==="

echo "=== STAGE B: seed-42 per-image dumps for paired stats $(date) ==="
$PY tools/eval_cloud_ablations.py $SEED42 \
  --work-root work_dirs/m2_pf_rpn/hn_lb \
  --metric-root artifacts/hn_lb_paired \
  --dump-per-image --splits test val \
  --cfg-options $BASE train_dataloader.dataset.data_root=$HN
for SPLIT in test val; do
  $PY tools/run_paired_stats.py \
    --dump-root artifacts/hn_lb_paired --split $SPLIT \
    --pairs A4_add_change:A3_optical_sar_topography \
            A5_add_modality_token:A4_add_change \
            A6_add_cross_modal_router:A5_add_modality_token \
            A7_no_router:A6_add_cross_modal_router \
            A9_no_change:A6_add_cross_modal_router \
            A6_add_cross_modal_router:A3_optical_sar_topography \
    --out artifacts/hn_lb_paired/paired_${SPLIT}.json \
    --out-md docs/paired_stats_seed42_${SPLIT}.md || true
done
echo "=== PHASE4B DONE $(date) ==="
df -h /root/autodl-tmp | tail -1

