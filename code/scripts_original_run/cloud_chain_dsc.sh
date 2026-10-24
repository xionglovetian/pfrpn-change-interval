#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
LOG=$ROOT/logs/chain_dsc.log
exec > >(tee -a "$LOG") 2>&1

echo "=== DSC CHAIN START (waiting for OD) $(date) ==="
for i in $(seq 1 5760); do
  grep -q "OD CHAIN DONE" "$ROOT/logs/chain_od.log" 2>/dev/null && break
  sleep 60
done
grep -q "OD CHAIN DONE" "$ROOT/logs/chain_od.log" || { echo "DSC WAIT TIMEOUT $(date)"; exit 1; }
echo "=== OD DONE -> DUAL-ORBIT SAR PHASE $(date) ==="

cd "$PROJ"
export PYTHONPATH="$PROJ"

echo "=== DSC CONVERT positives (asc+dsc) $(date) ==="
$PY tools/data/prepare_sen12landslides.py "$ROOT" \
  --task S12LS-LD --version harmonized \
  --splits-file "$ROOT/tasks/S12LS-LD/harmonized/splits_aligned.json" \
  --out data/landslide_multimodal_dsc_pos --sar-source both || exit 1

echo "=== DSC CONVERT negatives (asc+dsc) $(date) ==="
$PY tools/data/prepare_sen12landslides.py "$ROOT" \
  --task S12LS-AD --version harmonized \
  --splits-file "$ROOT/tasks/ad_splits.json" \
  --out data/landslide_multimodal_dsc_neg --sar-source both \
  --skip-annotated || exit 1

echo "=== DSC MERGE $(date) ==="
$PY tools/data/merge_positive_negative.py \
  --positive-root data/landslide_multimodal_dsc_pos \
  --negative-root data/landslide_multimodal_dsc_neg \
  --out-root data/landslide_multimodal_dsc \
  --positive-prefix ../landslide_multimodal_dsc_pos \
  --negative-prefix ../landslide_multimodal_dsc_neg \
  --drop-inventories lanaodelnorte || exit 1

echo "=== DSC STATS (19 channels) $(date) ==="
$PY tools/data/compute_sen12_stats.py data/landslide_multimodal_dsc || exit 1

echo "=== DSC A0-A9 TRAIN $(date) ==="
$PY tools/run_cloud_ablations.py \
  --config configs/m2-pf-rpn/m2_pf_rpn_cloud_dsc.py \
  --work-root work_dirs/m2_pf_rpn/cloud_dsc
echo "=== DSC TRAIN DONE $(date) ==="

echo "=== DSC EVAL $(date) ==="
$PY tools/eval_cloud_ablations.py \
  --config configs/m2-pf-rpn/m2_pf_rpn_cloud_dsc.py \
  --work-root work_dirs/m2_pf_rpn/cloud_dsc \
  --metric-root artifacts/cloud_dsc_metrics \
  --proposal-root artifacts/proposals/cloud_dsc \
  --missing-modality --missing-split test
echo "=== DSC EVAL DONE $(date) ==="

$PY tools/report_cloud_results.py \
  --summary artifacts/cloud_dsc_metrics/summary.json \
  --out-md docs/cloud_m1_results_dsc.md \
  --out-dir artifacts/cloud_dsc_metrics/tables || true
echo "=== DSC CHAIN DONE $(date) ==="