#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
LOG=$ROOT/logs/chain_hn.log
exec > >(tee -a "$LOG") 2>&1

echo "=== HN CHAIN START (waiting for P set) $(date) ==="
for i in $(seq 1 2880); do
  grep -q "CHAIN DONE" "$ROOT/logs/chain.log" 2>/dev/null && break
  sleep 60
done
grep -q "CHAIN DONE" "$ROOT/logs/chain.log" || { echo "HN WAIT TIMEOUT $(date)"; exit 1; }
echo "=== P SET DONE -> HN TRAIN $(date) ==="

cd "$PROJ"
export PYTHONPATH="$PROJ"

$PY - <<PYEOF
import json, pathlib
root = pathlib.Path("data/landslide_multimodal_hn/manifests")
ok = True
for s in ("train", "val", "test"):
    items = json.loads((root / f"{s}.json").read_text())
    neg = sum(1 for it in items if not it["labels"]["annotations"])
    print(s, len(items), "neg", neg)
    ok = ok and neg > 0
print("HN_GATE_OK" if ok else "HN_GATE_FAIL")
PYEOF

echo "=== HN A0-A9 TRAIN $(date) ==="
$PY tools/run_cloud_ablations.py \
  --work-root work_dirs/m2_pf_rpn/cloud_hn \
  --cfg-options train_dataloader.dataset.data_root=data/landslide_multimodal_hn/
echo "=== HN A0-A9 TRAIN DONE $(date) ==="

echo "=== HN EVAL + MISSING MODALITY + EXPORT $(date) ==="
$PY tools/eval_cloud_ablations.py \
  --work-root work_dirs/m2_pf_rpn/cloud_hn \
  --metric-root artifacts/cloud_hn_metrics \
  --proposal-root artifacts/proposals/cloud_hn \
  --missing-modality --missing-split test \
  --cfg-options train_dataloader.dataset.data_root=data/landslide_multimodal_hn/
echo "=== HN EVAL DONE $(date) ==="

echo "=== HN REPORT $(date) ==="
$PY tools/report_cloud_results.py \
  --summary artifacts/cloud_hn_metrics/summary.json \
  --out-md docs/cloud_m1_results_hn.md \
  --out-dir artifacts/cloud_hn_metrics/tables || true

echo "=== HN CHAIN DONE $(date) ==="
df -h /root/autodl-tmp