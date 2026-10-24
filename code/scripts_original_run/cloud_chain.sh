#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
LOG=$ROOT/logs/chain.log
OUT=$PROJ/data/landslide_multimodal_full

exec > >(tee -a "$LOG") 2>&1
echo "=== CHAIN START $(date) ==="

# gate
$PY - <<PYEOF
import json, pathlib
root = pathlib.Path("$OUT/manifests")
rep = json.loads((root / "split_report.json").read_text())
train = json.loads((root / "train.json").read_text())
assert rep.get("event_level_passed"), "event-level split not passed"
print("train/val/test:", len(train),
      len(json.loads((root / "val.json").read_text())),
      len(json.loads((root / "test.json").read_text())))
print("split events:", {k: len(v) for k, v in
      json.loads((root / "split_manifest.json").read_text())["splits"].items()})
PYEOF

cd "$PROJ"
export PYTHONPATH="$PROJ"
rm -rf work_dirs/m2_pf_rpn/probe_A0

echo "=== A0-A9 TRAIN $(date) ==="
$PY tools/run_cloud_ablations.py
echo "=== A0-A9 TRAIN DONE $(date) ==="

echo "=== EVAL $(date) ==="
$PY tools/eval_cloud_ablations.py --missing-modality --missing-split test
echo "=== EVAL DONE $(date) ==="

echo "=== SUMMARY TABLE $(date) ==="
$PY tools/summarize_cloud_ablations.py --split-prefix test || true
$PY tools/summarize_cloud_ablations.py --split-prefix val || true
echo "=== CHAIN DONE $(date) ==="
df -h /root/autodl-tmp