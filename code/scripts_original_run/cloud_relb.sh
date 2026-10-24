#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
LOG=$ROOT/logs/relb.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=== REBUILD (uniform long-baseline change) START $(date) ==="
cd "$PROJ"
export PYTHONPATH="$PROJ"

echo "=== 1/5 convert LD positives -> landslide_multimodal_lb_pos $()==="
$PY tools/data/prepare_sen12landslides.py "$ROOT" \
  --task S12LS-LD --version harmonized \
  --splits-file "$ROOT/tasks/S12LS-LD/harmonized/splits_aligned.json" \
  --out data/landslide_multimodal_lb_pos --sar-source asc \
  --pre-post-rule long-baseline || { echo "FAIL convert pos"; exit 1; }

echo "=== 2/5 convert AD negatives -> landslide_multimodal_lb_neg ==="
$PY tools/data/prepare_sen12landslides.py "$ROOT" \
  --task S12LS-AD --version harmonized \
  --splits-file "$ROOT/tasks/ad_splits.json" \
  --out data/landslide_multimodal_lb_neg --sar-source asc \
  --pre-post-rule long-baseline --skip-annotated || { echo "FAIL convert neg"; exit 1; }

echo "=== 3/5 event-level split on the positives ==="
mkdir -p data/landslide_multimodal_lb_pos/manifests/official
for s in train val test; do
  cp -f data/landslide_multimodal_lb_pos/manifests/$s.json data/landslide_multimodal_lb_pos/manifests/official/$s.json
done
$PY - <<PYEOF
import json, pathlib
root = pathlib.Path("data/landslide_multimodal_lb_pos/manifests")
items = []
for s in ("train","val","test"):
    items += json.loads((root/"official"/f"{s}.json").read_text())
(root/"all_official.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
print("combined official:", len(items))
PYEOF
$PY tools/data/create_event_split.py \
  data/landslide_multimodal_lb_pos/manifests/all_official.json \
  data/landslide_multimodal_lb_pos/manifests \
  --strategy balance --ratios 0.7 0.15 0.15 --seed 42 || { echo "FAIL split"; exit 1; }
$PY tools/data/validate_event_split.py data/landslide_multimodal_lb_pos/manifests \
  --output data/landslide_multimodal_lb_pos/manifests/split_report.json || true

echo "=== 4/5 merge positives + hard negatives ==="
$PY tools/data/merge_positive_negative.py \
  --positive-root data/landslide_multimodal_lb_pos \
  --negative-root data/landslide_multimodal_lb_neg \
  --out-root data/landslide_multimodal_hn_lb \
  --positive-prefix ../landslide_multimodal_lb_pos \
  --negative-prefix ../landslide_multimodal_lb_neg \
  --drop-inventories lanaodelnorte || { echo "FAIL merge"; exit 1; }

echo "=== 5/5 stats on the merged set, shared by the positive-only control ==="
$PY tools/data/compute_sen12_stats.py data/landslide_multimodal_hn_lb || { echo "FAIL stats"; exit 1; }
cp -f data/landslide_multimodal_hn_lb/norm_stats.json data/landslide_multimodal_lb_pos/norm_stats.json
cp -f data/landslide_multimodal_hn_lb/norm_stats.json data/landslide_multimodal_lb_neg/norm_stats.json

echo "=== REBUILD DONE $(date) ==="
df -h /root/autodl-tmp | tail -1