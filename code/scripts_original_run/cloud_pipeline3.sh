#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
D=$ROOT/data/data_harmonized
OUT=$PROJ/data/landslide_multimodal_full
LOG=$ROOT/logs/pipeline3.log
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "=== PIPELINE3 START $(date) ==="
s2n=$(ls $D/s2/*.nc 2>/dev/null | wc -l)
s1n=$(ls $D/s1asc/*.nc 2>/dev/null | wc -l)
echo "s2 nc: $s2n   s1asc nc: $s1n"
[ "$s2n" -gt 1000 ] || { echo "PIPELINE3 FAILED too few s2 nc"; exit 1; }
[ "$s1n" -gt 1000 ] || { echo "PIPELINE3 FAILED too few s1asc nc"; exit 1; }
df -h /root/autodl-tmp | tail -1

cd "$PROJ"
mkdir -p "$OUT"

echo "=== STEP 1 convert $(date) ==="
$PY tools/data/prepare_sen12landslides.py "$ROOT" \
  --task S12LS-LD --version harmonized \
  --splits-file "$ROOT/tasks/S12LS-LD/harmonized/splits_aligned.json" \
  --out "$OUT" --sar-source asc
echo "convert exit=$?"

echo "=== STEP 2 stats $(date) ==="
$PY tools/data/compute_sen12_stats.py "$OUT"

echo "=== STEP 3 combined official manifest $(date) ==="
mkdir -p "$OUT/manifests/official"
for s in train val test; do
  cp -f "$OUT/manifests/$s.json" "$OUT/manifests/official/$s.json" 2>/dev/null || true
done
$PY - <<PYEOF
import json, pathlib
root = pathlib.Path("$OUT/manifests")
items = []
for s in ("train", "val", "test"):
    p = root / "official" / f"{s}.json"
    if p.exists():
        items.extend(json.loads(p.read_text(encoding="utf-8")))
(root / "all_official.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
print("combined official items:", len(items))
PYEOF

echo "=== STEP 4 event-level split $(date) ==="
$PY tools/data/create_event_split.py \
  "$OUT/manifests/all_official.json" "$OUT/manifests" \
  --strategy balance --ratios 0.7 0.15 0.15 --seed 42

echo "=== STEP 5 validate $(date) ==="
$PY tools/data/validate_event_split.py "$OUT/manifests" \
  --output "$OUT/manifests/split_report.json"

echo "=== PIPELINE DONE $(date) ==="