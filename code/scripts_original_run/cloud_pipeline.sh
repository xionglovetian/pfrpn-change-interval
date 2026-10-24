#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
D=$ROOT/data/data_harmonized
OUT=$PROJ/data/landslide_multimodal_full
LOG=$ROOT/logs/pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "=== PIPELINE START $(date) ==="
# wait for extraction to finish
for i in $(seq 1 1200); do
  if grep -q "EXTRACT_DONE" "$ROOT/logs/extract.log" 2>/dev/null; then break; fi
  sleep 20
done
grep -q "EXTRACT_DONE" "$ROOT/logs/extract.log" || { echo "EXTRACT TIMEOUT"; exit 1; }
echo "s2 nc: $(ls $D/s2/*.nc 2>/dev/null | wc -l)  s1asc nc: $(ls $D/s1asc/*.nc 2>/dev/null | wc -l)"
df -h /root/autodl-tmp | tail -1

cd "$PROJ"
mkdir -p "$OUT"

echo "=== STEP 1: convert Sen12Landslides -> M1 npy $(date) ==="
$PY tools/data/prepare_sen12landslides.py "$ROOT" \
  --task S12LS-LD --version harmonized \
  --splits-file "$ROOT/tasks/S12LS-LD/harmonized/splits_aligned.json" \
  --out "$OUT" --sar-source asc
echo "convert exit=$?"

echo "=== STEP 2: normalisation stats $(date) ==="
$PY tools/data/compute_sen12_stats.py "$OUT"

echo "=== STEP 3: keep official manifests, build combined manifest $(date) ==="
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

echo "=== STEP 4: event-level split $(date) ==="
$PY tools/data/create_event_split.py \
  "$OUT/manifests/all_official.json" "$OUT/manifests" \
  --ratios 0.7 0.15 0.15 --seed 42

echo "=== STEP 5: validate event-level split $(date) ==="
$PY tools/data/validate_event_split.py "$OUT/manifests" \
  --output "$OUT/manifests/split_report.json"

echo "=== STEP 6: patch cloud config with real stats is automatic (config reads norm_stats.json) ==="
cat "$OUT/norm_stats.json" | head -5
echo "=== PIPELINE DONE $(date) ==="