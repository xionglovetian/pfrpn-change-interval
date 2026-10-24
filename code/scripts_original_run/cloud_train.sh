#!/bin/bash
set -u
ROOT=/root/autodl-tmp/Sen12Landslides
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
PLOG=$ROOT/logs/pipeline.log
TLOG=$ROOT/logs/train.log
OUT=$PROJ/data/landslide_multimodal_full

echo "=== TRAIN WAIT START $(date) ===" >> "$TLOG"
for i in $(seq 1 900); do
  grep -q "PIPELINE DONE" "$PLOG" 2>/dev/null && break
  if grep -q "PIPELINE FAILED\|EXTRACT TIMEOUT" "$PLOG" 2>/dev/null; then
    echo "PIPELINE FAILED, aborting" >> "$TLOG"; exit 1
  fi
  sleep 30
done
grep -q "PIPELINE DONE" "$PLOG" || { echo "TRAIN WAIT TIMEOUT" >> "$TLOG"; exit 1; }

echo "=== GATE CHECKS $(date) ===" >> "$TLOG"
$PY - <<PYEOF >> "$TLOG" 2>&1
import json, pathlib
root = pathlib.Path("$OUT/manifests")
ok = True
for s in ("train", "val", "test"):
    p = root / f"{s}.json"
    if not p.exists():
        print(f"MISSING {p}"); ok = False; continue
    items = json.loads(p.read_text(encoding="utf-8"))
    print(f"{s}: {len(items)} samples")
rep = root / "split_report.json"
if rep.exists():
    r = json.loads(rep.read_text(encoding="utf-8"))
    print("event_level_passed:", r.get("event_level_passed"))
    ok = ok and bool(r.get("event_level_passed"))
else:
    print("MISSING split_report.json"); ok = False
train = json.loads((root / "train.json").read_text(encoding="utf-8")) if (root / "train.json").exists() else []
print("train samples:", len(train))
if len(train) < 200: ok = False
print("GATE_OK" if ok else "GATE_FAIL")
PYEOF
if ! grep -q "GATE_OK" "$TLOG"; then
  echo "GATE FAILED, not starting training" >> "$TLOG"; exit 1
fi

echo "=== START A0-A9 $(date) ===" >> "$TLOG"
cd "$PROJ"
export PYTHONPATH="$PROJ"
$PY tools/run_cloud_ablations.py >> "$TLOG" 2>&1
echo "=== A0-A9 DONE $(date) ===" >> "$TLOG"