#!/bin/bash
# post_extra_evals.sh - runs AFTER the A0/A9 multi-seed training finishes.
#   1) change-channel perturbation (which change index carries the signal)
#   2) val paired statistics for seed 123 / 777
#   3) size-stratified summary for every external baseline vs Module 1
set -u
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
OUT=/root/autodl-tmp/baselines/analysis
LOG=/root/autodl-tmp/post_extra.log
exec > >(tee -a "$LOG") 2>&1
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
mkdir -p "$OUT"

echo "=== training finished, starting evals $(date) ==="

CFG=work_dirs/m2_pf_rpn/hn_lb/A4_add_change/m2_pf_rpn_cloud_full.py
CK=work_dirs/m2_pf_rpn/hn_lb/A4_add_change/epoch_8.pth

echo "--- 1/3 change-channel perturbation (A4, test) ---"
for i in 0 1 2; do
  echo "[zero change:$i]"
  $PY tools/eval_m1_split.py "$CFG" "$CK" --split test --batch-size 8 \
     --zero-model-channels change:$i \
     --out "$OUT/A4_zero_change_${i}_test.json" > /dev/null 2>&1 || echo "failed $i"
done

echo "--- 2/3 val paired stats (seed 123 / 777) ---"
for S in 123 777; do
  $PY tools/run_paired_stats.py --dump-root artifacts/hn_lb_seed${S}_metrics --split val \
    --pairs A4_add_change:A3_optical_sar_topography \
            A5_add_modality_token:A4_add_change \
            A6_add_cross_modal_router:A5_add_modality_token \
            A9_no_change:A6_add_cross_modal_router \
    --out "$OUT/paired_seed${S}_val.json" --out-md "$OUT/paired_seed${S}_val.md" || true
done

echo "--- 3/3 size-stratified summary ---"
$PY - <<'PYEOF'
import json, os, glob
OUT = "/root/autodl-tmp/baselines/analysis"
PF = "/root/autodl-tmp/PF-RPN/artifacts/hn_lb_metrics"
rows = []
def add(name, path, key="recall_by_size"):
    try:
        d = json.load(open(path))
        r = d.get(key, {})
        rows.append((name, d["AR"]["AR@100"], d["AR"]["AR@300"], d["AR"]["AR@900"],
                     r.get("small", 0), r.get("medium", 0), r.get("large", 0)))
    except Exception as e:
        rows.append((name + "(err)", 0,0,0,0,0,0))
add("M1/A0 (PF-RPN RGB)", f"{PF}/A0_rgb/test.json")
add("M1/A4 (ours)", f"{PF}/A4_add_change/test.json")
add("M1/A6 (ours)", f"{PF}/A6_add_cross_modal_router/test.json")
for n in ["y8_200e_test","y8_640b_test","y8_50e_test","yw_50e_test"]:
    add(n, f"/root/autodl-tmp/baselines/{n}_metrics.json")
add("GDINO zero-shot", "/root/autodl-tmp/baselines/gdino_zeroshot_test_landslide_metrics.json")
lines = ["| method | AR@100 | AR@300 | AR@900 | R-small | R-med | R-large |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    lines.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % r)
open(os.path.join(OUT, "size_stratified.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PYEOF

echo "=== POST_EXTRA DONE $(date) ==="

