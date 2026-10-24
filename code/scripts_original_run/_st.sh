#!/bin/bash
cd /root/autodl-tmp || exit 1
date
echo "=== A3 完成度 ==="
echo "  LOEO: $(ls -1 loeo/A3_loeo_*.json 2>/dev/null | wc -l)/7"
echo "  LORO: $(ls -1 loro/A3_loro_*.json 2>/dev/null | wc -l)/3"
echo "=== 当前折 ==="
grep -a "FOLD" a3_all.log | tail -2
f=$(grep -a "FOLD" a3_all.log | tail -1 | sed 's/.*FOLD //;s/ .*//')
echo "  当前: $f"
for b in loeo loro; do
  L=PF-RPN/work_dirs/m2_pf_rpn/$b/$f/A3_optical_sar_topography/train.log
  [ -f "$L" ] && { echo "  epoch:"; grep -a "Epoch(train)" "$L" | tail -1 | cut -c1-110; }
done
echo "=== 队列脚本存活 ==="
ps -eo pid,etimes,cmd | grep -E "[a]3_all_ablation" | head -2
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader

