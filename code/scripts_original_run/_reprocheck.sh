#!/bin/bash
cd /root/autodl-tmp/PF-RPN || exit 1
echo "=== A) 硬编码 /root/autodl-tmp 在「要上传的代码」里的分布 ==="
for d in tools configs mmdet tests schemas; do
  n=$(grep -rl "/root/autodl-tmp" --include="*.py" --include="*.sh" --include="*.cfg" "$d" 2>/dev/null | wc -l)
  echo "   $d : $n files"
done
echo "--- 明细（前 15 行）---"
grep -rn "/root/autodl-tmp" --include="*.py" --include="*.sh" --include="*.cfg" tools configs 2>/dev/null | head -15
echo
echo "=== B) 数据集构建脚本 ==="
ls -1 tools/dataset_converters/ 2>/dev/null || echo "   (no dataset_converters)"
echo
echo "=== C) 基线脚本位置 ==="
ls -1 /root/autodl-tmp/*.py 2>/dev/null | head -25
echo "--- PF-RPN/tools/baselines ---"
ls -1 tools/baselines 2>/dev/null || echo "   (不存在)"
echo
echo "=== D) README / 复现说明 ==="
ls -1 *.md docs/*.md 2>/dev/null | head -20

