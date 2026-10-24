#!/bin/bash
cd /root/autodl-tmp || exit 1
echo "=== 1) landslide-framework/module2 里到底有什么（lb_neg 的来源）==="
find landslide-framework -maxdepth 3 -type f -name "*.py" | head -20
echo "  体量: $(du -sh landslide-framework 2>/dev/null | cut -f1)"
echo
echo "=== 2) mine_hard_negatives.py 的作用 ==="
head -25 landslide-framework/module2/src/mine_hard_negatives.py 2>/dev/null
echo
echo "=== 3) 环境规格（requirements.txt 内容）==="
cat PF-RPN/requirements.txt
echo "--- requirements/ ---"
for f in PF-RPN/requirements/*.txt; do echo "# $f"; cat "$f"; done
echo
echo "=== 4) 关键版本（云端实测）==="
/root/miniconda3/bin/python3 -c "
import sys, torch
print('python', sys.version.split()[0]); print('torch', torch.__version__); print('cuda', torch.version.cuda)
import mmcv, mmengine, mmdet
print('mmcv', mmcv.__version__); print('mmengine', mmengine.__version__); print('mmdet', mmdet.__version__)
"
echo
echo "=== 5) 顶层还有哪些「非 PF-RPN 内」但必需的脚本 ==="
ls -1 *.sh *.py 2>/dev/null | wc -l
echo "  其中与数据/折/基线直接相关的:"
ls -1 *.sh *.py 2>/dev/null | grep -iE "download|make_|prepare|mine|run_yolo|run_grounding|train_yolo|baseline" | head -20

