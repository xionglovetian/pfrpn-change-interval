#!/bin/bash
cd /root/autodl-tmp || exit 1
echo "=== 1) tools/data/ 内容（数据构建链）==="
ls -la PF-RPN/tools/data/
echo
echo "=== 2) prepare_sen12landslides.py 做什么（前 40 行 + 参数）==="
head -40 PF-RPN/tools/data/prepare_sen12landslides.py
echo "  ..."
grep -n "add_argument\|def \|lb_pos\|lb_neg\|hn_lb\|manifests" PF-RPN/tools/data/prepare_sen12landslides.py | head -30
echo
echo "=== 3) landslide-framework 是否独立于 PF-RPN ==="
ls -d PF-RPN/landslide-framework 2>/dev/null && echo "  在 PF-RPN 内" || echo "  **在 PF-RPN 之外（独立目录）**"
ls -1 landslide-framework/ 2>/dev/null | head
echo
echo "=== 4) 折划分脚本 ==="
ls -1 PF-RPN/tools/ | grep -iE "fold|split|loeo|loro" 
echo "  顶层:"; ls -1 *.py | grep -iE "fold|split|loeo|loro"
echo
echo "=== 5) manifests 里点的路径能否自洽 ==="
/root/miniconda3/bin/python3 - <<PY
import json,os
p="PF-RPN/data/landslide_multimodal_hn_lb/manifests/train.json"
d=json.load(open(p))
print("  train entries:",len(d))
print("  sample change path:",d[0]["modalities"]["change"])
base="PF-RPN/data/landslide_multimodal_hn_lb"
tgt=os.path.normpath(os.path.join(base,d[0]["modalities"]["change"]))
print("  resolves to:",tgt)
print("  exists:",os.path.exists(tgt))
PY

