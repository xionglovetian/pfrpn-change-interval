#!/bin/bash
set -u
cd /root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/work_dirs/w1/chain5.log
while pgrep -f "chain4.sh" > /dev/null; do sleep 120; done
echo "=== chain5: 补齐 plus1 模型的交叉评测 $(date) ===" | tee -a $LOG
$PY tools/eval_m1_split.py configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py \
  /root/autodl-tmp/work_dirs/w1/A4_plus1/epoch_8.pth --split test --ks 100 300 900 \
  --batch-size 4 --workers 8 --out /root/autodl-tmp/work_dirs/w1/eval_plus1_on_deployed.json >> $LOG 2>&1
$PY - <<'PY' | tee -a /root/autodl-tmp/work_dirs/w1/chain5.log
import json
d=json.load(open('/root/autodl-tmp/work_dirs/w1/eval_plus1_on_deployed.json'))
print('plus1 train -> long test  AR@100 %.4f  AR@300 %.4f  AR@900 %.4f | small %.3f medium %.3f large %.3f'
      % (d['AR']['AR@100'],d['AR']['AR@300'],d['AR']['AR@900'],
         d['recall_by_size']['small'],d['recall_by_size']['medium'],d['recall_by_size']['large']))
PY
echo "=== chain5 done $(date) ===" | tee -a $LOG
