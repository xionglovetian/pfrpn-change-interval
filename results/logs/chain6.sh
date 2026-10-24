#!/bin/bash
set -u
cd /root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/work_dirs/w1/chain6.log
while pgrep -f "chain5.sh" > /dev/null; do sleep 120; done
echo "=== chain6 (change6) start $(date) ===" | tee -a $LOG
wd=/root/autodl-tmp/work_dirs/w1/A4_change6
$PY tools/train.py /root/autodl-tmp/data_change6/config.py --work-dir $wd > $wd.train.log 2>&1
echo "train exit=$? $(date)" | tee -a $LOG
if [ -f $wd/epoch_8.pth ]; then
  $PY tools/eval_m1_split.py /root/autodl-tmp/data_change6/config.py $wd/epoch_8.pth \
      --split test --ks 100 300 900 --batch-size 4 --workers 8 \
      --dump-per-image $wd/per_image.npz --out $wd/test_metrics.json >> $wd.eval.log 2>&1
  $PY - <<'PY' | tee -a /root/autodl-tmp/work_dirs/w1/chain6.log
import json
d=json.load(open('/root/autodl-tmp/work_dirs/w1/A4_change6/test_metrics.json'))
print('A4_change6  AR@100 %.4f  AR@300 %.4f  AR@900 %.4f | small %.3f medium %.3f large %.3f'
      % (d['AR']['AR@100'],d['AR']['AR@300'],d['AR']['AR@900'],
         d['recall_by_size']['small'],d['recall_by_size']['medium'],d['recall_by_size']['large']))
print('thresholds: anchor 0.4652/0.5886/0.7094 | plus1 0.4990/0.6089/0.7179 | gate AR@900>0.7394')
PY
else
  echo "change6: no ckpt" | tee -a $LOG; tail -n 30 $wd.train.log | tee -a $LOG
fi
echo "=== chain6 done $(date) ===" | tee -a $LOG
