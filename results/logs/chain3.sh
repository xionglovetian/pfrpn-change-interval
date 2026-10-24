#!/bin/bash
set -u
cd /root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/work_dirs/w1/chain3.log
echo "=== chain3 start $(date) ===" | tee -a $LOG
while pgrep -f "chain.sh" > /dev/null; do sleep 60; done
echo "=== chain1 (pair arm) done $(date) ===" | tee -a $LOG

# wait for the plus1 arrays to finish building
while pgrep -f build_plus1_train > /dev/null; do sleep 60; done
echo "=== plus1 arrays: $(ls /root/autodl-tmp/_w1/variants_plus1 2>/dev/null | wc -l) files ===" | tee -a $LOG

# training root for plus1
$PY /root/autodl-tmp/_w1/build_train_root.py \
  --data /root/autodl-tmp/PF-RPN/data \
  --man /root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb/manifests \
  --calendar /root/autodl-tmp/_w1/variants_plus1 \
  --test-arrays /root/autodl-tmp/_w1/variants/win_plus1,/root/autodl-tmp/_w1/variants_bracket_test \
  --config /root/autodl-tmp/PF-RPN/configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py \
  --out /root/autodl-tmp/data_plus1 >> $LOG 2>&1
echo "root built: $? $(date)" | tee -a $LOG

# train the plus1 arm
wd=/root/autodl-tmp/work_dirs/w1/A4_plus1
$PY tools/train.py /root/autodl-tmp/data_plus1/config.py --work-dir $wd > $wd.train.log 2>&1
echo "plus1 train exit=$? $(date)" | tee -a $LOG
if [ -f $wd/epoch_8.pth ]; then
  $PY tools/eval_m1_split.py /root/autodl-tmp/data_plus1/config.py $wd/epoch_8.pth \
      --split test --ks 100 300 900 --batch-size 4 --workers 8 \
      --out $wd/test_metrics.json >> $wd.eval.log 2>&1
  $PY - <<'PY' | tee -a /root/autodl-tmp/work_dirs/w1/chain3.log
import json
d = json.load(open('/root/autodl-tmp/work_dirs/w1/A4_plus1/test_metrics.json'))
print('A4_plus1  AR@100 %.4f  AR@300 %.4f  AR@900 %.4f  | small %.3f medium %.3f large %.3f'
      % (d['AR']['AR@100'], d['AR']['AR@300'], d['AR']['AR@900'],
         d['recall_by_size']['small'], d['recall_by_size']['medium'], d['recall_by_size']['large']))
PY
else
  echo "plus1: no ckpt, tail follows" | tee -a $LOG
  tail -n 30 $wd.train.log | tee -a $LOG
fi
echo "=== chain3 done $(date) ===" | tee -a $LOG
