#!/bin/bash
set -u
cd /root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python
W=/root/autodl-tmp/work_dirs/w1
LOG=$W/chain9.log
while pgrep -f "chain8.sh" > /dev/null; do sleep 120; done
echo "=== chain9 start $(date) ===" | tee -a $LOG

echo "--- A. channel-count control: [long|long], 6ch ---" | tee -a $LOG
wd=$W/A4_longlong
$PY tools/train.py /root/autodl-tmp/data_longlong/config.py --work-dir $wd > $wd.train.log 2>&1
echo "control train exit=$? $(date)" | tee -a $LOG
if [ -f $wd/epoch_8.pth ]; then
  $PY tools/eval_m1_split.py /root/autodl-tmp/data_longlong/config.py $wd/epoch_8.pth \
    --split test --ks 100 300 900 --batch-size 4 --workers 8 \
    --out $wd/test_metrics.json >> $wd.eval.log 2>&1
  $PY - <<'PY' | tee -a /root/autodl-tmp/work_dirs/w1/chain9.log
import json
d=json.load(open('/root/autodl-tmp/work_dirs/w1/A4_longlong/test_metrics.json')); a=d['AR']
print('CONTROL [long|long]  AR@100 %.4f AR@300 %.4f AR@900 %.4f | CI@900 %s'
      % (a['AR@100'],a['AR@300'],a['AR@900'],[round(x,4) for x in d['AR_ci95']['AR@900']]))
print('reference: anchor 0.4652/0.5886/0.7094 | fusion 0.5789/0.6867/0.7634')
PY
fi

echo "--- B. fusion arm, seed 123 ---" | tee -a $LOG
wd=$W/A4_change6_s123
$PY tools/train.py /root/autodl-tmp/data_change6/config.py --work-dir $wd \
    --cfg-options randomness.seed=123 > $wd.train.log 2>&1
echo "fusion seed123 exit=$? $(date)" | tee -a $LOG
if [ -f $wd/epoch_8.pth ]; then
  $PY tools/eval_m1_split.py /root/autodl-tmp/data_change6/config.py $wd/epoch_8.pth \
    --split test --ks 100 300 900 --batch-size 4 --workers 8 \
    --out $wd/test_metrics.json >> $wd.eval.log 2>&1
  $PY - <<'PY' | tee -a /root/autodl-tmp/work_dirs/w1/chain9.log
import json
d=json.load(open('/root/autodl-tmp/work_dirs/w1/A4_change6_s123/test_metrics.json')); a=d['AR']
print('FUSION seed123  AR@100 %.4f AR@300 %.4f AR@900 %.4f'
      % (a['AR@100'],a['AR@300'],a['AR@900']))
PY
fi
echo "=== chain9 done $(date) ===" | tee -a $LOG
