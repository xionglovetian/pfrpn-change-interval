#!/bin/bash
set -u
cd /root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python
W=/root/autodl-tmp/work_dirs/w1
LOG=$W/chain8.log
while pgrep -f "chain6b.sh" > /dev/null; do sleep 120; done
echo "=== chain8 (seed replicates of the +/-1 train arm) start $(date) ===" | tee -a $LOG
for s in 42 123 777; do
  wd=$W/A4_plus1_s$s
  echo "--- TRAIN plus1 seed $s $(date) ---" | tee -a $LOG
  $PY tools/train.py /root/autodl-tmp/data_plus1/config.py --work-dir $wd \
      --cfg-options randomness.seed=$s > $wd.train.log 2>&1
  echo "train exit=$? $(date)" | tee -a $LOG
  if [ -f $wd/epoch_8.pth ]; then
    # deploy on the LONG baseline (the configuration that produced the headline)
    $PY tools/eval_m1_split.py configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py $wd/epoch_8.pth \
        --split test --ks 100 300 900 --batch-size 4 --workers 8 \
        --dump-per-image $wd/dump_long.npz --out $wd/eval_on_long.json > $wd.eval.log 2>&1
    $PY - $s $wd/eval_on_long.json <<'PY' | tee -a $LOG
import json, sys
s, p = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p))
    print('plus1 seed %s on LONG: AR@100 %.4f AR@300 %.4f AR@900 %.4f CI@900 %s'
          % (s, d['AR']['AR@100'], d['AR']['AR@300'], d['AR']['AR@900'],
             [round(x,4) for x in d['AR_ci95']['AR@900']]))
except Exception as e:
    print('seed', s, 'eval failed', e)
PY
  else
    echo "seed $s: no ckpt" | tee -a $LOG; tail -n 20 $wd.train.log | tee -a $LOG
  fi
done
echo "=== chain8 done $(date) ===" | tee -a $LOG
