#!/bin/bash
set -u
cd /root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python
W=/root/autodl-tmp/work_dirs/w1
LOG=$W/chain10.log
while pgrep -f "chain9.sh" > /dev/null; do sleep 120; done
echo "=== chain10 (cross-protocol: LOEO indonesia) start $(date) ===" | tee -a $LOG

echo "--- dash the fold anchor with a per-image dump ---" | tee -a $LOG
$PY tools/eval_m1_split.py /root/autodl-tmp/data_loeo_indonesia_long/config.py \
  work_dirs/m2_pf_rpn/loeo/indonesia/A4_add_change/epoch_8.pth \
  --split test --ks 100 300 900 --batch-size 4 --workers 8 \
  --dump-per-image $W/loeo_ind_anchor.npz --out $W/loeo_ind_anchor.json >> $LOG 2>&1
echo "anchor dump exit=$?" | tee -a $LOG

echo "--- TRAIN: +/-1 window on the fold's train set ---" | tee -a $LOG
wd=$W/A4_loeo_ind_plus1
$PY tools/train.py /root/autodl-tmp/data_loeo_indonesia_plus1/config.py --work-dir $wd > $wd.train.log 2>&1
echo "train exit=$? $(date)" | tee -a $LOG

if [ -f $wd/epoch_8.pth ]; then
  echo "--- EVAL: deployed on the long root (same as the anchor) ---" | tee -a $LOG
  $PY tools/eval_m1_split.py /root/autodl-tmp/data_loeo_indonesia_long/config.py $wd/epoch_8.pth \
    --split test --ks 100 300 900 --batch-size 4 --workers 8 \
    --dump-per-image $W/loeo_ind_plus1.npz --out $W/loeo_ind_plus1.json >> $LOG 2>&1
  $PY - <<'PY' | tee -a /root/autodl-tmp/work_dirs/w1/chain10.log
import json, numpy as np
W = '/root/autodl-tmp/work_dirs/w1'
for tag, p in (('ANCHOR (long->long)', W + '/loeo_ind_anchor.json'),
               ('PLUS1 train -> long deploy', W + '/loeo_ind_plus1.json')):
    try:
        d = json.load(open(p)); a = d['AR']
        print('%-28s AR@100 %.4f AR@300 %.4f AR@900 %.4f'
              % (tag, a['AR@100'], a['AR@300'], a['AR@900']))
    except Exception as e:
        print(tag, 'failed', e)
def small(p):
    z = np.load(p, allow_pickle=True); k = {}
    for q in z.files:
        a = z[q]
        if a.size and a.size <= 20000:
            k[q] = a.tolist() if a.ndim <= 2 else a.ravel()[:20000].tolist()
    return k
import json as J
J.dump({'anchor': small(W + '/loeo_ind_anchor.npz'),
        'plus1': small(W + '/loeo_ind_plus1.npz')},
       open(W + '/paired_ar_loeo.json', 'w'))
print('wrote paired_ar_loeo.json')
PY
else
  echo "fold: no ckpt" | tee -a $LOG; tail -n 25 $wd.train.log | tee -a $LOG
fi
echo "=== chain10 done $(date) ===" | tee -a $LOG
