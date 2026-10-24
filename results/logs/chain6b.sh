#!/bin/bash
set -u
cd /root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python
W=/root/autodl-tmp/work_dirs/w1
LOG=$W/chain6b.log
while pgrep -f "chain5.sh" > /dev/null; do sleep 60; done
echo "=== chain6b start $(date) ===" | tee -a $LOG

echo "--- dump: anchor (A4 + long window) ---" | tee -a $LOG
$PY tools/eval_m1_split.py configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py \
  work_dirs/m2_pf_rpn/hn_lb/A4_add_change/epoch_8.pth --split test --ks 100 300 900 \
  --batch-size 4 --workers 8 --dump-per-image $W/dump_anchor.npz \
  --out $W/dump_anchor.json >> $LOG 2>&1
echo "anchor exit=$? $(date)" | tee -a $LOG

echo "--- dump: plus1 arm ---" | tee -a $LOG
$PY tools/eval_m1_split.py /root/autodl-tmp/data_plus1/config.py $W/A4_plus1/epoch_8.pth \
  --split test --ks 100 300 900 --batch-size 4 --workers 8 \
  --dump-per-image $W/dump_plus1.npz --out $W/dump_plus1.json >> $LOG 2>&1
echo "plus1 exit=$? $(date)" | tee -a $LOG

echo "--- extract paired per-image AR ---" | tee -a $LOG
$PY - <<'PY' | tee -a $LOG
import json, numpy as np
W='/root/autodl-tmp/work_dirs/w1'
out={}
for tag in ('anchor','plus1'):
    z=np.load(f'{W}/dump_{tag}.npz', allow_pickle=True)
    keep={}
    for k in z.files:
        a=z[k]
        try:
            if a.size and a.size<=20000:
                keep[k]=a.tolist() if a.ndim<=2 else a.ravel()[:20000].tolist()
        except Exception:
            pass
    out[tag]=keep
    print(tag, 'keys kept:', sorted(keep.keys()))
json.dump(out, open(f'{W}/paired_ar.json','w'))
print('wrote paired_ar.json')
PY

echo "--- train change6 ---" | tee -a $LOG
wd=$W/A4_change6
$PY tools/train.py /root/autodl-tmp/data_change6/config.py --work-dir $wd > $wd.train.log 2>&1
echo "change6 train exit=$? $(date)" | tee -a $LOG
if [ -f $wd/epoch_8.pth ]; then
  $PY tools/eval_m1_split.py /root/autodl-tmp/data_change6/config.py $wd/epoch_8.pth \
      --split test --ks 100 300 900 --batch-size 4 --workers 8 \
      --dump-per-image $wd/per_image.npz --out $wd/test_metrics.json >> $wd.eval.log 2>&1
  $PY - <<'PY' | tee -a $LOG
import json
d=json.load(open('/root/autodl-tmp/work_dirs/w1/A4_change6/test_metrics.json'))
print('A4_change6  AR@100 %.4f  AR@300 %.4f  AR@900 %.4f | small %.3f medium %.3f large %.3f'
      % (d['AR']['AR@100'],d['AR']['AR@300'],d['AR']['AR@900'],
         d['recall_by_size']['small'],d['recall_by_size']['medium'],d['recall_by_size']['large']))
print('gate: AR@900 > 0.7394')
PY
else
  echo "change6: no ckpt" | tee -a $LOG; tail -n 25 $wd.train.log | tee -a $LOG
fi
echo "=== chain6b done $(date) ===" | tee -a $LOG
