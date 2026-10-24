#!/bin/bash
set -u
cd /root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python
CK=/root/autodl-tmp/work_dirs/w1/calendar/epoch_8.pth
while pgrep -f "tools/train.py" > /dev/null; do sleep 60; done
echo "=== training exited at $(date) ===" | tee -a /root/autodl-tmp/work_dirs/w1/eval.log
if [ ! -f "$CK" ]; then
  echo "MISSING $CK -- training likely failed" | tee -a /root/autodl-tmp/work_dirs/w1/eval.log
  tail -n 40 /root/autodl-tmp/work_dirs/w1/calendar.log | tee -a /root/autodl-tmp/work_dirs/w1/eval.log
  exit 1
fi
ls -la "$CK" | tee -a /root/autodl-tmp/work_dirs/w1/eval.log
echo "--- eval A: calendar-trained model on calendar test root ---" | tee -a /root/autodl-tmp/work_dirs/w1/eval.log
$PY tools/eval_m1_split.py /root/autodl-tmp/data_calendar/config.py "$CK" \
  --split test --ks 100 300 900 --batch-size 4 --workers 8 \
  --dump-per-image /root/autodl-tmp/work_dirs/w1/cal_per_image.npz \
  --out /root/autodl-tmp/work_dirs/w1/eval_calendar_on_calendar.json >> /root/autodl-tmp/work_dirs/w1/eval.log 2>&1
echo "--- eval B: calendar-trained model on deployed (long) test root ---" | tee -a /root/autodl-tmp/work_dirs/w1/eval.log
$PY tools/eval_m1_split.py configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py "$CK" \
  --split test --ks 100 300 900 --batch-size 4 --workers 8 \
  --out /root/autodl-tmp/work_dirs/w1/eval_calendar_on_deployed.json >> /root/autodl-tmp/work_dirs/w1/eval.log 2>&1
$PY - <<'PY' | tee -a /root/autodl-tmp/work_dirs/w1/eval.log
import json
for tag, p in (('cal->cal', '/root/autodl-tmp/work_dirs/w1/eval_calendar_on_calendar.json'),
               ('cal->dep', '/root/autodl-tmp/work_dirs/w1/eval_calendar_on_deployed.json')):
    try:
        d = json.load(open(p))
        print('%-9s AR@100 %.4f  AR@300 %.4f  AR@900 %.4f  small %.3f medium %.3f large %.3f'
              % (tag, d['AR']['AR@100'], d['AR']['AR@300'], d['AR']['AR@900'],
                 d['recall_by_size']['small'], d['recall_by_size']['medium'],
                 d['recall_by_size']['large']))
    except Exception as e:
        print(tag, 'FAILED', e)
print('anchor on this machine: AR@100 0.4652  AR@300 0.5886  AR@900 0.7094')
PY
