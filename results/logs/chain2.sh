#!/bin/bash
set -u
cd /root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/work_dirs/w1/p0.log
# wait for the calendar arm (+ its evals) to finish
while pgrep -f "chain.sh" > /dev/null; do sleep 120; done
echo "=== chain1 done at $(date) ===" | tee -a $LOG

run_arm () {
  name="$1"; shift
  wd=/root/autodl-tmp/work_dirs/w1/$name
  echo "--- TRAIN $name  $(date) ---" | tee -a $LOG
  $PY tools/train.py configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py --work-dir "$wd" --cfg-options "$@" \
      > "$wd.train.log" 2>&1
  echo "train exit=$? $(date)" | tee -a $LOG
  if [ -f "$wd/epoch_8.pth" ]; then
    $PY tools/eval_m1_split.py configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py "$wd/epoch_8.pth" \
        --split test --ks 100 300 900 --batch-size 4 --workers 8 \
        --out "$wd/test_metrics.json" >> "$wd.eval.log" 2>&1
    $PY - "$name" "$wd/test_metrics.json" <<'PY' | tee -a $LOG
import json, sys
name, p = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p))
    print('%-14s AR@100 %.4f  AR@300 %.4f  AR@900 %.4f' %
          (name, d['AR']['AR@100'], d['AR']['AR@300'], d['AR']['AR@900']))
except Exception as e:
    print(name, 'eval failed:', e)
PY
  else
    echo "$name: no epoch_8.pth" | tee -a $LOG
    tail -n 25 "$wd.train.log" | tee -a $LOG
  fi
}

run_arm A0_rgb \
  model.use_sar=False model.use_topography=False model.use_change=False \
  model.use_modality_tokens=False model.use_cross_modal_router=False model.use_router=False

run_arm A9_no_change \
  model.use_sar=True model.use_topography=True model.use_change=False \
  model.use_modality_tokens=True model.use_cross_modal_router=True model.use_router=True

echo "=== P0 DONE at $(date) ===" | tee -a $LOG
echo "reference: A0 0.3533  A9 0.4799 (+/-0.025)" | tee -a $LOG
