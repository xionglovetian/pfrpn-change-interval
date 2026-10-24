#!/bin/bash
cd /root/autodl-tmp || exit 1
for f in cloud_pipeline3.sh cloud_adneg.sh; do
  echo "########## $f ##########"
  grep -vE "^\s*#|^\s*$" "$f" | head -40
  echo
done
echo "########## 折脚本的调用方式 ##########"
echo "--- make_loro_folds.py ---"; grep -n "add_argument\|def main" make_loro_folds.py | head -10
echo "--- make_loeo_yolo_folds.py ---"; grep -n "add_argument\|def main" make_loeo_yolo_folds.py | head -10
echo
echo "########## loeo_run.sh / loro_run.sh 的核心循环参数 ##########"
grep -nE "BASE=|FOLDS=|ANN=|MAN|=.*manifest|run_cloud_ablations|eval_m1_split|--split" loeo_run.sh | head -12
echo "---"
grep -nE "BASE=|FOLDS=|M=|run_cloud_ablations|eval_m1_split|--split" loro_run.sh | head -12

