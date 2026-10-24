#!/bin/bash
# a3_all_ablation.sh -- cross-event AND cross-region mechanism control (A3 = A4 minus change).
#
# Replaces the earlier split loeo_a3/loro_a3 scripts, which died of CUDA OOM because
# their "wait" used a log-mtime heuristic that misfired (training output goes to
# work_dirs/*/train.log, NOT to the queue log).  The real safety net here is a
# RESOURCE guard: before every fold we wait until the GPU is actually free.
#
# Instance deadline: 2026-10-14 12:17:12 CST.  Internal hard stop: 2026-10-14 08:00.
# Waits on a MARKER FILE only (no pgrep/pkill).  Idempotent: finished folds are skipped.
set -u
PROJ=/root/autodl-tmp/PF-RPN
PY=/root/miniconda3/bin/python3
OUT_LOEO=/root/autodl-tmp/loeo
OUT_LORO=/root/autodl-tmp/loro
LOG=/root/autodl-tmp/a3_all.log
LORO_DONE=/root/autodl-tmp/loro/LORO_DONE
BASE="train_dataloader.batch_size=8 optim_wrapper.optimizer.lr=0.00002"
HN=data/landslide_multimodal_hn_lb/
M=$PROJ/data/landslide_multimodal_hn_lb/manifests
DEADLINE=$(date -d '2026-10-14 08:00:00' +%s)
GPU_NEED=18000          # MiB free required before starting any training

exec > >(tee -a "$LOG") 2>&1
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"

log(){ echo "[$(date '+%F %T')] $*"; }

wait_for_gpu(){
  local need=$1
  for _ in $(seq 1 480); do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    free=${free:-0}
    if [ "$free" -ge "$need" ]; then log "gpu free ${free}MiB ok"; return 0; fi
    log "waiting for gpu: ${free}MiB free (< ${need})"
    sleep 60
  done
  log "!! gave up waiting for gpu"; return 1
}

run_fold(){           # $1=fold  $2=loeo|loro
  local F=$1 MODE=$2
  local ann out work eval_split
  if [ "$MODE" = "loeo" ]; then
    ann=$M/loeo/${F}_train.json; out=$OUT_LOEO/A3_loeo_${F}.json
    work=work_dirs/m2_pf_rpn/loeo/$F; eval_split=loeo_$F
  else
    ann=$M/loro/${F}_train.json; out=$OUT_LORO/A3_loro_${F}.json
    work=work_dirs/m2_pf_rpn/loro/$F; eval_split=loro_$F
  fi
  if [ -f "$out" ]; then log "[$F] already done, skip"; return 0; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then log "!! deadline hit, refusing $F"; return 2; fi
  wait_for_gpu "$GPU_NEED" || return 2

  log "########## A3 $MODE FOLD $F ##########"
  $PY tools/run_cloud_ablations.py A3_optical_sar_topography \
      --work-root "$work" \
      --cfg-options $BASE train_dataloader.dataset.data_root=$HN \
        train_dataloader.dataset.ann_file="$ann" \
        train_dataloader.persistent_workers=False
  if [ ! -f "$work/A3_optical_sar_topography/epoch_8.pth" ]; then
    log "[$F] !!! A3 checkpoint missing -> fold FAILED"; return 1
  fi
  $PY tools/eval_m1_split.py \
      "$work/A3_optical_sar_topography/m2_pf_rpn_cloud_full.py" \
      "$work/A3_optical_sar_topography/epoch_8.pth" \
      --split "$eval_split" --batch-size 8 --out "$out" > /dev/null 2>&1 \
      || log "[$F] eval cmd returned non-zero"
  if [ -f "$out" ]; then log "[$F] OK -> $out"; return 0; fi
  log "[$F] !!! eval produced no json"; return 1
}

log "=== A3-ALL ABLATION START ==="
log "hard stop: $(date -d @$DEADLINE '+%F %T')"

# 1) wait for LORO to publish its marker (no log-mtime guessing)
for _ in $(seq 1 720); do
  [ -f "$LORO_DONE" ] && { log "LORO_DONE found"; break; }
  sleep 60
done

fails=0
for F in italy newzealand indonesia hiroshima hokkaido chimanimani kyrgyzstan1; do
  run_fold "$F" loeo; rc=$?
  [ $rc -eq 2 ] && break
  if [ $rc -ne 0 ]; then fails=$((fails+1)); else fails=0; fi
  [ $fails -ge 3 ] && { log "!! 3 consecutive failures, aborting"; break; }
done

fails=0
for F in EastAsia CentralAsia SEAsia; do
  run_fold "$F" loro; rc=$?
  [ $rc -eq 2 ] && break
  if [ $rc -ne 0 ]; then fails=$((fails+1)); else fails=0; fi
  [ $fails -ge 2 ] && { log "!! consecutive failures, aborting LORO-A3"; break; }
done

log "=== A3-ALL summary ==="
$PY - <<'PYEOF'
import json
def show(path_tpl, folds, label):
    print("\n" + label)
    print("| fold | A3 (no change) | A4 (+change) | delta |")
    print("|---|---:|---:|---:|")
    d = []
    for f in folds:
        try: a3 = json.load(open(path_tpl.format(m="A3", f=f)))["AR"]["AR@900"]
        except Exception: a3 = float("nan")
        try: a4 = json.load(open(path_tpl.format(m="A4", f=f)))["AR"]["AR@900"]
        except Exception: a4 = float("nan")
        if a3 == a3 and a4 == a4: d.append(a4 - a3)
        print("| %s | %.4f | %.4f | %+.4f |" % (f, a3, a4, a4 - a3))
    if d: print("| **mean delta** | | | **%+.4f** |" % (sum(d)/len(d)))
show("/root/autodl-tmp/loeo/{m}_loeo_{f}.json",
     ["italy","newzealand","indonesia","hiroshima","hokkaido","chimanimani","kyrgyzstan1"],
     "LOEO (cross-event)")
show("/root/autodl-tmp/loro/{m}_loro_{f}.json",
     ["EastAsia","CentralAsia","SEAsia"], "LORO (cross-region)")
PYEOF
log "=== A3-ALL ABLATION DONE ==="

