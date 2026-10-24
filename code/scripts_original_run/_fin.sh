#!/bin/bash
cd /root/autodl-tmp || exit 1
date
echo "=== 终检：A3 完整性 ==="
echo "  LOEO A3: $(ls -1 loeo/A3_loeo_*.json 2>/dev/null | wc -l)/7"
echo "  LORO A3: $(ls -1 loro/A3_loro_*.json 2>/dev/null | wc -l)/3"
echo "=== 终检：全部产物清单 ==="
printf "  LOEO A4 %s/7  y8 %s/7  A3 %s/7\n" "$(ls -1 loeo/A4_loeo_*.json|wc -l)" "$(ls -1 loeo/y8_loeo_*_metrics.json|wc -l)" "$(ls -1 loeo/A3_loeo_*.json|wc -l)"
printf "  LORO A4 %s/3  y8 %s/3  A3 %s/3\n" "$(ls -1 loro/A4_loro_*.json|wc -l)" "$(ls -1 loro/y8_loro_*_metrics.json|wc -l)" "$(ls -1 loro/A3_loro_*.json|wc -l)"
echo "  LORO_DONE: $([ -f loro/LORO_DONE ] && echo yes || echo NO)"
echo
echo "=== 终检：数值合法性 ==="
/root/miniconda3/bin/python3 - <<'PY'
import json, glob, os, statistics as st
def a9(p): return json.load(open(p))["AR"]["AR@900"]
def a1(p): return json.load(open(p))["AR"]["AR@100"]
def a3(p): return json.load(open(p))["AR"]["AR@300"]
bad=[]
files = glob.glob("loeo/*.json")+glob.glob("loro/*.json")
for f in files:
    try:
        d=json.load(open(f)); ar=d["AR"]
        if not (ar["AR@100"] <= ar["AR@300"] <= ar["AR@900"]+1e-9): bad.append(f+" non-monotonic")
        if d.get("num_gt_boxes",0)==0: bad.append(f+" n_gt=0")
    except Exception as e: bad.append(f+" " + str(e))
print("  检查文件数:", len(files), " 异常:", len(bad))
for b in bad: print("   !!", b)
if not bad: print("  ✅ 全部单调、无 NaN、n_gt 正常")

ph=json.load(open("baselines/analysis/phenology_folds.json"))
rows=[]
for f in ["italy","kyrgyzstan1","newzealand","chimanimani","hiroshima","indonesia","hokkaido"]:
    rows.append(("LOEO",f,ph["loeo_"+f]["auc_inside_vs_tilebg"],a9(f"loeo/A3_loeo_{f}.json"),a9(f"loeo/A4_loeo_{f}.json")))
for f in ["EastAsia","CentralAsia","SEAsia"]:
    rows.append(("LORO",f,ph["loro_"+f]["auc_inside_vs_tilebg"],a9(f"loro/A3_loro_{f}.json"),a9(f"loro/A4_loro_{f}.json")))
aucs=[r[2] for r in rows]; rats=[r[4]/r[3] for r in rows]
import math
ma,mr=st.mean(aucs),st.mean(rats)
r=sum((a-ma)*(b-mr) for a,b in zip(aucs,rats))/math.sqrt(sum((a-ma)**2 for a in aucs)*sum((b-mr)**2 for b in rats))
print()
print("=== A3 最终表（10 折）===")
print("| 协议 | 折 | AUC | A3 | A4 | delta | 倍数 |")
print("|---|---|---:|---:|---:|---:|---:|")
for p,f,au,x,y in sorted(rows,key=lambda z:z[2]):
    print("| %s | %s | %.3f | %.4f | %.4f | %+.4f | %.2fx |" % (p,f,au,x,y,y-x,y/x))
print("  Pearson(AUC, ratio) = %+.3f   均值倍数 = %.2fx   delta 范围 %+.4f ~ %+.4f" % (r, mr, min(y-x for _,_,_,x,y in rows), max(y-x for _,_,_,x,y in rows)))
PY
echo
echo "=== 磁盘 ==="; df -h /root/autodl-tmp | tail -1

