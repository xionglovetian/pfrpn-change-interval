#!/usr/bin/env python3
"""analysis_b.py -- B：类无关可分性能否预测范式差距（n=10 折）

输入（均为已有产物，无需新算）：
  baselines/analysis/phenology_folds.json    逐折可分性（AUC / background_share / inside-outside 比值）
  loeo|loro/A4_*.json                        PF-RPN 逐折 AR@900
  baselines/metrics_loeo_loro/*.json         GDINO / YOLO-World 逐折 AR@900
输出：皮尔逊/斯皮尔曼相关 + bootstrap 置信区间，写入 baselines/analysis/b_criterion.json
"""
import json, glob, os, random
import numpy as np

P = "/root/autodl-tmp/PF-RPN"
phen = json.load(open(f"{P}/baselines/analysis/phenology_folds.json"))
folds = ["loeo_"+x for x in ["italy","kyrgyzstan1","newzealand","chimanimani","hiroshima","indonesia","hokkaido"]] + \
        ["loro_"+x for x in ["EastAsia","CentralAsia","SEAsia"]]

def a4(fold):
    sub, name = fold.split("_", 1)
    p = f"{P}/{sub}/A4_{sub}_{name}.json"
    return json.load(open(p))["AR"]["AR@900"] if os.path.exists(p) else None

rows = []
for f in folds:
    a = a4(f)
    if a is None: continue
    d = phen.get(f)
    if d is None: continue
    g = y = v = None
    for kind in ("gdino", "yw"):
        p = f"{P}/baselines/metrics_loeo_loro/{kind}_{f}_metrics.json"
        if os.path.exists(p):
            if kind == "gdino": g = json.load(open(p))["AR"]["AR@900"]
            else: y = json.load(open(p))["AR"]["AR@900"]
    sub, name = f.split("_", 1)
    pv = f"{P}/{sub}/y8_{sub}_{name}_metrics.json"
    if os.path.exists(pv): v = json.load(open(pv))["AR"]["AR@900"]
    rows.append(dict(fold=f, pf=a, gdino=g, yw=y, y8=v,
                     auc_tilebg=d["auc_inside_vs_tilebg"], auc_neg=d["auc_inside_vs_negtiles"],
                     bg_share=d["background_share"], ratio_in_out=d["ratio_median_inside_outside"]))

print("%-16s %7s %7s %7s %7s | %7s %7s %8s" % ("fold","PF-RPN","GDINO","YOLOW","YOLOv8s","AUC_bg","AUC_neg","ratio"))
for r in rows:
    print("%-16s %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %8.3f" % (
        r["fold"], r["pf"], r["gdino"] or float("nan"), r["yw"] or float("nan"), r["y8"] or float("nan"),
        r["auc_tilebg"], r["auc_neg"], r["ratio_in_out"]))

def pear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def spear(x, y):
    def rk(v):
        v = np.asarray(v, float); o = v.argsort(); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
    return pear(rk(x), rk(y))

def boot(x, y, n=5000, seed=0):
    rng = random.Random(seed); out = []
    idx = list(range(len(x)))
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        v = pear([x[i] for i in s], [y[i] for i in s])
        if v == v: out.append(v)
    if not out: return (float("nan"), float("nan"))
    out.sort(); return (out[int(0.025*len(out))], out[int(0.975*len(out))-1])

sep = [r["auc_tilebg"] for r in rows]
sep_neg = [r["auc_neg"] for r in rows]
bg = [r["bg_share"] for r in rows]
pf = [r["pf"] for r in rows]
gd = [r["gdino"] for r in rows]
yw = [r["yw"] for r in rows]
v8 = [r["y8"] for r in rows]
gap_pf_gd = [r["pf"]/r["gdino"] for r in rows if r["gdino"]]
sepg = [r["auc_tilebg"] for r in rows if r["gdino"]]

res = {}
def rep(name, x, y):
    r = pear(x, y); s = spear(x, y); lo, hi = boot(list(x), list(y))
    res[name] = dict(pearson=r, spearman=s, boot95=[lo, hi], n=len(x))
    print("%-42s pearson=%+.3f spearman=%+.3f  boot95=[%+.3f, %+.3f]  n=%d" % (name, r, s, lo, hi, len(x)))

print("\n== 相关（n=10 折） ==")
rep("sep(AUC vs tilebg)  vs PF-RPN", sep, pf)
rep("sep(AUC vs negtiles) vs PF-RPN", sep_neg, pf)
rep("sep(AUC vs tilebg)  vs 范式差距 PF/GDINO", sepg, gap_pf_gd)
rep("sep(AUC vs tilebg)  vs GDINO", sep, gd)
rep("sep(AUC vs tilebg)  vs YOLO-World", sep, yw)
rep("bg_share            vs PF-RPN", bg, pf)

out = dict(folds=rows, correlations=res,
           note="n=10（7 LOEO + 3 LORO）。可分性来自 baselines/analysis/phenology_folds.json；"
                "基线为本次公平口径（900 候选预算）。样本极小，相关系数仅作描述性证据。")
json.dump(out, open(f"{P}/baselines/analysis/b_criterion.json", "w"), indent=2, ensure_ascii=False)
print("\nwrote", f"{P}/baselines/analysis/b_criterion.json")