#!/usr/bin/env python3
"""b_fair.py -- B 判据重算（公平口径）：可分性 -> 范式差距，含 bootstrap CI 与留一折"""
import json, os, random, statistics
from pathlib import Path
import numpy as np

P = Path("/root/autodl-tmp/PF-RPN")
phen = json.loads((P/"baselines/analysis/phenology_folds.json").read_text())
folds = ["loeo_"+x for x in ["italy","kyrgyzstan1","newzealand","chimanimani","hiroshima","indonesia","hokkaido"]] + \
        ["loro_"+x for x in ["EastAsia","CentralAsia","SEAsia"]]

def g(p):
    try: return json.load(open(p))["AR"]["AR@900"]
    except Exception: return None

rows = []
for f in folds:
    sub, name = f.split("_", 1)
    a = g(P/("%s/A4_%s_%s.json" % (sub, sub, name)))
    d = g(P/("baselines/metrics_fair/gdino_%s_metrics.json" % f))
    w = g(P/("baselines/metrics_fair/yw_%s_metrics.json" % f))
    d0 = phen.get(f)
    if None in (a, d, w) or d0 is None: continue
    rows.append(dict(fold=f, pf=a, gdino=d, yw=w, auc=d0["auc_inside_vs_tilebg"],
                     auc_neg=d0["auc_inside_vs_negtiles"], bg=d0["background_share"]))

def pear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def rank(v):
    v = np.asarray(v, float); o = v.argsort(); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
def spear(x, y): return pear(rank(x), rank(y))

def boot(x, y, n=5000, seed=0):
    rng = random.Random(seed); idx = list(range(len(x))); out = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        v = pear([x[i] for i in s], [y[i] for i in s])
        if v == v: out.append(v)
    if not out: return [float("nan")]*2
    out.sort(); return [out[int(0.025*len(out))], out[int(0.975*len(out))-1]]

def lofo(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    pred = np.full(len(x), np.nan)
    for i in range(len(x)):
        m = np.ones(len(x), bool); m[i] = False
        A = np.vstack([x[m], np.ones(m.sum())]).T
        c, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        pred[i] = c[0]*x[i] + c[1]
    full = pear(x, y)
    inf = {rows[i]["fold"]: round(pear(x[[j != i for j in range(len(x))]], y[[j != i for j in range(len(x))]]) - full, 4) for i in range(len(x))}
    mx = max(inf.items(), key=lambda kv: abs(kv[1]))
    return dict(pearson_pred_vs_true=pear(pred, y), spearman_pred_vs_true=spear(pred, y),
                rmse=float(np.sqrt(np.mean((pred-y)**2))), drop_one=inf,
                max_influence=[mx[0], mx[1]], single_fold_driven=abs(mx[1]) >= 0.15,
                table=[dict(fold=rows[i]["fold"], x=float(x[i]), y_true=float(y[i]), y_pred=float(pred[i])) for i in range(len(x))])

sep = [r["auc"] for r in rows]
gd  = [r["gdino"] for r in rows]
yw  = [r["yw"] for r in rows]
pf  = [r["pf"] for r in rows]
gap_gd = [r["pf"]/r["gdino"] for r in rows]
gap_yw = [r["pf"]/r["yw"] for r in rows]

res = {}
def rep(name, x, y):
    r, s = pear(x, y), spear(x, y); lo, hi = boot(list(x), list(y))
    res[name] = dict(pearson=r, spearman=s, boot95=[lo, hi], n=len(x),
                     lofo=lofo(x, y))
    print("%-40s pearson=%+.3f spearman=%+.3f boot95=[%+.3f,%+.3f] LOFO r=%.3f 最大影响%s %s" % (
        name, r, s, lo, hi, res[name]["lofo"]["pearson_pred_vs_true"],
        res[name]["lofo"]["max_influence"][0], res[name]["lofo"]["max_influence"][1]))

print("== B（公平口径，n=%d 折）==" % len(rows))
rep("可分性 -> 范式差距 PF/GDINO(公平)", sep, gap_gd)
rep("可分性 -> 范式差距 PF/YW(公平)", sep, gap_yw)
rep("可分性 -> GDINO(公平)", sep, gd)
rep("可分性 -> YOLO-World(公平)", sep, yw)
rep("可分性 -> PF-RPN 自身", sep, pf)

out = dict(protocol="公平口径（GDINO 阈值0/min-box0；YOLO-World conf=1e-8, NMS iou=0.99；均 max 900）",
           folds=rows, correlations=res,
           note="n=10（7 LOEO + 3 LORO）。相对旧版（YW 仅 4% 预算）的主要变化在 YOLO-World 相关。")
(P/"baselines/analysis").mkdir(parents=True, exist_ok=True)
(P/"baselines/analysis/b_criterion_fair.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\nwrote baselines/analysis/b_criterion_fair.json")