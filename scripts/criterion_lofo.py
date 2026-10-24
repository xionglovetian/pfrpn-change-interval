#!/usr/bin/env python3
"""b_lofo.py -- B 的统计加固：留一折（LOFO）预测检验

对 10 折中的每一折：用其余 9 折拟合线性关系，预测被留出折的"范式差距 PF-RPN/GDINO"，
再汇总"预测值 vs 实测值"的相关与误差。若关系只是被单折驱动，LOFO 会立刻暴露。
"""
import json, statistics
from pathlib import Path
import numpy as np

P = Path("/root/autodl-tmp/PF-RPN")
b = json.loads((P/"baselines/analysis/b_criterion.json").read_text())
rows = b["folds"]

x = np.array([r["auc_tilebg"] for r in rows], float)          # 可分性
y = np.array([r["pf"]/r["gdino"] for r in rows], float)       # 范式差距
folds = [r["fold"] for r in rows]

def pear(a, c):
    if len(a) < 3 or np.std(a) == 0 or np.std(c) == 0: return float("nan")
    return float(np.corrcoef(a, c)[0, 1])

def rank(v):
    v = np.asarray(v, float); o = v.argsort(); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
def spear(a, c): return pear(rank(a), rank(c))

pred = np.full(len(x), np.nan)
slopes = []
for i in range(len(x)):
    m = np.ones(len(x), bool); m[i] = False
    A = np.vstack([x[m], np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    pred[i] = coef[0]*x[i] + coef[1]
    slopes.append(coef[0])

print("== 留一折（LOFO）预测 ==")
print("%-18s %8s %8s %8s" % ("held-out fold", "x=AUC", "y_true", "y_pred"))
for i, f in enumerate(folds):
    print("%-18s %8.4f %8.3f %8.3f" % (f, x[i], y[i], pred[i]))

loo_r = pear(pred, y)
loo_s = spear(pred, y)
rmse = float(np.sqrt(np.mean((pred-y)**2)))
mae = float(np.mean(np.abs(pred-y)))
print("\nLOFO 预测 vs 实测: Pearson=%.3f  Spearman=%.3f  RMSE=%.3f  MAE=%.3f" % (loo_r, loo_s, rmse, mae))
print("逐折斜率 std=%.3f (均值 %.3f)，说明关系是否稳定" % (statistics.pstdev(slopes), statistics.mean(slopes)))

# 影响力：删掉某一折后全样本 r 的变化
full = pear(x, y)
inf = {}
for i, f in enumerate(folds):
    m = np.ones(len(x), bool); m[i] = False
    inf[f] = round(pear(x[m], y[m]) - full, 4)
print("\n单折影响力（删该折后 r 的变化）:")
for f, v in sorted(inf.items(), key=lambda kv: kv[1]):
    print("   %-18s %+.4f" % (f, v))
mx = max(inf.items(), key=lambda kv: abs(kv[1]))
print("最大影响折: %s (%+.4f) -> 关系%s单折驱动" % (mx[0], mx[1], "非" if abs(mx[1]) < 0.15 else "疑似"))

out = dict(source="baselines/analysis/b_criterion.json",
           lofo=dict(pearson_pred_vs_true=loo_r, spearman_pred_vs_true=loo_s,
                     rmse=rmse, mae=mae, slope_std=float(statistics.pstdev(slopes))),
           influence_drop_one={k: v for k, v in inf.items()},
           full_sample_pearson=full,
           conclusion=("关系非单折驱动" if abs(mx[1]) < 0.15 else "疑似单折驱动，需谨慎"),
           table=[dict(fold=f, x=float(x[i]), y_true=float(y[i]), y_pred=float(pred[i])) for i, f in enumerate(folds)])
(P/"baselines/analysis/b_lofo.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\nwrote baselines/analysis/b_lofo.json")