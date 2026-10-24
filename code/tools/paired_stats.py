# Copyright (c) OpenMMLab. All rights reserved.
"""Paired significance testing for M1 ablations (RESEARCH_ROADMAP.md 19.7).

Compares two runs scored on the *same* images and reports, for AR@k and
Recall@FPPI:

* the paired bootstrap 95% CI of ``A - B``;
* a paired permutation two-sided p-value (sign-flip for AR, prediction-swap
  for Recall@FPPI);
* an event-level cluster-bootstrap 95% CI (events, not images, are resampled);
* a Benjamini-Hochberg FDR across the whole metric family.

Reasoning: "CI overlap => not significant" is not a valid test, and both runs
are scored on identical tiles, so the comparison has to be paired.  Input is the
per-image dump written by ``--dump-per-image`` in ``eval_m1_split.py`` /
``eval_proposal_metrics.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import per_image_io as pio  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dump_a', help='per-image dump for run A (treatment)')
    p.add_argument('dump_b', help='per-image dump for run B (baseline)')
    p.add_argument('--label-a', default=None)
    p.add_argument('--label-b', default=None)
    p.add_argument('--ks', type=int, nargs='+', default=[100, 300, 900])
    p.add_argument('--fppi-points', type=float, nargs='+',
                   default=[0.1, 1.0, 10.0])
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--permutations', type=int, default=None,
                   help='default: same as --bootstrap')
    p.add_argument('--grid-size', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Recall@FPPI                                                                    #
# --------------------------------------------------------------------------- #
class FppiGrid:
    """Per-image TP/FP counts on a shared score grid, built once for resampling."""

    def __init__(self, dump, grid):
        n = len(dump['image_id'])
        self.n = n
        self.grid = np.asarray(grid, dtype=np.float64)
        self.tp = np.zeros((n, self.grid.size), dtype=np.float64)
        self.fp = np.zeros((n, self.grid.size), dtype=np.float64)
        self.num_gt = np.asarray(dump['num_gt'], dtype=np.int64)
        offsets = dump['offsets']
        scores = np.asarray(dump['scores'], dtype=np.float64)
        flags = np.asarray(dump['flags'], dtype=bool)
        for i in range(n):
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            s = scores[lo:hi]
            f = flags[lo:hi]
            if s.size == 0:
                continue
            order = np.argsort(-s, kind='stable')
            s = s[order]
            f = f[order]
            cum_tp = np.cumsum(f)
            cum_fp = np.cumsum(~f)
            # -s is ascending; count of detections with score >= t
            cnt = np.searchsorted(-s, -self.grid, side='right')
            safe = np.maximum(cnt - 1, 0)
            self.tp[i] = np.where(cnt > 0, cum_tp[safe], 0.0)
            self.fp[i] = np.where(cnt > 0, cum_fp[safe], 0.0)

    def recall_at(self, idx, point):
        tp = self.tp[idx].sum(axis=0)
        fp = self.fp[idx].sum(axis=0)
        total_gt = float(self.num_gt[idx].sum())
        if total_gt <= 0:
            return 0.0
        fppi = fp / max(len(idx), 1)
        valid = fppi <= point + 1e-12
        if not valid.any():
            return 0.0
        return float(tp[valid].max() / total_gt)

    def recall_at_from_grids(self, tp_sum, fp_sum, n_images, total_gt, point):
        if total_gt <= 0:
            return 0.0
        fppi = fp_sum / max(n_images, 1)
        valid = fppi <= point + 1e-12
        if not valid.any():
            return 0.0
        return float(tp_sum[valid].max() / total_gt)


def _shared_grid(a, b, size):
    pooled = np.concatenate([a['scores'], b['scores']])
    if pooled.size == 0:
        return np.asarray([0.0])
    qs = np.quantile(pooled, np.linspace(0.0, 1.0, max(size, 2)))
    return np.unique(np.concatenate([qs, [pooled.min(), pooled.max()]]))


# --------------------------------------------------------------------------- #
# Statistics                                                                     #
# --------------------------------------------------------------------------- #
def _ci(samples, alpha=0.05):
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0.0, 1.0)
    return adj


def _event_index(event_ids):
    groups = {}
    for i, e in enumerate(event_ids):
        groups.setdefault(str(e), []).append(i)
    return {e: np.asarray(v, dtype=np.int64) for e, v in groups.items()}


def _cluster_resample(groups, rng):
    names = list(groups)
    pick = rng.integers(0, len(names), size=len(names))
    return np.concatenate([groups[names[j]] for j in pick])


def paired_report(a, b, ia, ib, args):
    rng = np.random.default_rng(args.seed)
    n_boot = int(args.bootstrap)
    n_perm = int(args.permutations or args.bootstrap)

    grid = _shared_grid(a, b, args.grid_size)
    ga = FppiGrid(a, grid)
    gb = FppiGrid(b, grid)

    # paired arrays in the common image order
    num_gt = a['num_gt'][ia]
    assert np.array_equal(num_gt, b['num_gt'][ib]), 'GT counts differ across runs'
    gt_mask = num_gt > 0
    gt_pos = np.nonzero(gt_mask)[0]
    all_pos = np.arange(num_gt.size)
    ev_a = [a['event_id'][i] for i in ia]
    groups = _event_index(ev_a)

    metrics = []
    for k in args.ks:
        if k not in a['ar'] or k not in b['ar']:
            continue
        va = a['ar'][k][ia]
        vb = b['ar'][k][ib]
        obs = float(va[gt_pos].mean() - vb[gt_pos].mean())
        boot = np.empty(n_boot)
        for t in range(n_boot):
            pick = rng.choice(gt_pos, size=gt_pos.size, replace=True)
            boot[t] = va[pick].mean() - vb[pick].mean()
        d = va[gt_pos] - vb[gt_pos]
        perm = np.empty(n_perm)
        for t in range(n_perm):
            sign = rng.integers(0, 2, size=d.size) * 2 - 1
            perm[t] = (sign * d).mean()
        clust = np.empty(n_boot)
        for t in range(n_boot):
            pick = _cluster_resample(groups, rng)
            pick = pick[num_gt[pick] > 0]
            if pick.size:
                clust[t] = va[pick].mean() - vb[pick].mean()
            else:
                clust[t] = 0.0
        metrics.append(_summarize(
            f'AR@{k}', float(va[gt_pos].mean()), float(vb[gt_pos].mean()),
            obs, boot, perm, clust, float(va[gt_pos].mean() - vb[gt_pos].mean())))

    for point in args.fppi_points:
        obs_a = ga.recall_at(ia, point)
        obs_b = gb.recall_at(ib, point)
        obs = obs_a - obs_b
        boot = np.empty(n_boot)
        for t in range(n_boot):
            pick = rng.integers(0, all_pos.size, size=all_pos.size)
            boot[t] = ga.recall_at(ia[pick], point) - gb.recall_at(ib[pick], point)
        perm = np.empty(n_perm)
        for t in range(n_perm):
            swap = rng.integers(0, 2, size=all_pos.size).astype(bool)
            tpa = np.where(swap[:, None], gb.tp[ib], ga.tp[ia]).sum(axis=0)
            fpa = np.where(swap[:, None], gb.fp[ib], ga.fp[ia]).sum(axis=0)
            tpb = np.where(swap[:, None], ga.tp[ia], gb.tp[ib]).sum(axis=0)
            fpb = np.where(swap[:, None], ga.fp[ia], gb.fp[ib]).sum(axis=0)
            tot = float(num_gt.sum())
            ra = ga.recall_at_from_grids(tpa, fpa, all_pos.size, tot, point)
            rb = ga.recall_at_from_grids(tpb, fpb, all_pos.size, tot, point)
            perm[t] = ra - rb
        clust = np.empty(n_boot)
        for t in range(n_boot):
            pick = _cluster_resample(groups, rng)
            clust[t] = ga.recall_at(ia[pick], point) - gb.recall_at(ib[pick], point)
        metrics.append(_summarize(
            f'R@{point}FPPI', obs_a, obs_b, obs, boot, perm, clust, obs))

    _apply_fdr(metrics)
    return {
        'label_a': args.label_a or Path(args.dump_a).name,
        'label_b': args.label_b or Path(args.dump_b).name,
        'num_images': int(num_gt.size),
        'num_gt_images': int(gt_mask.sum()),
        'num_events': len(groups),
        'num_bootstrap': n_boot,
        'num_permutations': n_perm,
        'seed': int(args.seed),
        'metrics': {m['metric']: m for m in metrics},
    }


def _summarize(name, mean_a, mean_b, obs, boot, perm, clust, _obs_again):
    lo, hi = _ci(boot)
    clo, chi = _ci(clust)
    # two-sided permutation p with the +1 correction
    ge = np.count_nonzero(np.abs(perm) >= abs(obs) - 1e-12)
    pval = (ge + 1) / (perm.size + 1)
    return {
        'metric': name,
        'mean_a': float(mean_a),
        'mean_b': float(mean_b),
        'diff': float(obs),
        'bootstrap_ci95': [lo, hi],
        'perm_p': float(pval),
        'event_cluster_ci95': [clo, chi],
        'event_cluster_p': float(
            (np.count_nonzero(np.abs(clust - np.mean(clust)) >=
                              abs(np.mean(clust))) + 1) / (clust.size + 1)),
    }


def _apply_fdr(metrics):
    ps = [m['perm_p'] for m in metrics]
    adj = _bh_fdr(ps)
    for m, q in zip(metrics, adj):
        m['fdr_q'] = float(q)
        m['significant_fdr_05'] = bool(q < 0.05)


def main():
    args = parse_args()
    a = pio.load_dump(args.dump_a)
    b = pio.load_dump(args.dump_b)
    ia, ib = pio.align(a, b)
    report = paired_report(a, b, ia, ib, args)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
    print(text)
    print(f"\n{report['label_a']} vs {report['label_b']}"
          f"  (n_images={report['num_images']}, n_gt={report['num_gt_images']})")
    for name, m in report['metrics'].items():
        print(f"  {name:>14}  A={m['mean_a']:.4f} B={m['mean_b']:.4f}"
              f"  d={m['diff']:+.4f}"
              f"  CI[{m['bootstrap_ci95'][0]:+.4f},{m['bootstrap_ci95'][1]:+.4f}]"
              f"  p={m['perm_p']:.4g}  q={m['fdr_q']:.4g}"
              f"  {'SIG' if m['significant_fdr_05'] else 'ns'}")


if __name__ == '__main__':
    main()
