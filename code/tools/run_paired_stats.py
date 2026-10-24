# Copyright (c) OpenMMLab. All rights reserved.
"""Run paired significance tests for a set of M1 run pairs (roadmap 19.7).

Example::

    python tools/run_paired_stats.py --dump-root artifacts/hn_lb_metrics \
        --split test \
        --pairs A4_add_change:A3_optical_sar_topography \
                A5_add_modality_token:A4_add_change \
        --out-md docs/paired_stats_hn_lb_test.md

Each ``RUN_A:RUN_B`` compares A (treatment) against B (baseline) on the same
images using the per-image dumps written by ``--dump-per-image``.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import per_image_io as pio  # noqa: E402
from paired_stats import paired_report  # noqa: E402

METRIC_ORDER = ('AR@100', 'AR@300', 'AR@900',
                'R@0.1FPPI', 'R@1.0FPPI', 'R@10.0FPPI')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dump-root', required=True,
                   help='root holding <run>/<split>.per_image.npz')
    p.add_argument('--split', default='test')
    p.add_argument('--pairs', nargs='+', required=True,
                   help='RUN_A:RUN_B comparisons (A = treatment)')
    p.add_argument('--ks', type=int, nargs='+', default=[100, 300, 900])
    p.add_argument('--fppi-points', type=float, nargs='+',
                   default=[0.1, 1.0, 10.0])
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--permutations', type=int, default=None)
    p.add_argument('--grid-size', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--out-md', default=None)
    return p.parse_args()


def _ordered(metrics):
    keys = [k for k in METRIC_ORDER if k in metrics]
    keys += [k for k in metrics if k not in keys]
    return [(k, metrics[k]) for k in keys]


def main():
    args = parse_args()
    root = Path(args.dump_root)
    reports = []
    for pair in args.pairs:
        a_name, sep, b_name = pair.partition(':')
        if not sep:
            raise SystemExit(f'bad --pairs entry {pair!r}, expected RUN_A:RUN_B')
        da = root / a_name / f'{args.split}.per_image.npz'
        db = root / b_name / f'{args.split}.per_image.npz'
        if not da.exists() or not db.exists():
            raise SystemExit(f'missing dump: {da} exists={da.exists()}, '
                             f'{db} exists={db.exists()}')
        a = pio.load_dump(da)
        b = pio.load_dump(db)
        ia, ib = pio.align(a, b)
        sub = SimpleNamespace(
            seed=args.seed, bootstrap=args.bootstrap,
            permutations=args.permutations, grid_size=args.grid_size,
            ks=list(args.ks), fppi_points=list(args.fppi_points),
            label_a=a_name, label_b=b_name,
            dump_a=str(da), dump_b=str(db))
        report = paired_report(a, b, ia, ib, sub)
        report['split'] = args.split
        reports.append(report)
        print(f'{a_name} vs {b_name} ({args.split}, '
              f'n_gt={report["num_gt_images"]})')
        for name, m in _ordered(report['metrics']):
            print(f"  {name:>13}  A={m['mean_a']:.4f} B={m['mean_b']:.4f}"
                  f"  d={m['diff']:+.4f}"
                  f"  CI[{m['bootstrap_ci95'][0]:+.4f},{m['bootstrap_ci95'][1]:+.4f}]"
                  f"  p={m['perm_p']:.4g} q={m['fdr_q']:.4g}"
                  f"  {'SIG' if m['significant_fdr_05'] else 'ns'}")

    payload = {'split': args.split, 'bootstrap': args.bootstrap,
               'permutations': args.permutations or args.bootstrap,
               'seed': args.seed, 'comparisons': reports}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    if args.out_md:
        lines = [f'# Paired statistics ({args.split})', '',
                 f'- permutations: {payload["permutations"]}, '
                 f'bootstrap: {payload["bootstrap"]}, seed: {args.seed}',
                 '- p = paired permutation two-sided; q = BH-FDR across metrics '
                 'within each comparison.', '',
                 '| comparison | metric | A | B | Δ (A−B) | 95% CI | p | q |',
                 '|---|---|---:|---:|---:|:--:|---:|---:|']
        for rep in reports:
            for name, m in _ordered(rep['metrics']):
                ci = m['bootstrap_ci95']
                lines.append(
                    f"| {rep['label_a']} − {rep['label_b']} | {name} "
                    f"| {m['mean_a']:.4f} | {m['mean_b']:.4f} "
                    f"| {m['diff']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] "
                    f"| {m['perm_p']:.3g} | {m['fdr_q']:.3g} "
                    f"{'*' if m['significant_fdr_05'] else ''} |")
        Path(args.out_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2)[:0] + '')  # keep stdout compact


if __name__ == '__main__':
    main()
