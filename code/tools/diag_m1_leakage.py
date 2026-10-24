# Copyright (c) OpenMMLab. All rights reserved.
"""Trivial-separability check for a single input modality (roadmap 19.11).

Motivation: ``change`` is the only modality whose addition moves AR@900 by a
large margin.  Before that can be written up we have to rule out a residual
shortcut -- e.g. the change channel being trivially separable between positive
and negative tiles because of how it was built.

This tool reads the raw ``.npy`` tiles named in a manifest, computes a few
per-tile summary statistics of one modality, and reports how well a single
statistic separates GT-bearing ("positive") from GT-free ("negative") tiles via
the rank AUC.  It is CPU-only and needs no GPU.

* AUC ~ 0.5  -> the statistic carries no positive/negative information.
* AUC -> 1.0 -> the modality can separate the two classes on its own, which is
  a red flag worth investigating before claiming a genuine mechanism.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', help='manifests/<split>.json')
    p.add_argument('--modality', default='change',
                   help='key inside sample["modalities"]')
    p.add_argument('--channels', type=int, nargs='*', default=None,
                   help='restrict to these channel indices')
    p.add_argument('--out', default=None)
    return p.parse_args()


def _resolve(manifest_dir, rel):
    """Resolve a manifest-relative modality path.

    Module-1 manifests store paths relative to the *dataset root* (the parent
    of ``manifests/``), but older writers used the manifest directory itself,
    so try every plausible base and return the first that exists.
    """
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    for base in (manifest_dir.parent, manifest_dir, Path.cwd()):
        cand = (base / rel).resolve()
        if cand.exists():
            return cand
    return (manifest_dir.parent / rel)


def _rankdata(x):
    """Average ranks (ties share the mean rank), numpy only."""
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # average ties
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i + 1
        while j < sx.size and sx[j] == sx[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def _auc(values, labels):
    """Rank AUC of ``values`` separating label==1 from label==0."""
    pos = values[labels == 1]
    neg = values[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    ranks = _rankdata(values)
    u = ranks[labels == 1].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _summarize(name, values, labels):
    pos, neg = values[labels == 1], values[labels == 0]
    return {
        'statistic': name,
        'pos_mean': float(pos.mean()) if pos.size else float('nan'),
        'pos_std': float(pos.std()) if pos.size else float('nan'),
        'neg_mean': float(neg.mean()) if neg.size else float('nan'),
        'neg_std': float(neg.std()) if neg.size else float('nan'),
        'auc': _auc(values, labels),
    }


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest_dir = manifest_path.parent
    items = json.loads(manifest_path.read_text(encoding='utf-8'))

    rows = []
    missing = 0
    for item in items:
        rel = item.get('modalities', {}).get(args.modality)
        if rel is None:
            missing += 1
            continue
        arr = np.load(_resolve(manifest_dir, rel)).astype(np.float64)
        if args.channels:
            arr = arr[..., list(args.channels)]
        has_gt = 1 if item['labels']['annotations'] else 0
        rows.append({
            'mean_abs': float(np.abs(arr).mean()),
            'std': float(arr.std()),
            'frac_nonzero': float((np.abs(arr) > 1e-6).mean()),
            'has_gt': has_gt,
        })
    if not rows:
        raise SystemExit('no tiles matched --modality ' + args.modality)

    labels = np.asarray([r['has_gt'] for r in rows], dtype=np.int64)
    result = {
        'manifest': str(manifest_path),
        'modality': args.modality,
        'channels': args.channels,
        'num_tiles': len(rows),
        'num_positive': int(labels.sum()),
        'num_negative': int((labels == 0).sum()),
        'num_missing_paths': missing,
        'separability': [
            _summarize(name,
                       np.asarray([r[name] for r in rows], dtype=np.float64),
                       labels)
            for name in ('mean_abs', 'std', 'frac_nonzero')
        ],
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
    print(text)
    print()
    print(f"modality={args.modality}  n={len(rows)} "
          f"(pos={int(labels.sum())}, neg={int((labels == 0).sum())})")
    for s in result['separability']:
        print(f"  {s['statistic']:>12}  pos={s['pos_mean']:.4f}  "
              f"neg={s['neg_mean']:.4f}  AUC={s['auc']:.4f}")
    print('  (interpretation: AUC near 0.5 = no shortcut; near 1.0 = '
          'modality separates classes by itself)')


if __name__ == '__main__':
    main()
