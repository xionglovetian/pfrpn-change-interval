#!/usr/bin/env python
"""Paired bootstrap for the anchor vs plus1 arm, from per-image AR dumps.

Consumes ``paired_ar.json`` (produced on the GPU box by chain6b): a dict with
two entries, ``anchor`` and ``plus1``, each holding the small arrays from
``tools/infer``'s per-image dump.  Large arrays (proposal scores/flags) are
dropped by the extractor.

Reports, per k, the paired mean difference with a percentile bootstrap CI over
images, plus a sign test, because the claim we want to make is "the +/-1 arm is
better at a strict budget", which is a paired per-image statement.
"""
import argparse
import json
import sys

import numpy as np


def find_ar_keys(d, n_img):
    """Locate per-image AR vectors: numeric, length n_img, named like ar*."""
    out = {}
    for k, v in d.items():
        if not isinstance(v, list) or len(v) != n_img:
            continue
        try:
            arr = np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 1:
            continue
        out[k] = arr
    return out


def find_ids(d, n_img):
    for k in ('image_ids', 'image_id', 'img_ids', 'sample_ids', 'img_id'):
        v = d.get(k)
        if isinstance(v, list) and len(v) == n_img:
            return [str(x) for x in v]
    return None


def bootstrap_ci(x, n_boot=20000, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def sign_test(d):
    """Two-sided sign test p-value, exact binomial for n<=200 else normal."""
    n = int((d != 0).sum())
    pos = int((d > 0).sum())
    if n == 0:
        return 1.0
    from math import comb
    if n <= 200:
        k = max(pos, n - pos)
        tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
        return float(min(1.0, 2 * tail))
    z = (pos - n / 2) / np.sqrt(n / 4)
    from math import erfc
    return float(erfc(abs(z) / np.sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    blob = json.load(open(args.json, encoding='utf-8'))
    if 'anchor' not in blob or 'plus1' not in blob:
        sys.exit('expect keys anchor/plus1, got %s' % list(blob))

    A = blob['anchor']
    # image ids: prefer the anchor side
    n_img = None
    for k in ('image_ids', 'image_id', 'img_id'):
        v = A.get(k)
        if isinstance(v, list) and len(v) > 100:
            n_img = len(v)
            break
    if n_img is None:
        for k in ('ar_900', 'ar_100', 'ar_300'):
            v = A.get(k)
            if isinstance(v, list):
                n_img = len(v)
                break
    ids_a = find_ids(A, n_img) if n_img else None
    ids_p = find_ids(blob['plus1'], n_img) if n_img else None
    print('images: %d | ids found: %s' % (n_img or -1, ids_a is not None))

    ar_a = find_ar_keys(A, n_img)
    ar_p = find_ar_keys(blob['plus1'], n_img)
    # per_image_io appends 0.0 for images without GT; AR is defined over GT
    # images only, so mask before averaging or the mean is diluted.
    ng = A.get('num_gt')
    if isinstance(ng, list) and len(ng) == n_img:
        keep = [i for i, v in enumerate(ng) if float(v) > 0]
        print('images with GT: %d of %d' % (len(keep), n_img))
        for k in list(ar_a):
            ar_a[k] = ar_a[k][keep]
        for k in list(ar_p):
            ar_p[k] = ar_p[k][keep]
    print('anchor ar keys:', sorted(ar_a))
    print('plus1  ar keys:', sorted(ar_p))

    # align on image ids when available, else assume identical order
    if ids_a and ids_p and ids_a != ids_p:
        pos = {t: i for i, t in enumerate(ids_p)}
        order = [pos[t] for t in ids_a if t in pos]
        print('reordering plus1 to anchor order: %d/%d' % (len(order), len(ids_a)))
        for k in ar_p:
            ar_p[k] = ar_p[k][order]
        for k in ar_a:
            ar_a[k] = ar_a[k][order] if len(ar_a[k]) == len(ids_a) else ar_a[k]
        ids_p = [ids_p[i] for i in order]

    results = {}
    print('\n%-10s %8s %8s %9s %9s %22s %9s' %
          ('metric', 'anchor', 'plus1', 'diff', 'rel', '95% CI', 'sign-p'))
    for k in sorted(set(ar_a) & set(ar_p)):
        a, p = ar_a[k], ar_p[k]
        if a.size != p.size:
            continue
        d = p - a
        lo, hi = bootstrap_ci(d)
        rel = float(d.mean() / a.mean()) if a.mean() != 0 else float('nan')
        sp = sign_test(d)
        results[k] = dict(anchor_mean=float(a.mean()),
                          plus1_mean=float(p.mean()),
                          diff=float(d.mean()), rel=rel,
                          ci=[lo, hi], sign_p=sp,
                          frac_pos=float((d > 0).mean()),
                          n=int(a.size))
        print('%-10s %8.4f %8.4f %+9.4f %+8.1f%% [%+.4f,%+.4f] %9.2e'
              % (k, a.mean(), p.mean(), d.mean(), 100 * rel, lo, hi, sp))

    print('\nReading guide: a claim of improvement needs the CI to exclude 0.')
    if args.out:
        json.dump(results, open(args.out, 'w', encoding='utf-8'), indent=2)
        print('wrote %s' % args.out)


if __name__ == '__main__':
    main()


