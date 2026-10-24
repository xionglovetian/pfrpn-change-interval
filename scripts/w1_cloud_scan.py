#!/usr/bin/env python
"""Cloud-side (CPU-only) extraction for the W1 window plan.  ASCII only.

From the raw Sen12Landslides harmonised S2 NetCDF patches it produces:

  features_15f.jsonl        per tile: 15-frame tile-level stats + dating attrs  (F1/F3)
  mechanics.json            per annotated tile: window separability vs GT mask   (F2 mechanism)
  variants/<name>/<tile>.npy per-tile change stacks for the TEST split          (F2 eval arms)
  composite/<tile>.npy      temporal-composited dNDVI (pixel-wise pairing proxy)
  stats_raw.json            per-variant channel sums for TRAIN-side normalisation

Layout note (critical): the deployed converter calls ``.transpose('y','x')`` on
``(time, x, y)`` NetCDF variables, so every array that reached the model is the
spatial transpose of the raw block.  This script applies the same transpose;
otherwise new change stacks are silently mis-aligned with optical/topography.
(Verified against a deployed mask.npy: it matches only after transposing.)
"""
import argparse
import ast
import glob
import json
import os
import os.path as osp
import sys
import time
from multiprocessing import Pool

import numpy as np
from netCDF4 import Dataset

EPS = 1e-6
FILL = (0, 1, 3, 8, 9, 10, 11, 255)   # SCL classes treated as unusable
VARIANTS = ('win_long', 'win_pair', 'win_plus1', 'win_valid')


# ---------------------------------------------------------------- nc loading
def _read_var(ds, name, order=None):
    var = ds.variables[name]
    var.set_auto_mask(False)
    arr = np.asarray(var[:])
    if order is not None and arr.ndim == 3:
        arr = arr[order]
    return arr.astype(np.float32, copy=False)


def load_tile(path):
    with Dataset(path) as ds:
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        t = np.asarray(ds.variables['time'][:], dtype=np.float64)
        order = np.argsort(t)
        b03 = _read_var(ds, 'B03', order)
        b04 = _read_var(ds, 'B04', order)
        b08 = _read_var(ds, 'B08', order)
        b12 = _read_var(ds, 'B12', order)
        scl = np.asarray(ds.variables['SCL'][:])[order]
        mask = np.asarray(ds.variables['MASK'][0]) > 0
    ndvi = (b08 - b04) / (b08 + b04 + EPS)
    ndwi = (b03 - b08) / (b03 + b08 + EPS)
    nbr = (b08 - b12) / (b08 + b12 + EPS)
    valid = ~np.isin(scl, FILL)
    return dict(t=t[order],
                ndvi=ndvi.transpose(0, 2, 1),
                ndwi=ndwi.transpose(0, 2, 1),
                nbr=nbr.transpose(0, 2, 1),
                valid=valid.transpose(0, 2, 1),
                mask=mask.T, attrs=attrs)


def _attr(attrs, key):
    v = attrs.get(key, '')
    if isinstance(v, bytes):
        v = v.decode('utf-8', 'replace')
    return str(v).strip()


def parse_ppd(attrs):
    raw = _attr(attrs, 'pre_post_dates')
    if not raw:
        return None
    try:
        p = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(p, dict) and 'pre' in p and 'post' in p:
        try:
            return int(p['pre']), int(p['post'])
        except (TypeError, ValueError):
            return None
    return None


def change_stack(tile, i, j):
    return np.stack([tile['ndvi'][j] - tile['ndvi'][i],
                     tile['ndwi'][j] - tile['ndwi'][i],
                     tile['nbr'][j] - tile['nbr'][i]], -1).astype(np.float32)


def clip_pair(tile, i, j):
    n = tile['ndvi'].shape[0]
    return int(max(0, min(i, n - 1))), int(max(0, min(j, n - 1)))


# ------------------------------------------------------------- window rules
def rule_self_drop(tile, max_gap=None):
    """Label-free: adjacent pair with the largest tile-mean NDVI drop."""
    nd, vd = tile['ndvi'], tile['valid']
    best, pair = -np.inf, None
    for i in range(nd.shape[0] - 1):
        if max_gap is not None and tile['t'][i + 1] - tile['t'][i] > max_gap:
            continue
        ok = vd[i] & vd[i + 1]
        if ok.sum() < 0.25 * ok.size:
            continue
        d = float(nd[i][ok].mean()) - float(nd[i + 1][ok].mean())
        if d > best:
            best, pair = d, (i, i + 1)
    return pair


def rule_valid_pair(tile, pre, post, reach=2):
    """Widen +/-reach around the event but keep only usable frames."""
    n = tile['ndvi'].shape[0]
    vf = tile['valid'].mean(axis=(1, 2))
    lo, hi = max(0, pre - reach), min(n - 1, post + reach)
    cp = [i for i in range(lo, pre + 1) if vf[i] > 0.5] or [lo]
    cq = [j for j in range(post, hi + 1) if vf[j] > 0.5] or [hi]
    return int(min(cp)), int(max(cq))


def composite_dndvi(tile, pre, post, reach=2):
    """Per-pixel mean of usable frames before/after -> dNDVI (pairing proxy)."""
    n = tile['ndvi'].shape[0]
    lo, hi = max(0, pre - reach), min(n - 1, post + reach)
    out = np.zeros(tile['ndvi'].shape[1:], dtype=np.float32)
    good = np.zeros(tile['ndvi'].shape[1:], dtype=bool)
    acc = np.zeros(tile['ndvi'].shape[1:], dtype=np.float32)
    cnt = np.zeros(tile['ndvi'].shape[1:], dtype=np.float32)
    for sl in (range(lo, pre + 1), range(post, hi + 1)):
        sign = 1.0 if sl.start == lo else -1.0
        for k in sl:
            v = tile['valid'][k]
            acc[v] += sign * tile['ndvi'][k][v]
            cnt[v] += 1.0
    ok = cnt > 0.5
    out[ok] = acc[ok] / np.maximum(cnt[ok], 1.0)
    good[ok] = True
    return out.astype(np.float32), float(good.mean())


# ------------------------------------------------------------------ metrics
def auc(values, labels):
    values = np.asarray(values, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    order = np.argsort(values, kind='mergesort')
    sv = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    start = 0
    while start < sv.size:
        stop = start + 1
        while stop < sv.size and sv[stop] == sv[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return float((ranks[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def window_metrics(tile, i, j):
    ok = tile['valid'][i] & tile['valid'][j]
    drop = -change_stack(tile, i, j)[..., 0]
    pos = tile['mask'] & ok
    neg = (~tile['mask']) & ok
    if pos.sum() < 4 or neg.sum() < 100:
        return None
    sig, clut = float(drop[pos].mean()), float(drop[neg].mean())
    return dict(pair=[int(i), int(j)],
                gap_days=float(tile['t'][j] - tile['t'][i]),
                span_frac=float((tile['t'][j] - tile['t'][i]) /
                                max(tile['t'][-1] - tile['t'][0], EPS)),
                auc=auc(drop, pos),
                recall_fpr1=float((drop[pos] > np.percentile(drop[neg], 99.0)).mean()),
                signal=sig, clutter=clut,
                cohens_d=float((sig - clut) / max(float(drop[neg].std()), EPS)),
                valid_frac=float(ok.mean()))


# --------------------------------------------------------------- per-tile job
def job(task):
    kind, key, path, neg_mode = task
    out = dict(key=key, kind=kind, ok=False, err=None)
    try:
        tile = load_tile(path)
    except Exception as exc:                                   # noqa: BLE001
        out['err'] = f'{type(exc).__name__}: {exc}'
        return out
    n = tile['ndvi'].shape[0]
    attrs = tile['attrs']
    annotated = _attr(attrs, 'annotated').lower() == 'true'
    ppd = parse_ppd(attrs)

    # ---- 15-frame tile-level features (every tile) ------------------------
    if kind in ('features', 'both'):
        vf = tile['valid'].mean(axis=(1, 2))

        def _mean(arr, i):
            if vf[i] <= 0.05:
                return None
            return round(float(arr[i][tile['valid'][i]].mean()), 5)

        out['features'] = dict(
            tile=key, annotated=int(annotated),
            event_date=_attr(attrs, 'event_date'),
            date_confidence=_attr(attrs, 'date_confidence'),
            ppd=list(ppd) if ppd else None, n=n,
            t=[round(float(x), 2) for x in tile['t']],
            ndvi_mean=[_mean(tile['ndvi'], i) for i in range(n)],
            ndwi_mean=[_mean(tile['ndwi'], i) for i in range(n)],
            nbr_mean=[_mean(tile['nbr'], i) for i in range(n)],
            valid_frac=[round(float(x), 4) for x in vf],
            gt_px=int(tile['mask'].sum()))

    # ---- mechanics: does each window separate GT pixels? ------------------
    if kind in ('mech', 'both') and annotated and ppd:
        mech = {}
        cand = (('long', (0, n - 1)), ('pair', ppd),
                ('plus1', clip_pair(tile, ppd[0] - 1, ppd[1] + 1)),
                ('plus2', clip_pair(tile, ppd[0] - 2, ppd[1] + 2)),
                ('valid_pair', rule_valid_pair(tile, ppd[0], ppd[1])),
                ('mismatch', clip_pair(tile, ppd[0] - 6, max(ppd[1] - 6, 1))))
        for name, pair in cand:
            if pair[0] == pair[1]:
                continue
            m = window_metrics(tile, pair[0], pair[1])
            if m:
                mech[name] = m
        sp = rule_self_drop(tile)
        if sp:
            m = window_metrics(tile, sp[0], sp[1])
            if m:
                mech['self_drop'] = m
            mech['self_drop_pair'] = list(sp)
            mech['agree_exact'] = int(tuple(sp) == tuple(ppd))
            mech['agree_within1'] = int(abs(sp[0] - ppd[0]) <= 1 and
                                        abs(sp[1] - ppd[1]) <= 1)
        out['mechanics'] = mech

    # ---- test-split variant arrays ---------------------------------------
    if kind in ('variant', 'both'):
        pairs = {'win_long': (0, n - 1)}
        if ppd:
            pairs['win_pair'] = clip_pair(tile, ppd[0], ppd[1])
            pairs['win_plus1'] = clip_pair(tile, ppd[0] - 1, ppd[1] + 1)
            pairs['win_valid'] = rule_valid_pair(tile, ppd[0], ppd[1])
        else:
            # negatives have no event date.  'self' = same label-free rule for
            # every variant (keeps arms free of a variant-correlated input);
            # 'long' = leave negatives identical across arms (eval-only use).
            sp = rule_self_drop(tile, max_gap=30) if neg_mode == 'self' else (0, n - 1)
            sp = sp or (0, n - 1)
            for k in ('win_pair', 'win_plus1', 'win_valid'):
                pairs[k] = clip_pair(tile, sp[0], sp[1])
        out['variants'] = {k: change_stack(tile, v[0], v[1])
                           for k, v in pairs.items()}
        out['stats'] = {k: dict(sum=v.reshape(-1, 3).sum(0).tolist(),
                                sumsq=(v.reshape(-1, 3) ** 2).sum(0).tolist(),
                                count=int(v.shape[0] * v.shape[1]))
                        for k, v in out['variants'].items()}
        out['is_pos'] = int(annotated and ppd is not None)
        if annotated and ppd:
            comp, cov = composite_dndvi(tile, ppd[0], ppd[1])
            out['composite'] = comp
            out['composite_cov'] = cov
    out['ok'] = True
    return out


# --------------------------------------------------------------------- main
def build_index(s2dir):
    index = {}
    for path in glob.glob(osp.join(s2dir, '*.nc')):
        base = osp.basename(path)[:-3]
        index[base.replace('_s2_', '_')] = path
        index.setdefault(base, path)
    return index


def load_ids(path, only_gt=False):
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    ids = []
    for row in data:
        if only_gt and not row.get('labels', {}).get('annotations'):
            continue
        ids.append(row.get('image_id') or row.get('sample_id'))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/root/autodl-tmp/PF-RPN')
    ap.add_argument('--s2', default='/root/autodl-tmp/Sen12Landslides/data/'
                                    'data_harmonized/s2')
    ap.add_argument('--out', default='/root/autodl-tmp/_w1')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--neg-mode', choices=['long', 'self'], default='long')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    man = osp.join(args.repo, 'data/landslide_multimodal_hn_lb/manifests')
    ids_all = []
    for split in ('train', 'val', 'test'):
        p = osp.join(man, '%s.json' % split)
        if osp.exists(p):
            got = load_ids(p)
            ids_all.extend(got)
            print('%s: %d ids' % (split, len(got)), flush=True)
    ids_all = sorted(set(ids_all))
    if args.limit:
        ids_all = ids_all[:args.limit]
    test_ids = set(load_ids(osp.join(man, 'test.json')))

    print('unique tiles: %d  test-split: %d' % (len(ids_all), len(test_ids)),
          flush=True)
    index = build_index(args.s2)
    print('nc index: %d keys' % len(index), flush=True)

    tasks, missing = [], []
    for tid in ids_all:
        path = index.get(tid)
        if path is None or not osp.exists(path):
            missing.append(tid)
            continue
        tasks.append(('both' if tid in test_ids else 'features',
                      tid, path, args.neg_mode))
    print('matched: %d  missing: %d' % (len(tasks), len(missing)), flush=True)
    if missing:
        print('missing sample: %s' % missing[:10], flush=True)

    os.makedirs(args.out, exist_ok=True)
    for name in VARIANTS:
        os.makedirs(osp.join(args.out, 'variants', name), exist_ok=True)
    os.makedirs(osp.join(args.out, 'composite'), exist_ok=True)

    feats, mech = [], {}
    stats = {}
    comp_cov = []
    t0, done = time.time(), 0
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(job, tasks, chunksize=8):
            done += 1
            if not res['ok']:
                if done % 200 == 0:
                    print('  [%d/%d] fail %s: %s' % (done, len(tasks),
                                                     res['key'], res['err']),
                          flush=True)
                continue
            if 'features' in res:
                feats.append(res['features'])
            if 'mechanics' in res:
                mech[res['key']] = res['mechanics']
            if 'variants' in res:
                # negatives with neg-mode=long are identical across variants:
                # write win_long only and let the manifest builder reuse it
                save_keys = VARIANTS if res['is_pos'] else ('win_long',)
                for name in save_keys:
                    np.save(osp.join(args.out, 'variants', name,
                                     res['key'] + '.npy'), res['variants'][name])
                for k, st in res['stats'].items():
                    acc = stats.setdefault(k, dict(sum=[0.0] * 3,
                                                   sumsq=[0.0] * 3, count=0))
                    for c in range(3):
                        acc['sum'][c] += st['sum'][c]
                        acc['sumsq'][c] += st['sumsq'][c]
                    acc['count'] += st['count']
                if 'composite' in res:
                    np.save(osp.join(args.out, 'composite',
                                     res['key'] + '.npy'), res['composite'])
                    comp_cov.append(res['composite_cov'])
            if done % 500 == 0:
                print('  [%d/%d] %.0fs' % (done, len(tasks),
                                           time.time() - t0), flush=True)

    with open(osp.join(args.out, 'features_15f.jsonl'), 'w',
              encoding='utf-8') as fh:
        for row in sorted(feats, key=lambda r: r['tile']):
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    with open(osp.join(args.out, 'mechanics.json'), 'w', encoding='utf-8') as fh:
        json.dump(mech, fh, ensure_ascii=False)
    norm = {}
    for k, a in stats.items():
        m = [s / a['count'] for s in a['sum']]
        sd = [float(np.sqrt(max(q / a['count'] - (s / a['count']) ** 2, 0.0)))
              for s, q in zip(a['sum'], a['sumsq'])]
        norm[k] = dict(count=a['count'], mean=m, std=sd)
    with open(osp.join(args.out, 'stats_variants.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(norm, fh, indent=2)
    print('DONE %d tasks in %.0fs' % (done, time.time() - t0), flush=True)
    print('features=%d mechanics=%d missing=%d composite=%d cov_mean=%s'
          % (len(feats), len(mech), len(missing), len(comp_cov),
             round(float(np.mean(comp_cov)), 4) if comp_cov else None), flush=True)


if __name__ == '__main__':
    sys.exit(main())
