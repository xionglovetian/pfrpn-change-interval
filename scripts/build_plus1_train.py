#!/usr/bin/env python
"""plus1-window change arrays for train+val, with a calendar rule for every tile.

Window definition (in this order):
  1. ``pre_post_dates`` present          -> [pre-1, post+1]
  2. no ppd but ``event_date`` present   -> bracket the highest-confidence date
                                            (first entry of the comma list),
                                            then widen by 1 on each side
  3. no date at all (negatives)          -> [n//2-1, n//2+1], a span-matched
                                            surrogate (the event-prior baseline
                                            puts the real window at the series
                                            midpoint 86% of the time within +-1)

Rationale for rule 2: ~28% of positives are multi-event tiles whose event_date is
a comma-separated candidate list; leaving them on a mid-series surrogate would
make a third of the training set use an arbitrary window.
"""
import glob
import json
import os
import os.path as osp
from multiprocessing import Pool

import datetime
import numpy as np
from netCDF4 import Dataset, num2date

EPS = 1e-6
OUT = '/root/autodl-tmp/_w1/variants_plus1'
MAN = '/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb/manifests'


def _attr(a, key):
    v = a.get(key, '')
    if isinstance(v, bytes):
        v = v.decode('utf-8', 'replace')
    return str(v).strip()


def bracket(dates, event):
    try:
        e = datetime.date.fromisoformat(event)
    except ValueError:
        return None
    pre = max((i for i, x in enumerate(dates) if x < e), default=None)
    post = min((j for j, x in enumerate(dates) if x > e), default=None)
    if pre is None or post is None or pre >= post:
        return None
    return pre, post


def one(task):
    tid, path = task
    try:
        with Dataset(path) as ds:
            a = {k: ds.getncattr(k) for k in ds.ncattrs()}
            tv = ds.variables['time']
            t = np.asarray(tv[:], dtype='float64')
            order = np.argsort(t)
            units = getattr(tv, 'units', '')
            cal = getattr(tv, 'calendar', 'standard')
            dates = [datetime.date(d.year, d.month, d.day)
                     for d in num2date(t, units, cal)]
            dates = [dates[i] for i in order]

            def band(n):
                v = ds.variables[n]
                v.set_auto_mask(False)
                return np.asarray(v[:])[order].astype(np.float32)

            b03, b04, b08, b12 = band('B03'), band('B04'), band('B08'), band('B12')

        n = b04.shape[0]
        raw = _attr(a, 'pre_post_dates')
        pair, rule = None, 'surrogate'
        if raw:
            try:
                p = eval(raw)
                if isinstance(p, (tuple, list)) and p and isinstance(p[0], dict):
                    p = p[0]        # multi-event: highest-confidence candidate
                if isinstance(p, dict) and 'pre' in p and 'post' in p:
                    pair, rule = (int(p['pre']), int(p['post'])), 'ppd'
            except Exception:
                pair = None
        if pair is None:
            ev = _attr(a, 'event_date')
            if ev and ev != 'None':
                b = bracket(dates, ev.split(',')[0].strip())
                if b is not None:
                    pair, rule = b, 'bracket'
        if pair is None:
            i, j = max(0, n // 2 - 1), min(n - 1, n // 2 + 1)
        else:
            i, j = max(0, pair[0] - 1), min(n - 1, pair[1] + 1)
        if i >= j:
            i, j = 0, n - 1

        ndvi = (b08 - b04) / (b08 + b04 + EPS)
        ndwi = (b03 - b08) / (b03 + b08 + EPS)
        nbr = (b08 - b12) / (b08 + b12 + EPS)
        st = np.stack([ndvi[j] - ndvi[i], ndwi[j] - ndwi[i],
                       nbr[j] - nbr[i]], -1)
        np.save(osp.join(OUT, tid + '.npy'),
                st.transpose(1, 0, 2).astype(np.float32))
        return (tid, 1, rule)
    except Exception as exc:                                   # noqa: BLE001
        return (tid, 0, '%s: %s' % (type(exc).__name__, exc))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    ids = []
    for sp in ('train', 'val'):
        for row in json.load(open(osp.join(MAN, sp + '.json'))):
            ids.append(row.get('image_id') or row.get('sample_id'))
    ids = sorted(set(ids))
    idx = {}
    for p in glob.glob('/root/autodl-tmp/Sen12Landslides/data/'
                       'data_harmonized/s2/*.nc'):
        b = osp.basename(p)[:-3]
        idx[b.replace('_s2_', '_')] = p
    tasks = [(t, idx[t]) for t in ids if t in idx]
    print('tiles: %d  matched: %d' % (len(ids), len(tasks)), flush=True)
    with Pool(16) as pool:
        res = pool.map(one, tasks, chunksize=16)
    ok = sum(r[1] for r in res)
    bad = [r for r in res if not r[1]]
    rules = {}
    for r in res:
        if r[1]:
            rules[r[2]] = rules.get(r[2], 0) + 1
    print('written: %d  failed: %d %s' % (ok, len(bad), bad[:3]), flush=True)
    print('window rule used per tile: %s' % rules, flush=True)
