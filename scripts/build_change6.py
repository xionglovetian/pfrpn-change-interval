#!/usr/bin/env python
"""Build the 6-channel fused-interval arm: change = [long window | calendar +/-1].

Measured motivation (not assumed):
  * long window  -> better on large landslides   (anchor small/med/large 0.607/0.846/0.885)
  * calendar +-1 -> better on small & medium     (plus1 arm             0.621/0.880/0.827)
  * data layer: element-wise mean of the two windows gives AUC 0.854 /
    recall@1%FPR 0.289, never below either single window.

So give the existing per-level gate both intervals instead of choosing one.
No new module: GatedMultiModalFusion simply sees 6 change channels.

Passes (CPU + disk only):
  1. plus1 arrays for the 1,759 TEST tiles (unified rule) -> variants_plus1_test
     (the 225 multi-event positives get the widened window here, unlike the
      plus1 arm's test root which used the un-widened bracket)
  2. concat(deployed long, plus1) for every manifest tile -> variants_change6/
  3. data_change6/{manifests,norm_stats.json,config.py} with change=6
"""
import datetime
import glob
import json
import os
import os.path as osp
from multiprocessing import Pool

import numpy as np
from netCDF4 import Dataset, num2date

EPS = 1e-6
DATA = '/root/autodl-tmp/PF-RPN/data'
MAN = osp.join(DATA, 'landslide_multimodal_hn_lb/manifests')
W1 = '/root/autodl-tmp/_w1'
OUT6 = osp.join(W1, 'variants_change6')
PLUS1_TRAIN = osp.join(W1, 'variants_plus1')
PLUS1_TEST = osp.join(W1, 'variants_plus1_test')
ROOT = '/root/autodl-tmp/data_change6'
S22 = '/root/autodl-tmp/Sen12Landslides/data/data_harmonized/s2'
ORIG_NORM = osp.join(DATA, 'landslide_multimodal_hn_lb/norm_stats.json')
CFG_SRC = '/root/autodl-tmp/PF-RPN/configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py'


def resolve(rel):
    i = rel.find('landslide_multimodal_lb_')
    return osp.join(DATA, rel[i:].replace('\\', '/'))


def deployed_change(event, tile):
    for kind in ('landslide_multimodal_lb_pos', 'landslide_multimodal_lb_neg'):
        p = osp.join(DATA, kind, 'events', event, 'tiles', tile, 'change.npy')
        if osp.exists(p):
            return p
    raise FileNotFoundError('no deployed change for %s/%s' % (event, tile))


# --------------------------------------------------------------- plus1 unit
def _attr(a, key):
    v = a.get(key, '')
    return str(v.decode('utf-8', 'replace') if isinstance(v, bytes) else v).strip()


def _bracket(dates, event):
    try:
        e = datetime.date.fromisoformat(event)
    except ValueError:
        return None
    pre = max((i for i, x in enumerate(dates) if x < e), default=None)
    post = min((j for j, x in enumerate(dates) if x > e), default=None)
    if pre is None or post is None or pre >= post:
        return None
    return pre, post


def plus1_one(task):
    tid, path = task
    try:
        with Dataset(path) as ds:
            a = {k: ds.getncattr(k) for k in ds.ncattrs()}
            tv = ds.variables['time']
            t = np.asarray(tv[:], dtype='float64')
            order = np.argsort(t)
            dates = [datetime.date(d.year, d.month, d.day) for d in
                     num2date(t, getattr(tv, 'units', ''),
                              getattr(tv, 'calendar', 'standard'))]
            dates = [dates[i] for i in order]

            def band(nm):
                v = ds.variables[nm]
                v.set_auto_mask(False)
                return np.asarray(v[:])[order].astype(np.float32)

            b03, b04, b08, b12 = band('B03'), band('B04'), band('B08'), band('B12')
        n = b04.shape[0]
        raw = _attr(a, 'pre_post_dates')
        pair = None
        if raw:
            try:
                p = eval(raw)
                if isinstance(p, (tuple, list)) and p and isinstance(p[0], dict):
                    p = p[0]
                if isinstance(p, dict) and 'pre' in p and 'post' in p:
                    pair = (int(p['pre']), int(p['post']))
            except Exception:
                pair = None
        if pair is None:
            ev = _attr(a, 'event_date')
            if ev and ev != 'None':
                b = _bracket(dates, ev.split(',')[0].strip())
                if b is not None:
                    pair = b
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
        np.save(osp.join(PLUS1_TEST, tid + '.npy'),
                st.transpose(1, 0, 2).astype(np.float32))
        return (tid, 1, None)
    except Exception as exc:                                   # noqa: BLE001
        return (tid, 0, '%s: %s' % (type(exc).__name__, exc))


# ------------------------------------------------------------------ fuse
def build6(task):
    event, tile = task
    try:
        long_ = np.load(deployed_change(event, tile))
        short = None
        for d in (PLUS1_TRAIN, PLUS1_TEST):
            p = osp.join(d, tile + '.npy')
            if osp.exists(p):
                short = np.load(p)
                break
        if short is None:
            return (tile, 0, 'no plus1 array')
        if long_.shape[:2] != short.shape[:2]:
            return (tile, 0, 'shape %s vs %s' % (long_.shape, short.shape))
        np.save(osp.join(OUT6, tile + '.npy'),
                np.concatenate([long_.astype(np.float32),
                                short.astype(np.float32)], -1))
        return (tile, 1, None)
    except Exception as exc:                                   # noqa: BLE001
        return (tile, 0, '%s: %s' % (type(exc).__name__, exc))


def main():
    rows = {sp: json.load(open(osp.join(MAN, sp + '.json')))
            for sp in ('train', 'val', 'test')}

    # 1) plus1 arrays for the test split
    os.makedirs(PLUS1_TEST, exist_ok=True)
    idx = {}
    for p in glob.glob(osp.join(S22, '*.nc')):
        b = osp.basename(p)[:-3]
        idx[b.replace('_s2_', '_')] = p
    test_tasks = [(r['image_id'], idx[r['image_id']]) for r in rows['test']
                  if r['image_id'] in idx]
    print('test tiles needing plus1: %d' % len(test_tasks), flush=True)
    with Pool(24) as pool:
        r1 = pool.map(plus1_one, test_tasks, chunksize=8)
    print('plus1_test written: %d failed: %d %s'
          % (sum(x[1] for x in r1), len([x for x in r1 if not x[1]]),
             [x for x in r1 if not x[1]][:2]), flush=True)

    # 2) fuse
    os.makedirs(OUT6, exist_ok=True)
    seen, tasks = set(), []
    for sp in ('train', 'val', 'test'):
        for r in rows[sp]:
            tid = r['image_id']
            if tid in seen:
                continue
            seen.add(tid)
            tasks.append((r['event_id'], tid))
    print('tiles to fuse: %d' % len(tasks), flush=True)
    with Pool(24) as pool:
        res = pool.map(build6, tasks, chunksize=16)
    bad = [r for r in res if not r[1]]
    print('change6 written: %d  failed: %d %s'
          % (sum(x[1] for x in res), len(bad), bad[:3]), flush=True)

    # 3) root: manifests + norm + config
    os.makedirs(osp.join(ROOT, 'manifests'), exist_ok=True)
    sums = [np.zeros(6), np.zeros(6), 0]
    for sp in ('train', 'val', 'test'):
        out_rows, n6, nfb = [], 0, 0
        for r in rows[sp]:
            r = json.loads(json.dumps(r))
            mods = r['modalities']
            p6 = osp.join(OUT6, r['image_id'] + '.npy')
            for k, rel in list(mods.items()):
                if k != 'change':
                    mods[k] = resolve(rel)
            if osp.exists(p6):
                mods['change'] = p6
                n6 += 1
                if sp == 'train':
                    a = np.load(p6).reshape(-1, 6).astype(np.float64)
                    sums[0] += a.sum(0)
                    sums[1] += (a ** 2).sum(0)
                    sums[2] += a.shape[0]
            else:
                mods['change'] = resolve(mods['change'])
                nfb += 1
            out_rows.append(r)
        json.dump(out_rows, open(osp.join(ROOT, 'manifests', sp + '.json'), 'w'),
                  ensure_ascii=False)
        print('%-6s rows=%5d change6=%5d fallback3=%d'
              % (sp, len(out_rows), n6, nfb), flush=True)

    orig = json.load(open(ORIG_NORM))
    m6 = (sums[0] / sums[2]).tolist()
    s6 = np.sqrt(np.maximum(sums[1] / sums[2] - (sums[0] / sums[2]) ** 2,
                            0)).tolist()
    mean = list(orig['mean'][:12]) + m6
    std = list(orig['std'][:12]) + s6
    print('change6 mean=%s' % [round(v, 4) for v in m6], flush=True)
    print('change6 std =%s' % [round(v, 4) for v in s6], flush=True)
    json.dump(dict(mean=mean, std=std, keys=orig.get('keys'),
                   note='6ch fused interval [long3 | plus1_3]'),
              open(osp.join(ROOT, 'norm_stats.json'), 'w'), indent=2)

    cfg = open(CFG_SRC, encoding='utf-8').read()
    for old in ("data_root = 'data/landslide_multimodal_hn_lb/'",
                "data_root = 'data/landslide_multimodal_hn_lb'"):
        cfg = cfg.replace(old, "data_root = r'%s'" % ROOT)
    base = osp.join(osp.dirname(osp.abspath(CFG_SRC)), 'm2_pf_rpn_multimodal.py')
    cfg = cfg.replace("_base_ = ['./m2_pf_rpn_multimodal.py']",
                      "_base_ = [r'%s']" % base)
    cfg += ("\n\n# ---- 6-channel fused-interval overrides ----\n"
            "_modality_channels = dict(optical=3, sar=4, topography=5, change=6)\n"
            "modality_channels = _modality_channels\n"
            "model['modality_channels'] = _modality_channels\n"
            "model['data_preprocessor']['mean'] = %r\n"
            "model['data_preprocessor']['std'] = %r\n"
            "train_dataloader['dataset']['pipeline'][0]['modality_channels'] = "
            "_modality_channels\n" % (mean, std))
    open(osp.join(ROOT, 'config.py'), 'w', encoding='utf-8').write(cfg)
    print('wrote %s/config.py (len(mean)=%d)' % (ROOT, len(mean)), flush=True)


if __name__ == '__main__':
    main()
