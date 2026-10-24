"""Build change stacks for test positives that lack pre_post_dates.

Uses the calendar rule (bracket the event date with the nearest acquisitions),
which reproduces the published window at 97.9% exact / 100% +/-1.
"""
import glob, json, os, os.path as osp
from multiprocessing import Pool
import datetime, numpy as np
from netCDF4 import Dataset, num2date

EPS = 1e-6
OUT = '/root/autodl-tmp/_w1/variants_bracket_test'
os.makedirs(OUT, exist_ok=True)
MAN = '/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb/manifests'

def bracket(dates, event):
    e = datetime.date.fromisoformat(event)
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
            var = ds.variables['time']
            t = var[:]
            dates = [datetime.date(d.year, d.month, d.day) for d in num2date(t, getattr(var, 'units', ''), getattr(var, 'calendar', 'standard'))]
            order = np.argsort(np.asarray(t, dtype='float64'))
            dates = [dates[i] for i in order]
            def band(n):
                v = ds.variables[n]; v.set_auto_mask(False)
                return np.asarray(v[:])[order].astype(np.float32)
            b03, b04, b08, b12 = band('B03'), band('B04'), band('B08'), band('B12')
        ev = a.get('event_date', '')
        ev = ev.decode() if isinstance(ev, bytes) else str(ev)
        ev = ev.strip().split(',')[0].strip()
        b = bracket(dates, ev)
        if b is None:
            return (tid, 0, 'no-bracket')
        i, j = b
        ndvi = (b08 - b04) / (b08 + b04 + EPS)
        ndwi = (b03 - b08) / (b03 + b08 + EPS)
        nbr = (b08 - b12) / (b08 + b12 + EPS)
        st = np.stack([ndvi[j]-ndvi[i], ndwi[j]-ndwi[i], nbr[j]-nbr[i]], -1)
        np.save(osp.join(OUT, tid + '.npy'), st.transpose(1, 0, 2).astype(np.float32))
        return (tid, 1, '')
    except Exception as e:
        return (tid, 0, '%s: %s' % (type(e).__name__, e))

if __name__ == '__main__':
    ids = []
    for row in json.load(open(osp.join(MAN, 'test.json'))):
        if not row.get('labels', {}).get('annotations'):
            continue
        tid = row['image_id']
        if osp.exists(osp.join('/root/autodl-tmp/_w1/variants/win_pair', tid + '.npy')):
            continue          # already has a pre_post window
        ids.append(tid)
    idx = {}
    for p in glob.glob('/root/autodl-tmp/Sen12Landslides/data/data_harmonized/s2/*.nc'):
        b = osp.basename(p)[:-3]
        idx[b.replace('_s2_', '_')] = p
    tasks = [(t, idx[t]) for t in ids if t in idx]
    print('missing test positives:', len(ids), 'matched:', len(tasks), flush=True)
    with Pool(8) as pool:
        res = pool.map(one, tasks, chunksize=8)
    ok = sum(r[1] for r in res)
    fails = [r for r in res if not r[1]]
    print('written:', ok, 'failed:', len(fails), fails[:5], flush=True)
