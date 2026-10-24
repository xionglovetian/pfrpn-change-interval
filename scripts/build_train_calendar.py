"""Train-cohort change stacks under the calendar (bracket-the-event) window."""
import glob, json, os, os.path as osp, sys
from multiprocessing import Pool
import numpy as np
from netCDF4 import Dataset

EPS = 1e-6
OUT = '/root/autodl-tmp/_w1/variants_calendar'
os.makedirs(OUT, exist_ok=True)

def one(task):
    tid, path = task
    try:
        with Dataset(path) as ds:
            a = {k: ds.getncattr(k) for k in ds.ncattrs()}
            var = ds.variables['time']
            t = np.asarray(var[:], dtype='float64')
            order = np.argsort(t)
            def band(n):
                v = ds.variables[n]; v.set_auto_mask(False)
                return np.asarray(v[:])[order].astype(np.float32)
            b03, b04, b08, b12 = band('B03'), band('B04'), band('B08'), band('B12')
        def s(k):
            v = a.get(k, '')
            if isinstance(v, bytes): v = v.decode('utf-8', 'replace')
            return str(v).strip()
        ppd = s('pre_post_dates')
        pair = None
        if ppd:
            try:
                p = eval(ppd)
                if isinstance(p, dict) and 'pre' in p and 'post' in p:
                    pair = (int(p['pre']), int(p['post']))
            except Exception:
                pair = None
        n = b04.shape[0]
        if pair is None:
            pair = (n // 2 - 1, n // 2)     # negatives: mid-series surrogate
        i, j = max(0, pair[0]), min(n - 1, pair[1])
        if i == j:
            j = min(n - 1, i + 1)
        ndvi = (b08 - b04) / (b08 + b04 + EPS)
        ndwi = (b03 - b08) / (b03 + b08 + EPS)
        nbr = (b08 - b12) / (b08 + b12 + EPS)
        st = np.stack([ndvi[j] - ndvi[i], ndwi[j] - ndwi[i], nbr[j] - nbr[i]], -1)
        np.save(osp.join(OUT, tid + '.npy'), st.transpose(1, 0, 2).astype(np.float32))
        return (tid, 1)
    except Exception as e:
        return (tid, 0)

if __name__ == '__main__':
    man = '/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb/manifests'
    ids = []
    for sp in ('train', 'val'):
        for row in json.load(open(osp.join(man, sp + '.json'))):
            ids.append(row.get('image_id') or row.get('sample_id'))
    ids = sorted(set(ids))
    idx = {}
    for p in glob.glob('/root/autodl-tmp/Sen12Landslides/data/data_harmonized/s2/*.nc'):
        b = osp.basename(p)[:-3]
        idx[b.replace('_s2_', '_')] = p
    tasks = [(t, idx[t]) for t in ids if t in idx]
    print('tiles:', len(tasks), flush=True)
    with Pool(8) as pool:
        res = pool.map(one, tasks, chunksize=16)
    ok = sum(r[1] for r in res)
    print('written:', ok, 'failed:', len(res) - ok, flush=True)
