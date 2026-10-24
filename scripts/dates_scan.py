import glob, json, os, os.path as osp
from multiprocessing import Pool
from netCDF4 import Dataset, num2date

def one(path):
    try:
        with Dataset(path) as ds:
            var = ds.variables['time']
            units = getattr(var, 'units', '')
            cal = getattr(var, 'calendar', 'standard')
            t = var[:]
            dates = [d.strftime('%Y-%m-%d') for d in num2date(t, units, cal)]
            a = {k: ds.getncattr(k) for k in ds.ncattrs()}
    except Exception as e:
        return None
    def s(k):
        v = a.get(k, '')
        if isinstance(v, bytes):
            v = v.decode('utf-8', 'replace')
        return str(v).strip()
    base = osp.basename(path)[:-3]
    return dict(tile=base.replace('_s2_', '_'), units=units,
                event_date=s('event_date'), ppd=s('pre_post_dates'),
                annotated=s('annotated'), date_confidence=s('date_confidence'),
                dates=dates)

if __name__ == '__main__':
    files = sorted(glob.glob('/root/autodl-tmp/Sen12Landslides/data/data_harmonized/s2/*.nc'))
    with Pool(16) as pool:
        rows = [r for r in pool.imap_unordered(one, files, chunksize=32) if r]
    with open('/root/autodl-tmp/_w1/dates.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print('wrote', len(rows))
