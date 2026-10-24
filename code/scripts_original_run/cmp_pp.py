import ast, json, pathlib, numpy as np, xarray as xr
root = pathlib.Path("/root/autodl-tmp/PF-RPN/data/landslide_multimodal_full")
raw = pathlib.Path("/root/autodl-tmp/Sen12Landslides/data/data_harmonized/s2")

def parse_pairs(a):
    if isinstance(a, str) and a.strip():
        try:
            v = ast.literal_eval(a)
        except Exception:
            return None
        return v
    return a

def first_nondeg(pairs):
    if isinstance(pairs, dict):
        return int(pairs.get("pre",0)), int(pairs.get("post",-1))
    if isinstance(pairs, (tuple, list)):
        for p in pairs:
            if isinstance(p, dict):
                a,b = int(p.get("pre",0)), int(p.get("post",-1))
                if a != b: return a,b
    return 0,-1

def band(ds,n,i):
    return ds[n].isel(time=i).transpose("y","x").values.astype(np.float32)

def ndvi_change(ds, pre, post):
    ds = ds.sortby("time")
    b4p,b8p = band(ds,"B04",pre), band(ds,"B08",pre)
    b4q,b8q = band(ds,"B04",post), band(ds,"B08",post)
    return ((b8q-b4q)/(b8q+b4q+1e-6)) - ((b8p-b4p)/(b8p+b4p+1e-6))

for ev, split in (("indonesia","test"), ("newzealand","val"), ("kyrgyzstan1","train")):
    items = json.loads((root/"manifests"/f"{split}.json").read_text())
    ev_items = [i for i in items if i["event_id"]==ev][:25]
    old_c, new_c, gaps_old, gaps_new = [], [], [], []
    for it in ev_items:
        tile = it["image_id"]; e2,_,tid = tile.rpartition("_")
        p = raw/f"{e2}_s2_{tid}.nc"
        if not p.exists(): continue
        mp = root/it["modalities"]["change"]
        msk_p = mp.parent/"mask.npy"
        if not msk_p.exists(): continue
        msk = np.load(msk_p) > 0
        if msk.sum() < 4: continue
        with xr.open_dataset(p) as ds:
            t = np.asarray(ds["time"].values)
            pairs = parse_pairs(ds.attrs.get("pre_post_dates"))
            pre_n, post_n = first_nondeg(pairs)
            old = ndvi_change(ds, 0, -1)
            new = ndvi_change(ds, pre_n, post_n)
            gaps_old.append((t[-1]-t[0])/np.timedelta64(1,"D"))
            gaps_new.append((t[post_n]-t[pre_n])/np.timedelta64(1,"D"))
        old_c.append(float(old[msk].mean()-old[~msk].mean()))
        new_c.append(float(new[msk].mean()-new[~msk].mean()))
    if not old_c: continue
    print("%-13s n=%2d  gap: old median %.0fd -> new median %.0fd" % (
        ev, len(old_c), np.median(gaps_old), np.median(gaps_new)))
    print("   contrast(old fallback 0,-1): mean %+.4f  frac_negative %.2f" % (
        np.mean(old_c), np.mean(np.array(old_c)<0)))
    print("   contrast(new first non-deg): mean %+.4f  frac_negative %.2f" % (
        np.mean(new_c), np.mean(np.array(new_c)<0)))