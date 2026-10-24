import json, pathlib, numpy as np, xarray as xr, collections
root = pathlib.Path("/root/autodl-tmp/PF-RPN/data/landslide_multimodal_full")
raw = pathlib.Path("/root/autodl-tmp/Sen12Landslides/data/data_harmonized/s2")
items = json.loads((root / "manifests" / "val.json").read_text())
nz = [i for i in items if i["event_id"] == "newzealand"][:40]
errs = collections.Counter()
ok = 0
gaps = []
for it in nz:
    tile = it["image_id"]
    ev2, _, tid = tile.rpartition("_")
    p = raw / f"{ev2}_s2_{tid}.nc"
    if not p.exists():
        errs["missing_file"] += 1
        continue
    try:
        with xr.open_dataset(p) as ds:
            ds = ds.sortby("time")
            a = ds.attrs.get("pre_post_dates")
            if isinstance(a, str) and a.strip():
                d = eval(a)
                pre, post = int(d.get("pre", 0)), int(d.get("post", -1))
            else:
                pre, post = 0, -1
                errs["empty_attr"] += 1
            t = np.asarray(ds["time"].values)
            gaps.append(float((t[post] - t[pre]) / np.timedelta64(1, "D")))
            if "SCL" not in ds:
                errs["no_scl"] += 1
            ok += 1
    except Exception as e:
        errs[type(e).__name__ + ":" + str(e)[:40]] += 1
print("ok:", ok, "of", len(nz))
print("errors:", dict(errs))
if gaps:
    g = np.array(gaps)
    print("gap days: min %.0f p25 %.0f median %.0f p75 %.0f max %.0f" % (
        g.min(), np.percentile(g, 25), np.median(g), np.percentile(g, 75), g.max()))
    print("gap value counts:", collections.Counter(g.astype(int)).most_common(8))