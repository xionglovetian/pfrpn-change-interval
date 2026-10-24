import json, pathlib, numpy as np, xarray as xr, collections
root = pathlib.Path("/root/autodl-tmp/PF-RPN/data/landslide_multimodal_full")
raw = pathlib.Path("/root/autodl-tmp/Sen12Landslides/data/data_harmonized/s2")
entries = {}
for split in ("val", "test"):
    for it in json.loads((root / "manifests" / f"{split}.json").read_text()):
        entries.setdefault(it["event_id"], (split, []))[1].append(it)

CLOUDY = {3, 8, 9, 10, 11}
print("%-13s %5s %6s %8s %8s %9s %9s %9s" % (
    "event", "split", "n", "gap_d", "gap_med", "cloud_pre", "cloud_post", "gap|min,max"))
for ev, (split, lst) in sorted(entries.items()):
    gaps, cpre, cpost = [], [], []
    for it in lst[:40]:
        tile = it["image_id"]
        ev2, _, tid = tile.rpartition("_")
        p = raw / f"{ev2}_s2_{tid}.nc"
        if not p.exists():
            continue
        try:
            with xr.open_dataset(p) as ds:
                ds = ds.sortby("time")
                raw_attr = ds.attrs.get("pre_post_dates")
                if isinstance(raw_attr, str) and raw_attr.strip():
                    d = eval(raw_attr)
                    pre, post = int(d.get("pre", 0)), int(d.get("post", -1))
                else:
                    pre, post = 0, -1
                t = np.asarray(ds["time"].values)
                gap = (t[post] - t[pre]) / np.timedelta64(1, "D")
                gaps.append(float(gap))
                if "SCL" in ds:
                    scl_pre = np.asarray(ds["SCL"].isel(time=pre).values)
                    scl_post = np.asarray(ds["SCL"].isel(time=post).values)
                    cpre.append(float(np.isin(scl_pre, list(CLOUDY)).mean()))
                    cpost.append(float(np.isin(scl_post, list(CLOUDY)).mean()))
        except Exception:
            continue
    if not gaps:
        continue
    print("%-13s %5s %6d %8.1f %8.1f %9.3f %9.3f   [%.0f,%.0f]" % (
        ev, split, len(gaps), float(np.mean(gaps)), float(np.median(gaps)),
        float(np.mean(cpre)) if cpre else float("nan"),
        float(np.mean(cpost)) if cpost else float("nan"),
        float(np.min(gaps)), float(np.max(gaps))))