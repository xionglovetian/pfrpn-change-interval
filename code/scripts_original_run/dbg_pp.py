import ast, numpy as np, xarray as xr, pathlib
raw = pathlib.Path("/root/autodl-tmp/Sen12Landslides/data/data_harmonized/s2")
for f in ("indonesia_s2_157.nc", "newzealand_s2_5971.nc", "kyrgyzstan1_s2_1710.nc"):
    p = raw / f
    if not p.exists():
        print("missing", f); continue
    with xr.open_dataset(p) as ds:
        t = np.asarray(ds["time"].values)
        def parse(key):
            v = ds.attrs.get(key)
            if v is None: return None
            if isinstance(v, str) and v.strip():
                try: return ast.literal_eval(v) if key == "pre_post_dates" else [x.strip() for x in v.split(",")]
                except Exception: return None
            return v
        pp = parse("pre_post_dates")
        ed = parse("event_date")
        ai = parse("ann_id")
        dc = parse("date_confidence")
        print("===", f, " n_time =", len(t))
        print("  len(pre_post)=%s len(event_date)=%s len(ann_id)=%s len(conf)=%s" % (
            len(pp) if hasattr(pp, "__len__") else 1, len(ed) if ed else 0,
            len(ai) if ai else 0, len(dc) if dc else 0))
        if isinstance(pp, tuple):
            for i, pair in enumerate(pp[:5]):
                if isinstance(pair, dict):
                    pre, post = int(pair.get("pre", 0)), int(pair.get("post", -1))
                    g = (t[post] - t[pre]) / np.timedelta64(1, "D")
                    print("    pair %2d: pre=%2d post=%2d  %s -> %s  gap=%.0fd" % (
                        i, pre, post, str(t[pre])[:10], str(t[post])[:10], g))
            print("    ... total %d pairs" % len(pp))
            gaps = []
            for pair in pp:
                if isinstance(pair, dict):
                    pre, post = int(pair.get("pre", 0)), int(pair.get("post", -1))
                    gaps.append((t[post] - t[pre]) / np.timedelta64(1, "D"))
            if gaps:
                g = np.array(gaps, dtype=float)
                print("    gap stats: min %.0f median %.0f max %.0f" % (g.min(), np.median(g), g.max()))
        else:
            print("    single/other:", repr(pp)[:80])