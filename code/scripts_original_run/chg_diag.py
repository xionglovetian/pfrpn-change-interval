import json, pathlib, numpy as np
root = pathlib.Path("/root/autodl-tmp/PF-RPN/data/landslide_multimodal_full")
man = root / "manifests"
groups = {"val": None, "test": None}
entries = {}
for split in ("train", "val", "test"):
    items = json.loads((man / f"{split}.json").read_text())
    for it in items:
        entries.setdefault(it["event_id"], []).append((split, it))

print("%-13s %5s %6s %7s %8s %8s %8s %8s %8s" % (
    "event", "split", "tiles", "ftiles", "dNDVI|in", "dNDVI|out", "contrast", "|chg|", "mask%"))
rows = {}
for ev, lst in sorted(entries.items()):
    split = lst[0][0]
    n = 0
    fr = 0
    in_m, out_m, cont, mag = [], [], [], []
    for sp, it in lst[:80]:
        d = root / it["modalities"]["change"]
        mp = d.parent / "mask.npy"
        if not d.exists() or not mp.exists():
            continue
        chg = np.load(d).astype(np.float64)
        msk = np.load(mp) > 0
        n += 1
        fr += int(msk.sum())
        if msk.sum() >= 4:
            for c in range(chg.shape[-1]):
                a = chg[..., c]
                iv = float(a[msk].mean()); ov = float(a[~msk].mean())
                in_m.append(iv); out_m.append(ov); cont.append(iv - ov)
        mag.append(float(np.abs(chg[..., 0]).mean()))
    if not n:
        continue
    rows[ev] = dict(split=split, n=n, in_m=in_m, cont=cont, mag=mag, fr=fr)
    print("%-13s %5s %6d %7d %8.4f %8.4f %8.4f %8.4f %8.2f" % (
        ev, split, n, len(cont), np.mean(in_m) if in_m else float("nan"),
        np.mean(out_m) if out_m else float("nan"),
        np.mean(cont) if cont else float("nan"),
        np.mean(mag) if mag else float("nan"),
        100.0 * fr / max(n, 1) / (128 * 128)))

print()
print("contrast = mean(change inside landslide mask) - mean(change outside)")
print("negative contrast for dNDVI/dNBR is the expected landslide signature (vegetation loss).")
print()
for ev, r in sorted(rows.items()):
    c = np.array(r["cont"]).reshape(r["n"], -1) if r["cont"] else None
    print("%-13s (%s) n=%3d  contrast mean=%+.4f  frac_tiles_negative=%.2f" % (
        ev, r["split"], len(r["cont"]),
        float(np.mean(r["cont"])) if r["cont"] else float("nan"),
        float(np.mean(np.array(r["cont"]) < 0)) if r["cont"] else float("nan")))