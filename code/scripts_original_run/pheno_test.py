"""Phenological-confound test.

Question: does Module 1's change channel respond to NON-landslide (seasonal)
change?  If the false-positive rate on GT-free tiles rises sharply with the
tile's background change magnitude, the "change dominates" gain is partly a
phenological artefact.

CPU only: reads the per-image dump + the raw change.npy tiles.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/root/autodl-tmp/PF-RPN/tools")
import per_image_io as pio

ROOT = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
DUMP = "/root/autodl-tmp/PF-RPN/artifacts/hn_lb_paired/A4_add_change/test.per_image.npz"
OUT = "/root/autodl-tmp/baselines/analysis"
os.makedirs(OUT, exist_ok=True)

man = json.load(open(os.path.join(ROOT, "manifests/test.json")))
ent = {}
for e in man:
    ent[str(e["image_id"])] = e
    if e.get("sample_id"): ent[str(e["sample_id"])] = e
d = pio.load_dump(DUMP)
ids = d["image_id"]; off = d["offsets"]; sc = d["scores"]; ng = d["num_gt"]

rows = []
for i, iid in enumerate(ids):
    e = ent.get(iid)
    if e is None:
        continue
    a = np.load(os.path.join(ROOT, e["modalities"]["change"])).astype(np.float32)
    bg = float(np.abs(a).mean())
    lo, hi = int(off[i]), int(off[i + 1])
    s = sc[lo:hi]
    rows.append(dict(iid=iid, event=str(e["event_id"]), ng=int(ng[i]), bg=bg,
                     n=len(s), top=float(s.max()) if len(s) else 0.0,
                     n_ge50=int((s >= 0.5).sum()), n_ge20=int((s >= 0.2).sum())))

neg = [r for r in rows if r["ng"] == 0]
pos = [r for r in rows if r["ng"] > 0]
print("tiles: %d (neg=%d, pos=%d)" % (len(rows), len(neg), len(pos)))

med = float(np.median([r["bg"] for r in neg]))
lo_g = [r for r in neg if r["bg"] <= med]
hi_g = [r for r in neg if r["bg"] > med]

def stat(g, name):
    if not g: return
    print("  %-10s n=%4d  bg=%.4f  cand/img=%.1f  top=%.3f  #>=0.5=%.2f  #>=0.2=%.2f" % (
        name, len(g), np.mean([r["bg"] for r in g]), np.mean([r["n"] for r in g]),
        np.mean([r["top"] for r in g]), np.mean([r["n_ge50"] for r in g]),
        np.mean([r["n_ge20"] for r in g])))

print()
print("=== Test 1: GT-free tiles, split by background |change| (median=%.4f) ===" % med)
stat(lo_g, "low-bg"); stat(hi_g, "high-bg")
r = (np.mean([x["n_ge20"] for x in hi_g]) / max(np.mean([x["n_ge20"] for x in lo_g]), 1e-9))
print("  ratio high/low (#candidates >=0.2) = %.2fx" % r)
r2 = (np.mean([x["n"] for x in hi_g]) / max(np.mean([x["n"] for x in lo_g]), 1e-9))
print("  ratio high/low (#candidates total)  = %.2fx" % r2)

print()
print("=== Test 2: per-event background change (GT-free tiles only) ===")
ev = {}
for x in neg: ev.setdefault(x["event"], []).append(x)
for e, g in sorted(ev.items(), key=lambda kv: -np.mean([r["bg"] for r in kv[1]])):
    print("  %-14s neg=%3d  bg=%.4f  cand/img=%.1f  #>=0.2/img=%.2f" % (
        e, len(g), np.mean([r["bg"] for r in g]),
        np.mean([r["n"] for r in g]), np.mean([r["n_ge20"] for r in g])))
json.dump(rows, open(os.path.join(OUT, "phenology_per_tile.json"), "w"))
print("\nwrote", os.path.join(OUT, "phenology_per_tile.json"))
