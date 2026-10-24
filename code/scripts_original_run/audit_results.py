#!/usr/bin/env python3
"""Audit of all M1 result artifacts on the cloud instance."""
import json, os, glob

def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"__err__": str(e)}

def ar(d):
    a = d.get("AR") or {}
    return (a.get("AR@100"), a.get("AR@300"), a.get("AR@900"))

def check(d):
    flags = []
    if "__err__" in d:
        return [f"JSON ERROR {d['__err__']}"], None
    a1, a3, a9 = ar(d)
    if a1 is None:
        return ["no AR block"], None
    if not (a1 <= a3 <= a9 + 1e-9):
        flags.append("!! AR not monotonic: %.4f/%.4f/%.4f" % (a1, a3, a9))
    for v in (a1, a3, a9):
        if v != v: flags.append("NaN in AR")
        elif v < 0 or v > 1: flags.append("out of range %.4f" % v)
    mr = d.get("micro_recall_at_iou") or {}
    m9 = mr.get("recall@900")
    if m9 is not None and a9 is not None and m9 > a9 + 0.15:
        flags.append("micro(%.3f) >> AR900(%.3f)" % (m9, a9))
    if d.get("num_images", 1) == 0: flags.append("num_images=0")
    if d.get("num_gt_boxes", 1) == 0: flags.append("num_gt_boxes=0")
    return flags, (a1, a3, a9, d.get("num_images"), d.get("num_gt_boxes"))

GROUPS = [
    ("LOEO A4 (PF-RPN)",       "/root/autodl-tmp/loeo/A4_loeo_*.json"),
    ("LOEO y8 (baseline)",     "/root/autodl-tmp/loeo/y8_loeo_*_metrics.json"),
    ("LORO A4 (PF-RPN)",       "/root/autodl-tmp/loro/A4_loro_*.json"),
    ("LORO y8 (baseline)",     "/root/autodl-tmp/loro/y8_loro_*_metrics.json"),
    ("A3 LOEO (control)",      "/root/autodl-tmp/loeo/A3_loeo_*.json"),
    ("A3 LORO (control)",      "/root/autodl-tmp/loro/A3_loro_*.json"),
    ("main ablations A0-A9",   "/root/autodl-tmp/PF-RPN/artifacts/hn_lb_metrics/*/test.json"),
    ("seed123",                "/root/autodl-tmp/PF-RPN/artifacts/hn_lb_seed123_metrics/*/test.json"),
    ("seed777",                "/root/autodl-tmp/PF-RPN/artifacts/hn_lb_seed777_metrics/*/test.json"),
    ("optical-dropout od_lb",  "/root/autodl-tmp/PF-RPN/artifacts/od_lb_metrics/*/test.json"),
    ("positive-only p_lb",     "/root/autodl-tmp/PF-RPN/artifacts/p_lb_metrics/*/test.json"),
    ("external baselines",     "/root/autodl-tmp/baselines/*_metrics.json"),
]
allflags = []
for label, pat in GROUPS:
    files = sorted(glob.glob(pat))
    print("=" * 106)
    print("### %s   (%d file(s))" % (label, len(files)))
    if not files:
        print("   (none)"); allflags.append(label + ": NO FILES"); continue
    print("   %-54s %8s %8s %8s %7s %8s" % ("file", "AR@100", "AR@300", "AR@900", "n_img", "n_gt"))
    for f in files:
        base = os.path.basename(f)
        d = load(f); fl, row = check(d)
        if row:
            a1, a3, a9, ni, ng = row
            print("   %-54s %8.4f %8.4f %8.4f %7s %8s" % (base, a1, a3, a9, ni, ng))
        else:
            print("   %-54s   --" % base)
        for x in fl:
            print("        >> " + x); allflags.append("%s/%s: %s" % (label, base, x))
print("\n" + "=" * 106)
print("### ANOMALY SUMMARY")
if allflags:
    for x in allflags: print("  -", x)
else:
    print("  none - all AR monotonic, in range, non-empty")
