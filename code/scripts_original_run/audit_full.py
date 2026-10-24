#!/usr/bin/env python3
"""Comprehensive audit of everything valuable on the cloud instance."""
import json, os, glob, subprocess, hashlib

ROOT = "/root/autodl-tmp"
def sh(c): return subprocess.run(["bash","-lc",c],capture_output=True,text=True).stdout.strip()
def du(p): return sh(f"du -sh {p} 2>/dev/null | cut -f1") or "-"
def load(p):
    try: return json.load(open(p))
    except Exception as e: return {"__err__": str(e)}

print("#"*100); print("## A. 目录体量"); print("#"*100)
for p in ["PF-RPN/artifacts","PF-RPN/docs","PF-RPN/tools","PF-RPN/configs","PF-RPN/tests","PF-RPN/work_dirs",
          "baselines","loeo","loro","PF-RPN/data/landslide_multimodal_hn_lb/manifests"]:
    print("  %-52s %s" % (p, du(os.path.join(ROOT,p))))
print("  %-52s %s" % ("[work_dirs checkpoints]", sh(f"find {ROOT}/PF-RPN/work_dirs -name 'epoch_*.pth' | wc -l") + " pth files"))

print(); print("#"*100); print("## B. 结果 JSON 完整性校验"); print("#"*100)
GROUPS = [
 ("LOEO A4", f"{ROOT}/loeo/A4_loeo_*.json"),
 ("LOEO y8", f"{ROOT}/loeo/y8_loeo_*_metrics.json"),
 ("LOEO A3", f"{ROOT}/loeo/A3_loeo_*.json"),
 ("LORO A4", f"{ROOT}/loro/A4_loro_*.json"),
 ("LORO y8", f"{ROOT}/loro/y8_loro_*_metrics.json"),
 ("LORO A3", f"{ROOT}/loro/A3_loro_*.json"),
 ("main A0-A9", f"{ROOT}/PF-RPN/artifacts/hn_lb_metrics/*/test.json"),
 ("seed123", f"{ROOT}/PF-RPN/artifacts/hn_lb_seed123_metrics/*/test.json"),
 ("seed777", f"{ROOT}/PF-RPN/artifacts/hn_lb_seed777_metrics/*/test.json"),
 ("od_lb", f"{ROOT}/PF-RPN/artifacts/od_lb_metrics/*/test.json"),
 ("p_lb", f"{ROOT}/PF-RPN/artifacts/p_lb_metrics/*/test.json"),
 ("external baselines", f"{ROOT}/baselines/*_metrics.json"),
 ("analysis", f"{ROOT}/baselines/analysis/*.json"),
]
bad = []
for label, pat in GROUPS:
    fs = sorted(glob.glob(pat))
    print("\n-- %s : %d file(s)" % (label, len(fs)))
    for f in fs:
        d = load(f)
        if "__err__" in d:
            print("   [BAD] %s : %s" % (os.path.basename(f), d["__err__"])); bad.append(f); continue
        ar = d.get("AR") or {}
        a1,a3,a9 = ar.get("AR@100"), ar.get("AR@300"), ar.get("AR@900")
        if a1 is None:
            print("   [??] %s  (keys=%s)" % (os.path.basename(f), list(d)[:4])); continue
        ok = (a1 <= a3 <= a9 + 1e-9) and 0 <= a1 and a9 <= 1
        if not ok: bad.append(f)
        print("   %-46s AR900=%.4f n_gt=%-6s %s" % (os.path.basename(f), a9, d.get("num_gt_boxes"), "" if ok else "!!NON-MONOTONIC"))

print(); print("#"*100); print("## C. 标记与脚本"); print("#"*100)
for f in ["loro/LORO_DONE","a3_all.log","loro.log","loeo.log","stall_watch.log","pheno_folds.log","yw_lo_conf.log"]:
    p = os.path.join(ROOT,f)
    print("  %-30s %s" % (f, "EXISTS (%s)" % sh(f"stat -c %s {p}") if os.path.exists(p) else "MISSING"))
print("  scripts:", sh(f"ls {ROOT}/*.sh 2>/dev/null | wc -l"), "sh +", sh(f"ls {ROOT}/*.py 2>/dev/null | wc -l"), "py")

print(); print("#"*100); print("## D. 异常汇总"); print("#"*100)
print("  bad JSON / non-monotonic:", len(bad))
for b in bad: print("   -", b)

