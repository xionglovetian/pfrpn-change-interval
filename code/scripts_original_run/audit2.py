import json, glob, hashlib, os
print("=== duplicate detection among valid metrics (n_gt>0) ===")
rows=[]
for f in sorted(glob.glob("/root/autodl-tmp/baselines/*_metrics.json")):
    try:
        d=json.load(open(f))
    except Exception as e:
        print("ERR", f, e); continue
    ng=d.get("num_gt_boxes")
    h=hashlib.md5(json.dumps(d.get("AR"), sort_keys=True).encode()).hexdigest()[:8]
    rows.append((h, os.path.basename(f), ng, (d.get("AR") or {}).get("AR@900")))
for h,f,ng,a9 in rows:
    tag = "BROKEN(n_gt=0)" if ng==0 else ""
    print("  %s  %-46s n_gt=%-6s AR900=%s %s" % (h, f, ng, ("%.4f"%a9) if a9 is not None else "-", tag))
print()
print("=== duplicate groups ===")
from collections import defaultdict
g=defaultdict(list)
for h,f,ng,a9 in rows:
    if ng and ng>0: g[h].append(f)
for h,fs in g.items():
    if len(fs)>1: print("  SAME AR block:", " == ".join(fs))
print()
print("=== search whole tree for yoloworld zero-shot metrics ===")
import subprocess
out=subprocess.run(["bash","-lc","grep -rl 'zeroshot' /root/autodl-tmp/baselines/*.json 2>/dev/null; ls -d /root/autodl-tmp/baselines/*zeroshot* 2>/dev/null"],capture_output=True,text=True).stdout
print(out or "  (nothing)")
print("=== proposal counts in zeroshot export dirs ===")
for d in glob.glob("/root/autodl-tmp/baselines/*zeroshot*"):
    n=len(glob.glob(os.path.join(d,"**","*.json"),recursive=True))
    print("  %s -> %d json" % (d, n))
