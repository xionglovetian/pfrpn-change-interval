"""Protocol audit: random tile split vs event-level split, same model/data.

Controlled comparison -- only the split assignment differs.
"""
import os, sys, json, random, time
import numpy as np
sys.path.insert(0, "/root/autodl-tmp/landslide-framework/module2")
from src.mine_hard_negatives import MiningConfig, load_gt_from_manifest, mine_proposal_dir
from src.training_data import build_samples
from src.train import TrainConfig, train_module2

DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
OUT  = "/root/autodl-tmp/m2_roi_cache_v2"
MODS = ("optical", "sar", "topography", "change")

def gt_of(split):
    g = load_gt_from_manifest(f"{DATA}/manifests/{split}.json", box_scale=1.0)
    for v in g.values():
        td = v.get("tile_dir")
        if td and not td.startswith("/"):
            v["tile_dir"] = os.path.normpath(os.path.join(DATA, td))
    return g

gtr = gt_of("train")
labels = mine_proposal_dir(f"{OUT}/A4_train_pos", "/tmp/d", gtr, split="train",
                           config=MiningConfig(score_thr=0.40), write=False)["labels"]
all_s = build_samples(f"{OUT}/A4_train_pos", "train", labels, gt_index=gtr)
print("pool size:", len(all_s), flush=True)

# cap the pool: keep ALL itogon, sample the rest
rnd = random.Random(0)
ito = [s for s in all_s if s.event_id == "itogon"]
oth = [s for s in all_s if s.event_id != "itogon"]
rnd.shuffle(oth)
pool = ito + oth[:40000]
pos = [s for s in pool if s.label == 1]; neg = [s for s in pool if s.label == 0]
print("pool=%d  pos=%d neg=%d  itogon=%d" % (len(pool), len(pos), len(neg), len(ito)), flush=True)

# ---- split builder -------------------------------------------------------
def random_split(samples, dev_frac=0.2, seed=0):
    r = random.Random(seed)
    imgs = sorted({s.image_id for s in samples}); r.shuffle(imgs)
    k = int(len(imgs) * dev_frac)
    dev_imgs = set(imgs[:k])
    tr = [s for s in samples if s.image_id not in dev_imgs]
    dv = [s for s in samples if s.image_id in dev_imgs]
    return tr, dv

def event_split(samples, dev_event="itogon"):
    tr = [s for s in samples if s.event_id != dev_event]
    dv = [s for s in samples if s.event_id == dev_event]
    return tr, dv

config = {"model": {"dim": 128, "roi_size": 6, "mask_size": 28, "num_hard_negatives": 1,
                    "use_evidential": True, "mask_decoder": "cnn",
                    "predict_mask_uncertainty": True, "fusion": "concat", "dropout": 0.2},
          "loss": {}}

def run(tag, tr, dv):
    t0 = time.time()
    cfg = TrainConfig(epochs=8, batch_size=64, lr=1e-4, out_dir=f"/root/autodl-tmp/audit_{tag}",
                      device="cuda", mask_size=28, balance=True, preload=True, seed=42)
    print(f"[{tag}] train={len(tr)} dev={len(dv)} dev_pos={sum(1 for s in dv if s.label==1)}", flush=True)
    summ = train_module2(config, tr, dv, None, train_cfg=cfg, dev_samples=dv)
    best = max((h.get("dev_ap") or 0) for h in summ["history"])
    last = summ["history"][-1]
    print(f"[{tag}] best_dev_AP={best:.4f}  last_dev_AP={last.get('dev_ap')}  ({time.time()-t0:.0f}s)", flush=True)
    return best, summ["history"]

trA, dvA = random_split(pool, 0.2, seed=0)
trB, dvB = event_split(pool, "itogon")

print("=== RUN A: random tile split (same events in train & dev) ===", flush=True)
apA, histA = run("random", trA, dvA)
print("=== RUN B: event-level split (dev event unseen in training) ===", flush=True)
apB, histB = run("event", trB, dvB)
print()
print("=" * 62)
print("PROTOCOL AUDIT")
print("  random tile split  best dev AP = %.4f" % apA)
print("  event-level split  best dev AP = %.4f" % apB)
print("  GAP                            = %.4f" % (apA - apB))
print("=" * 62)
json.dump({"random_split_AP": apA, "event_split_AP": apB, "gap": apA - apB,
           "history_random": histA, "history_event": histB},
          open("/root/autodl-tmp/audit_result.json", "w"), ensure_ascii=False, indent=2)