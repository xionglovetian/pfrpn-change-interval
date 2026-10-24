import os, resource, time
from src.mine_hard_negatives import DEFAULT_EXPORT_SCALE, MiningConfig, load_gt_from_manifest, mine_proposal_dir
DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
ROOT = "/root/autodl-tmp/m2_roi_cache/A4_train_pos"
t0 = time.time()
gt = load_gt_from_manifest(DATA + "/manifests/train.json", box_scale=DEFAULT_EXPORT_SCALE)
for v in gt.values():
    td = v.get("tile_dir")
    if td and not td.startswith("/"):
        v["tile_dir"] = os.path.normpath(os.path.join(DATA, td))
p = mine_proposal_dir(ROOT, "/tmp/x", gt, split="train", config=MiningConfig(score_thr=0.40), write=False)
s = p["summary"]
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
print("num_images=%d num_candidates=%d num_positive=%d num_mined=%d labels=%d" % (
    s["num_images"], s["num_candidates"], s["num_positive"], s["num_mined_negative"], len(p["labels"])))
print("PEAK_RSS_GB=%.2f  elapsed=%.1fs" % (peak_gb, time.time() - t0))