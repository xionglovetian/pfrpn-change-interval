import json, os, sys
sys.path.insert(0, "/root/autodl-tmp/landslide-framework/module2")
from src.mine_hard_negatives import (DEFAULT_EXPORT_SCALE, MiningConfig,
                                     load_gt_from_manifest, mine_proposal_dir)
from src.training_data import build_samples
from src.train import TrainConfig, train_module2

DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
ROOT = "/root/autodl-tmp/m2_roi_cache/A4_hokkaido"

gt = load_gt_from_manifest(DATA + "/manifests/val.json", box_scale=DEFAULT_EXPORT_SCALE)
for v in gt.values():
    td = v.get("tile_dir")
    if td and not td.startswith("/"):
        v["tile_dir"] = os.path.normpath(os.path.join(DATA, td))   # resolves ../

labels = mine_proposal_dir(ROOT, "/tmp/m2_val", gt, split="val",
                           config=MiningConfig(score_thr=0.40), write=False)["labels"]
samples = build_samples(ROOT, "val", labels, gt_index=gt)          # <-- gt_index THIS time
pos = [s for s in samples if s.label == 1]
neg = [s for s in samples if s.label == 0]
have_mask = sum(1 for s in pos if s.tile_dir and os.path.exists(os.path.join(s.tile_dir, "mask.npy")))
print("COUNTS total=%d pos=%d mined_neg=%d | pos with mask=%d/%d" % (
    len(samples), len(pos), len(neg), have_mask, len(pos)), flush=True)

train = pos[:54] + neg[:600]
dev   = pos[54:66] + neg[600:750]
cal   = pos[66:] + neg[750:900]
print("splits train=%d(pos %d) dev=%d(pos %d) cal=%d(pos %d)" % (
    len(train), len(pos[:54]), len(dev), len(pos[54:66]), len(cal), len(pos[66:])), flush=True)

config = {"model": {"dim": 64, "roi_size": 6, "mask_size": 28, "num_hard_negatives": 1,
                    "use_evidential": True, "mask_decoder": "cnn",
                    "predict_mask_uncertainty": True, "fusion": "concat", "dropout": 0.0},
          "loss": {}}
summary = train_module2(config, train, cal, None,
                        train_cfg=TrainConfig(epochs=2, batch_size=32, lr=1e-4,
                                              out_dir="/root/autodl-tmp/m2_smoke",
                                              device="cuda", mask_size=28, balance=True),
                        dev_samples=dev)
for row in summary["history"]:
    print("EPOCH", json.dumps(row, ensure_ascii=False))
print("MASK_FRACTION_IN_SUMMARY", summary.get("num_train"))