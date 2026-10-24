import json, pathlib, random
root = pathlib.Path("/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb/manifests")
items = json.loads((root / "train.json").read_text())
pos = [i for i in items if i["labels"]["annotations"]]
neg = [i for i in items if not i["labels"]["annotations"]]
random.Random(0).shuffle(pos); random.Random(0).shuffle(neg)
tiny = pos[:2] + neg[:2]
for name in ("overfit_train.json", "overfit_test.json"):
    (root / name).write_text(json.dumps(tiny, indent=2, ensure_ascii=False), encoding="utf-8")
print("tiny set:", [(i["image_id"], len(i["labels"]["annotations"])) for i in tiny])