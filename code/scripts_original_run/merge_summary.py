import json, pathlib
root = pathlib.Path("/root/autodl-tmp/PF-RPN/artifacts/hn_lb_metrics")
runs = ["A0_rgb","A1_optical_topography","A2_optical_sar","A3_optical_sar_topography",
        "A4_add_change","A5_add_modality_token","A6_add_cross_modal_router",
        "A7_no_router","A8_no_sar","A9_no_change"]
summary = {}
for r in runs:
    d = {}
    for split in ("test", "val"):
        p = root / r / f"{split}.json"
        if p.exists():
            d[split] = json.loads(p.read_text())
    if r == "A6_add_cross_modal_router":
        for p in sorted((root / r).glob("test.drop_*.json")):
            d[p.stem.replace("test.", "")] = json.loads(p.read_text())
    summary[r] = d
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("rebuilt summary.json with:", list(summary))
print("A6 keys:", [k for k in summary["A6_add_cross_modal_router"] if k.startswith("drop_")])