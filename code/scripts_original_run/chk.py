import json, pathlib
p = pathlib.Path("/root/autodl-tmp/PF-RPN/artifacts/hn_lb_metrics/summary.json")
d = json.loads(p.read_text())
print("summary runs:", len(d))
print("A6 groups:", len([k for k in d.get("A6_add_cross_modal_router", {}) if k.startswith("drop_")]))
print("report exists:", pathlib.Path("/root/autodl-tmp/PF-RPN/docs/cloud_m1_results_hn_lb.md").exists())
print("phase2 ckpts:", len(list(pathlib.Path("/root/autodl-tmp/PF-RPN/work_dirs/m2_pf_rpn/p_lb").glob("*/epoch_8.pth"))))