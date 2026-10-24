import json, pathlib, statistics as st, math
base = pathlib.Path("/root/autodl-tmp/PF-RPN/artifacts")
SRC = {42: "hn_lb_metrics", 123: "hn_lb_seed123_metrics", 777: "hn_lb_seed777_metrics"}
CFG = ["A4_add_change", "A5_add_modality_token", "A6_add_cross_modal_router", "A7_no_router"]
out = []
for split in ("test", "val"):
    vals = {}
    for c in CFG:
        row = []
        for s, root in SRC.items():
            p = base / root / c / f"{split}.json"
            row.append(json.loads(p.read_text())["AR"]["AR@900"] if p.exists() else float("nan"))
        vals[c] = row
    out.append(f"### {split}")
    out.append("| 配置 | seed42 | seed123 | seed777 | mean±std |")
    out.append("|---|---:|---:|---:|---|")
    for c in CFG:
        r = vals[c]
        out.append("| %s | %.4f | %.4f | %.4f | %.4f±%.4f |" % (c, r[0], r[1], r[2], st.mean(r), st.pstdev(r)))
    out.append("")
    for a, b, lab in (("A4_add_change","A5_add_modality_token","+terrain token"),
                      ("A5_add_modality_token","A6_add_cross_modal_router","+router"),
                      ("A6_add_cross_modal_router","A7_no_router","A6 vs A7(去router)")):
        d = [vals[b][i] - vals[a][i] for i in range(3)]
        m, sd = st.mean(d), st.pstdev(d)
        if sd > 0:
            t = m / (sd / math.sqrt(3))
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        else:
            t, p = float("inf"), 0.0
        same = "三 seed 同号" if all(x > 0 for x in d) or all(x < 0 for x in d) else "符号不一致"
        out.append("- **%s**：per-seed Δ = %s；mean %+.4f ± %.4f；配对 t=%.2f, p≈%.3f；%s" % (
            lab, ["%+.4f" % x for x in d], m, sd, t, p, same))
    out.append("")
print("\n".join(out))
pathlib.Path("/root/autodl-tmp/multiseed.md").write_text("\n".join(out), encoding="utf-8")