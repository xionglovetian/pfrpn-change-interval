# M3 EG-RR: Evidence-Grounded Risk and Reporting

> 权威路线：`E:\PF-RPN\RESEARCH_ROADMAP.md` v0.1（section 7.1–7.8）
> 控制基线：`E:\PF-RPN\PROJECT_CONTROL.md`
> 写入范围：`tools/reporting/`、`configs/reporting/`、`tests/test_reporting.py`
> 状态：框架 + v1 最小实现；**真实数据/正式实验均为 NOT VERIFIED**

## 1. Mission

把 M1 候选（`proposal.v1`）和 M2 验证结果（`verification.v1`）转换成证据图，通过确定性模型计算危险性/易发性，再由证据约束的 LLM 生成可追溯报告。

M3 **不负责**：视觉检测、实例分类、掩膜分割、风险等级判定。

## 2. 输入契约（m3_input.v1）

严格按路线 section 7.3：

```json
{
  "proposal":     { "schema_version": "proposal.v1", ... },
  "verification": { "schema_version": "verification.v1", ... },
  "gis_context":  { "slope_mean_deg": 28.4, "rainfall_7d_mm": 182.0, ... },
  "spectral_context": { "ndvi_drop": 0.31, "ndwi_change": 0.08, "nbr_drop": 0.22 },
  "sar_context":  { "vv_change_db": 2.7, "vh_change_db": 1.9, "coherence_drop": 0.34 }
}
```

适配器：`tools/reporting/contracts.py::parse_m3_input`。旧契约（`instance` /
`module1_output` / `module2_output`）会被显式拒绝并抛出 `ContractError`，
防止接口漂移再次发生。

## 3. 输出契约

```text
artifacts/reports/<candidate_id>.evidence_graph.json   # evidence_graph.v1
artifacts/reports/<candidate_id>.report.json           # report.v1
artifacts/reports/<candidate_id>.report.md
```

`report.v1` 必需字段：`schema_version, candidate_id, landslide_probability,
hazard_score, hazard_level, risk_level, risk_level_reason, evidence_ids,
uncertainty, possible_false_positive_sources, human_review_required,
report_text`。schema 见 `tools/reporting/schemas/report.schema.json`。

## 4. 文件布局

```text
tools/reporting/
├── contracts.py          # proposal.v1 / verification.v1 / m3_input.v1 严格适配器
├── evidence_graph.py     # 8 组证据节点、归一化、支持/削弱边
├── hazard_model.py       # 确定性规则危险性模型 + 证据归因
├── uncertainty.py        # 完整性 / M2 不确定度 / 人工复核策略
├── llm_interface.py      # LLM 系统约束、prompt、validator、模板回退
├── pipeline.py           # 端到端编排与产物写出
├── run_reporting.py      # CLI
├── schemas/              # evidence_graph.v1 / report.v1 JSON Schema
├── examples/             # mock m3_input.v1
└── docs/
    ├── human_eval_template.md
    └── agent_handoff.md
configs/reporting/reporting.yaml
tests/test_reporting.py
```

## 5. 运行

```powershell
cd E:\PF-RPN
python tools\reporting\run_reporting.py --input tools\reporting\examples\mock_m3_input.json
python -m unittest discover -s tests -p 'test_reporting.py' -v
```

Mock 结果（仅为契约验证，不得作为论文指标）：

- 23 个证据节点，覆盖 geometry/terrain/spectral/sar/meteorology/environment/model
- `hazard_score` ≈ 0.54，`hazard_level` = moderate
- `uncertainty` = medium，`human_review_required` = true
- `risk_level` = null
- validator：0 违规，证据覆盖率 1.0，幻觉率 0.0（模板生成器）

## 6. 关键设计

1. **确定性优先**：危险性分数由规则模型计算，LLM 只能引用，不能改数。
2. **证据可追溯**：每个节点有 `evidence_id / value / unit / normalized_score /
   quality / uncertainty / source / method`；每条 claim 必须引用 evidence_id。
3. **三类校验**：引用存在性、数值可追溯性、术语与复核一致性。
4. **风险边界**：无承灾体暴露数据时 `risk_level=null`；把“风险”当作
   危险性/易发性表述，并拦截未加否定的 risk/风险 用词。
5. **安全回退**：LLM 输出非法 JSON 或校验失败时，自动回退确定性模板报告，
   并把失败原因写入 `provenance.validator_issues`。
6. **复核策略**：证据完整性不足、M2 不确定度过高、概率落入模糊区间、
   高风险等级、M2 请求复核时，`human_review_required=true`。

## 7. 接口问题（需要 M0/M1/M2/M4 确认）

1. **历史证据缺口**：路线 7.2 要求 M3 合并历史证据，但 7.3 输入契约没有
   历史字段。本实现支持可选 `historical_context`
   （`historical_landslide_density`、`distance_to_inventory_m`）；缺失时记为
   `missing_evidence` 并降低完整性。请确认该扩展是否写入路线。
2. **面积单位**：7.3 未提供 `resolution_m`，因此几何面积暂用像素面积
   `bbox_area_px`。若需要平方米，请确认 `gis_context.resolution_m`
   （或 `proposal`/`meta` 中的等效字段）。
3. **掩膜解码**：`verification.mask` 为 COCO RLE；v1 不解码，形状复杂度证据
   暂缺。若需要边界形态证据，请确认掩膜统一格式与解码依赖。
4. **`candidate_id` 归属**：路线要求 M1 输出 `candidate_id`，但 M2 输入契约
   同时携带该字段；请确认唯一分配方与冲突处理规则。
5. **暴露数据**：v1 无 `exposure_context`。若论文要报告 risk（而非 hazard），
   需要 M0 提供人口/基础设施/经济暴露字段并升级 schema 版本。

## 8. NOT VERIFIED

- 真实 M1/M2 产物的端到端运行：**NOT VERIFIED**（当前仅 mock `proposal.v1` /
  `verification.v1`）。
- 危险性模型权重与归一化区间：**NOT VERIFIED**（专家先验，未标定）。
- LLM 版本的事实一致性/幻觉率：**NOT VERIFIED**（当前默认模板生成器）。
- 专家评分与人工修改工作量：**NOT VERIFIED**（模板见 `docs/human_eval_template.md`）。
- 跨事件/跨区域泛化：**NOT VERIFIED**。

## 9. 下一步（P3 出口）

1. 接通 M1 `proposal.v1` 与 M2 `verification.v1` 真实产物；跑通同一最小样例。
2. 用事件级数据标定归一化区间与组权重，记录配置与随机种子。
3. 接入真实 LLM，跑 C1（无证据约束）/ C2（证据约束）/ C3（证据 + 风险模型）消融。
4. 用 `docs/human_eval_template.md` 完成至少两名专家的盲评与一致性统计。
5. 把本 README、`docs/agent_handoff.md` 与真实指标一起提交给 M4。