# Agent Handoff — M3 EG-RR

### Agent Handoff

- Agent: M3 EG-RR Agent (Codex)
- Module: M3 — 证据图 + 危险性/易发性模型 + 证据约束 LLM 报告
- Interface version:
  - 输入：`m3_input.v1`，消费 `proposal.v1` + `verification.v1` + `gis_context` /
    `spectral_context` / `sar_context`
  - 输出：`evidence_graph.v1` + `report.v1` + Markdown
- Files changed:
  - `tools/reporting/__init__.py`
  - `tools/reporting/contracts.py`
  - `tools/reporting/evidence_graph.py`
  - `tools/reporting/hazard_model.py`
  - `tools/reporting/uncertainty.py`
  - `tools/reporting/llm_interface.py`
  - `tools/reporting/pipeline.py`
  - `tools/reporting/run_reporting.py`
  - `tools/reporting/README.md`
  - `tools/reporting/schemas/evidence_graph.schema.json`
  - `tools/reporting/schemas/report.schema.json`
  - `tools/reporting/examples/mock_m3_input.json`
  - `tools/reporting/docs/human_eval_template.md`
  - `tools/reporting/docs/agent_handoff.md`
  - `configs/reporting/reporting.yaml`
  - `tests/test_reporting.py`
- Commands run:
  - `python tools\reporting\run_reporting.py --input tools\reporting\examples\mock_m3_input.json`
  - `python -m unittest discover -s tests -p 'test_reporting.py' -v` → 13 passed
  - `python -m compileall -q tools\reporting tests\test_reporting.py`
- Artifacts:
  - Mock（只验证契约，不是实验结果）：
    `artifacts/reports/tile_000123_000001.evidence_graph.json`,
    `.report.json`, `.report.md`
- Metrics:
  - Mock：23 证据节点；hazard_score ≈ 0.54 / moderate；uncertainty = medium；
    validator 0 违规；证据覆盖率 1.0；幻觉率 0.0（模板生成器）
  - **真实指标：NOT VERIFIED**（未接 M1/M2 真实产物，未标定权重，未跑 LLM）
- Ablations completed:
  - C0（纯规则危险性模型）可运行；
  - C1/C2/C3（LLM 无约束 / 证据约束 / 证据 + 风险模型）与专家评测：
    **NOT VERIFIED**
- Known limitations:
  - 归一化区间与组权重是专家先验，未用真实数据标定；
  - 规则模型未做概率校准与 conformal 区间（v1）；
  - COCO RLE 掩膜未解码，形状复杂度证据缺失；
  - 面积暂用像素面积，需要 `resolution_m` 才能输出平方米；
  - 历史证据在路线 7.3 输入契约中缺失（见下）。
- Blocking dependencies:
  1. M1 必须导出 `proposal.v1`（含候选 `candidate_id` 与 ROI 特征）；
  2. M2 必须导出 `verification.v1`（校准概率、困难负类、掩膜、不确定度）；
  3. 需确认可选 `historical_context` 扩展是否写入路线；
  4. 需确认 `resolution_m` 来源与掩膜统一格式；
  5. 若论文要报告 risk，需要 M0 提供承灾体暴露数据并升级 schema。
- Requested review:
  - M4 Integration Agent 审核 `m3_input.v1`/`report.v1` 与路线 7.3/7.4 的一致性；
  - M1/M2 负责人确认 `candidate_id` 唯一分配方与字段语义；
  - 用户确认历史证据与暴露数据的接口扩展。
- Next agent should read:
  - `E:\PF-RPN\RESEARCH_ROADMAP.md` section 3, 5.5, 6.4, 7.1–7.8, 12, 13
  - `E:\PF-RPN\PROJECT_CONTROL.md`
  - `E:\PF-RPN\tools\reporting\README.md`
  - `E:\PF-RPN\tools\reporting\docs\human_eval_template.md`

## 偏差登记（相对于 PROJECT_CONTROL.md）

| 原偏差 | 状态 | 说明 |
| --- | --- | --- |
| M3 输入契约不一致 | 已纠正 | 现严格消费 `{proposal, verification, gis_context, spectral_context, sar_context}`，旧契约抛 `ContractError` |
| M3 输出无版本化 schema | 已纠正 | 新增 `evidence_graph.v1` / `report.v1` JSON Schema 并做自动校验 |
| 术语风险 | 已纠正 | 无暴露数据时 `risk_level=null`，并拦截未加否定的 risk/风险 用词 |
| 历史证据缺口 | 待确认 | 路线 7.2 要求历史证据，7.3 未提供字段；本实现支持可选 `historical_context` |

## 复现边界

以上全部为 mock 契约验证。任何正式数值必须来自冻结后的 M1/M2 产物与一次性评测，
并记录 commit hash、配置、数据版本、随机种子与评测命令。