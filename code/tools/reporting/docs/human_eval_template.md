# M3 人工评测模板（专家盲评）

> 用途：评价 `report.v1` 的事实一致性、证据覆盖率、解释合理性与人工修改工作量。
> 要求（路线 7.7 / 7.8）：至少两名专家独立盲评，报告评分者间一致性。

## 1. 评测单位

每条样本 = 一个候选实例：

- 候选截图或 RoI（可选，取决于是否需要看图复核）
- `evidence_graph.json`
- `report.json`
- `report.md`

专家看不到生成器名称、提示词版本、模型名称。

## 2. 评分表（每位专家每样本一份）

| 维度 | 评分 (1–5) | 说明 |
| --- | --- | --- |
| 事实一致性 |  | 1 = 多数结论与证据不符；5 = 所有结论均被证据支持 |
| 证据覆盖充分性 |  | 关键证据是否都被引用；1 = 大量遗漏；5 = 完整 |
| 危险性解释合理性 |  | 驱动因素是否物理合理、强度是否与证据匹配 |
| 误检来源解释 |  | `possible_false_positive_sources` 是否合理 |
| 术语合规性 |  | 无暴露数据时是否避免风险表述（是/否） |
| 不确定性表达 |  | 是否诚实反映证据不足与模型不确定度 |
| 报告可用性 |  | 是否可直接用于复核/简报；1 = 不可用；5 = 可直接使用 |
| 人工修改工作量 | 分钟 | 修订到可发布所需时间 |

补充字段：

- 结论采纳：`accept` / `accept_with_edits` / `reject`
- 主要错误类型（可多选）：`unsupported_claim` / `missing_evidence` /
  `wrong_driver` / `overclaim` / `terminology` / `other`
- 自由文本意见：

## 3. 机器可读记录（JSONL）

```json
{"sample_id":"tile_000123_000001","expert_id":"E1","blinded":true,
 "factual_consistency":5,"evidence_coverage":4,"hazard_explanation":4,
 "false_positive_explanation":3,"terminology_compliant":true,
 "uncertainty_expression":4,"usability":4,"edit_minutes":6,
 "decision":"accept_with_edits","error_types":["missing_evidence"],
 "comment":"历史证据缺失未在正文说明，仅出现在局限性。"}
```

## 4. 评分者间一致性

- 二分类/有序评分：报告 Cohen's κ；多评分者用 Fleiss' κ 或 ICC(2,1)。
- 连续项（人工修改分钟数）：报告 ICC 或中位数绝对差。
- 若 κ < 0.6，需复核评分标准并复评 10% 样本。

## 5. 消融对比设计

同一批候选上比较：

- C0：纯规则危险性模型（无 LLM）
- C1：LLM 无证据约束
- C2：LLM + 证据约束
- C3：LLM + 证据 + 风险模型（本模块完整版）

报告每个条件的：

- 幻觉率 = (无引用 claim + 新造数值 claim) / 总 claim
- 证据覆盖率 = 被引用证据节点 / 证据图节点
- 事实一致性通过率
- 风险排序一致性（若有人工排序标注）
- 专家评分均值与置信区间
- 平均人工修改时间

## 6. 最小样本量建议

- 主评测：≥ 100 个候选，覆盖 ≥ 3 个事件；
- 每个消融条件至少在同一子集上评测；
- 失败案例分析：≥ 15 个高不确定性或高危险性样本。