"""Evidence-grounded LLM interface for M3 EG-RR.

Implements RESEARCH_ROADMAP 7.5-7.6:

* the deterministic hazard result is computed before the LLM runs;
* the LLM receives only the evidence graph + hazard result + retrieved KB;
* it must output strict JSON, cite ``evidence_ids`` and never invent numbers;
* ``risk_level`` stays ``null`` unless exposure data exists;
* high-hazard candidates are always flagged for human review.

A deterministic template reporter is provided as the offline default and as a
safe fallback whenever the LLM output fails validation.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from .contracts import M3Input, SCHEMA_REPORT
from .hazard_model import Attribution, HazardResult
from .uncertainty import UncertaintyResult

SYSTEM_PROMPT = """你是滑坡遥感解译报告助手。你只能使用输入 JSON 中提供的证据。

硬性约束：
1. 只能使用提供的证据，不允许引入输入中不存在的事实。
2. 无法确认时必须明确说明“证据不足”，不得猜测。
3. 不输出输入中没有的数值。
4. 只输出严格 JSON，不要 markdown 代码块，不要额外解释。
5. 自然语言报告只能解释 JSON 中已有的结论。
6. 所有关键结论必须给出 evidence_ids。
7. 高风险候选必须标记 human_review_required=true。
8. 没有承灾体暴露数据时，risk_level 必须为 null，并说明原因。
"""

REPORT_V1_KEYS = (
    "schema_version",
    "candidate_id",
    "landslide_probability",
    "hazard_score",
    "hazard_level",
    "risk_level",
    "risk_level_reason",
    "evidence_ids",
    "uncertainty",
    "possible_false_positive_sources",
    "human_review_required",
    "report_text",
)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class LLMClient(Protocol):
    def complete(self, messages: List[Dict[str, str]]) -> str:
        ...


def build_messages(
    graph: Mapping[str, Any],
    hazard: HazardResult,
    uncertainty: UncertaintyResult,
    retrieved: Optional[Sequence[Any]] = None,
) -> List[Dict[str, str]]:
    payload: Dict[str, Any] = {
        "evidence_graph": graph,
        "hazard_model": hazard.to_dict(),
        "uncertainty": uncertainty.to_dict(),
        "output_schema": "report.v1",
        "required_keys": list(REPORT_V1_KEYS),
    }
    if retrieved:
        payload["knowledge_base"] = [
            {"doc_id": getattr(doc, "doc_id", None), "title": getattr(doc, "title", None), "text": getattr(doc, "text", None)}
            for doc in retrieved
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _allowed_numbers(m3: M3Input, graph: Mapping[str, Any], hazard: HazardResult, knowledge_text: str = "") -> set[str]:
    allowed: set[str] = set()

    def add(value: Any) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        for digits in (0, 1, 2, 3, 4):
            allowed.add(f"{number:.{digits}f}")
        if number.is_integer():
            allowed.add(str(int(number)))

    for node in graph.get("evidence") or []:
        add(node.get("value"))
        add(node.get("normalized_score"))
        for field in ("name", "unit", "method", "raw_key", "evidence_id"):
            for number in _NUMBER.findall(str(node.get(field) or "")):
                allowed.add(number)
                try:
                    add(float(number))
                except ValueError:
                    pass
    for value in (
        hazard.score,
        hazard.probability,
        m3.verification.probability,
        m3.verification.total_uncertainty,
        m3.proposal.objectness,
        m3.proposal.uncertainty,
    ):
        add(value)
    for attribution in hazard.attributions:
        add(attribution.contribution)
    for blob in (m3.candidate_id, knowledge_text):
        for number in _NUMBER.findall(str(blob or "")):
            allowed.add(number)
            try:
                add(float(number))
            except ValueError:
                pass
    for name in ("candidate_id", "event_id", "image_id"):
        value = (graph.get(name) or "")
        for number in _NUMBER.findall(str(value)):
            allowed.add(number)
    return allowed


def _sanitize(text: str, m3: M3Input, graph: Mapping[str, Any]) -> str:
    cleaned = str(text)
    tokens = [m3.candidate_id, m3.proposal.image_id or "", m3.proposal.event_id or "", str(graph.get("event_id") or "")]
    tokens += [str(node.get("evidence_id", "")) for node in (graph.get("evidence") or [])]
    for token in sorted({t for t in tokens if t}, key=len, reverse=True):
        cleaned = cleaned.replace(token, " ")
    return cleaned


class ReportValidator:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        cfg = (config or {}).get("report") or {}
        self.require_citation = bool(cfg.get("require_evidence_citation", True))
        self.forbid_new_numbers = bool(cfg.get("forbid_new_numbers", True))
        self.min_coverage = float(cfg.get("min_evidence_coverage", 0.5))
        self.require_exposure_for_risk = bool(cfg.get("require_exposure_for_risk", True))

    def validate(self, report: Mapping[str, Any], m3: M3Input, graph: Mapping[str, Any], knowledge_text: str = "") -> Dict[str, Any]:
        issues: List[str] = []
        evidence_ids = {str(node.get("evidence_id")) for node in (graph.get("evidence") or [])}
        for key in REPORT_V1_KEYS:
            if key not in report:
                issues.append(f"missing report.v1 field: {key}")
        if report.get("schema_version") != SCHEMA_REPORT:
            issues.append(f"schema_version must be {SCHEMA_REPORT}")
        if report.get("candidate_id") != m3.candidate_id:
            issues.append("candidate_id mismatch")
        if report.get("uncertainty") not in ("low", "medium", "high"):
            issues.append("uncertainty must be low/medium/high")

        if self.require_exposure_for_risk and not m3.has_exposure and report.get("risk_level") is not None:
            issues.append("risk_level must be null when exposure data is missing")
        if not m3.has_exposure and not report.get("risk_level_reason"):
            issues.append("risk_level_reason is required when exposure data is missing")

        referenced = {str(i) for i in (report.get("evidence_ids") or [])}
        unknown = sorted(referenced - evidence_ids)
        if unknown:
            issues.append(f"unknown evidence ids: {unknown}")
        if report.get("hazard_level") in ("high", "very_high") and not report.get("human_review_required"):
            issues.append("high hazard requires human_review_required=true")

        if not m3.has_exposure and self.require_exposure_for_risk:
            narrative = str(report.get("report_text", ""))
            negation = re.compile(r"(不是|并非|不能|无法|不应|禁止|不认为|而不是|不表示|不代表|不输出|不给出|no|not)\s*[^。；，,\n]{0,14}(风险|\brisk\b)", re.IGNORECASE)
            for match in re.finditer(r"风险|\brisk\b", narrative, re.IGNORECASE):
                window = narrative[max(0, match.start() - 24): match.end()]
                if not negation.search(window):
                    issues.append("risk wording used without exposure evidence")
                    break

        claims = report.get("claims") or []
        unsupported_claims = 0
        for claim in claims:
            ids = [str(i) for i in (claim.get("evidence_ids") or [])]
            bad = sorted(set(ids) - evidence_ids)
            if bad:
                issues.append(f"claim references unknown evidence: {bad}")
            if self.require_citation and not ids:
                issues.append("claim has no evidence citation")
                unsupported_claims += 1
            referenced.update(set(ids) - set(bad))
        coverage = len(referenced & evidence_ids) / max(1, len(evidence_ids))
        if coverage < self.min_coverage:
            issues.append(f"evidence coverage {coverage:.2f} < {self.min_coverage:.2f}")

        numeric_issues: List[str] = []
        if self.forbid_new_numbers:
            allowed = _allowed_numbers(m3, graph, _hazard_from_graph(graph), knowledge_text)
            blobs = [str(report.get("report_text", ""))]
            blobs += [str(claim.get("text", "")) for claim in claims]
            for blob in blobs:
                for token in _NUMBER.findall(_sanitize(blob, m3, graph)):
                    if token not in allowed and token.rstrip("0").rstrip(".") not in allowed:
                        numeric_issues.append(f"unsupported number {token}")
        issues.extend(numeric_issues)

        return {
            "valid": not issues,
            "issues": issues,
            "metrics": {
                "claim_count": len(claims),
                "unsupported_claim_count": unsupported_claims,
                "unsupported_number_count": len(numeric_issues),
                "hallucination_rate": (unsupported_claims + len(numeric_issues)) / max(1, len(claims)),
                "evidence_coverage": coverage,
            },
        }


def _hazard_from_graph(graph: Mapping[str, Any]) -> HazardResult:
    hazard = graph.get("hazard") or {}
    attributions = [
        Attribution(
            evidence_id=str(item.get("evidence_id")),
            contribution=float(item.get("contribution", 0.0)),
            direction=str(item.get("direction", "increase")),
        )
        for item in (hazard.get("attributions") or [])
        if item.get("evidence_id") is not None
    ]
    return HazardResult(
        score=float(hazard.get("score", 0.0)),
        probability=float(hazard.get("probability", 0.0)),
        level=str(hazard.get("level", "unknown")),
        attributions=attributions,
    )


def _risk_level_reason(m3: M3Input) -> str:
    if m3.has_exposure:
        return "存在承灾体暴露数据，可在后续版本中计算完整风险等级（v1 未启用）"
    return "缺少承灾体暴露数据，不能计算完整风险等级"


class TemplateReporter:
    """Deterministic evidence-grounded reporter (offline default / fallback)."""

    name = "template_evidence_grounded"

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = dict(config or {})

    def generate(
        self,
        m3: M3Input,
        graph: Mapping[str, Any],
        hazard: HazardResult,
        uncertainty: UncertaintyResult,
        retrieved: Optional[Sequence[Any]] = None,
        validator_issues: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        nodes = {str(n["evidence_id"]): n for n in (graph.get("evidence") or [])}
        ranked = [a for a in hazard.attributions if a.evidence_id in nodes][:5]
        claims: List[Dict[str, Any]] = []
        for attribution in ranked:
            node = nodes[attribution.evidence_id]
            claims.append(
                {
                    "claim_id": f"C-{attribution.evidence_id}",
                    "text": (
                        f"{node['name']}={node['value']}"
                        f"{(' ' + str(node['unit'])) if node.get('unit') else ''}，"
                        f"归一化得分 {float(node['normalized_score']):.2f}"
                    ),
                    "claim_type": "observation",
                    "evidence_ids": [attribution.evidence_id],
                    "confidence": round(max(0.3, 1.0 - float(node.get("uncertainty", 0.1))), 3),
                }
            )
        claims.append(
            {
                "claim_id": "C-HAZARD",
                "text": (
                    f"确定性危险性模型给出指数 {hazard.score:.2f}、概率 {hazard.probability:.2f}，"
                    f"等级 {hazard.level}"
                ),
                "claim_type": "inference",
                "evidence_ids": [a.evidence_id for a in ranked[:3]],
                "confidence": round(max(0.3, 1.0 - 0.5 * (1.0 - uncertainty.evidence_completeness)), 3),
            }
        )

        drivers = [
            {
                "evidence_id": attribution.evidence_id,
                "effect": "increase" if attribution.contribution >= 0 else "decrease",
                "statement": f"{nodes[attribution.evidence_id]['name']} 贡献 {attribution.contribution:+.3f}",
            }
            for attribution in ranked
        ]
        limitations: List[str] = list(uncertainty.reasons)
        if graph.get("missing_evidence"):
            limitations.append("缺失证据：" + ", ".join(graph["missing_evidence"][:8]))

        evidence_ids: List[str] = []
        for node in graph.get("evidence") or []:
            eid = str(node.get("evidence_id"))
            if eid not in evidence_ids:
                evidence_ids.append(eid)

        report_text = self._report_text(m3, graph, hazard, uncertainty, drivers)
        return {
            "schema_version": SCHEMA_REPORT,
            "candidate_id": m3.candidate_id,
            "landslide_probability": m3.verification.probability,
            "hazard_score": round(hazard.score, 6),
            "hazard_level": hazard.level,
            "risk_level": None,
            "risk_level_reason": _risk_level_reason(m3),
            "evidence_ids": evidence_ids,
            "uncertainty": uncertainty.level,
            "possible_false_positive_sources": m3.verification.top_hard_negatives(
                float((self.config.get("report") or {}).get("hard_negative_threshold", 0.05))
            ),
            "human_review_required": bool(uncertainty.human_review_required),
            "report_text": report_text,
            "claims": claims,
            "drivers": drivers,
            "limitations": limitations,
            "provenance": {
                "generator": self.name,
                "model": None,
                "prompt_version": (self.config.get("llm") or {}).get("prompt_version", "m3-egrr-v1"),
                "validated": False,
                "validator_issues": list(validator_issues or []),
            },
        }

    @staticmethod
    def _report_text(
        m3: M3Input,
        graph: Mapping[str, Any],
        hazard: HazardResult,
        uncertainty: UncertaintyResult,
        drivers: List[Dict[str, Any]],
    ) -> str:
        parts = [
            f"候选 {m3.candidate_id} 由 M1 提出、经 M2 验证，"
            f"M2 校准滑坡概率为 {m3.verification.probability if m3.verification.probability is not None else float('nan'):.2f}。"
        ]
        parts.append(
            f"确定性危险性模型给出指数 {hazard.score:.2f}，等级 {hazard.level}；"
            "该结果表示危险性/易发性，不表示完整风险。"
        )
        if drivers:
            parts.append("主要证据：" + "；".join(driver["statement"] for driver in drivers[:3]) + "。")
        if uncertainty.reasons:
            parts.append("不确定性说明：" + "；".join(uncertainty.reasons) + "。")
        if uncertainty.human_review_required:
            parts.append("建议人工复核候选边界与关键证据。")
        return "".join(parts)


class EvidenceGroundedReporter:
    """LLM-backed generator with strict validation and template fallback."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        config: Optional[Mapping[str, Any]] = None,
        validator: Optional[ReportValidator] = None,
        fallback: Optional[TemplateReporter] = None,
    ) -> None:
        self.client = client
        self.config = dict(config or {})
        self.validator = validator or ReportValidator(self.config)
        self.fallback = fallback or TemplateReporter(self.config)

    def generate(
        self,
        m3: M3Input,
        graph: Mapping[str, Any],
        hazard: HazardResult,
        uncertainty: UncertaintyResult,
        retrieved: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            report = self.fallback.generate(m3, graph, hazard, uncertainty, retrieved)
            result = self.validator.validate(report, m3, graph)
            report["provenance"]["validated"] = bool(result["valid"])
            report["provenance"]["validator_issues"] = list(result["issues"])
            report["provenance"]["validator_metrics"] = result["metrics"]
            return report

        knowledge_text = "\n".join(getattr(doc, "text", "") for doc in (retrieved or []))
        try:
            raw = self.client.complete(build_messages(graph, hazard, uncertainty, retrieved))
            payload = json.loads(_strip_code_fence(raw))
        except Exception as exc:  # noqa: BLE001 - fallback is intentional
            return self._fallback(m3, graph, hazard, uncertainty, retrieved, [f"LLM call/parse failed: {exc}"])

        result = self.validator.validate(payload, m3, graph, knowledge_text)
        if not result["valid"]:
            return self._fallback(m3, graph, hazard, uncertainty, retrieved, result["issues"])
        payload.setdefault("claims", [])
        payload.setdefault("limitations", [])
        payload["provenance"] = {
            "generator": "llm_evidence_grounded",
            "model": getattr(self.client, "model", None),
            "prompt_version": (self.config.get("llm") or {}).get("prompt_version", "m3-egrr-v1"),
            "validated": True,
            "validator_issues": [],
            "validator_metrics": result["metrics"],
        }
        return payload

    def _fallback(self, m3, graph, hazard, uncertainty, retrieved, issues):  # noqa: ANN001
        report = self.fallback.generate(m3, graph, hazard, uncertainty, retrieved, validator_issues=issues)
        report["provenance"]["generator"] = "template_fallback"
        report["provenance"]["validated"] = True
        report["provenance"]["validator_issues"] = list(issues)
        return report


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def to_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# 滑坡危险性报告：{report.get('candidate_id')}",
        "",
        f"- 危险性指数：{report.get('hazard_score')}",
        f"- 危险性等级：{report.get('hazard_level')}",
        f"- M2 校准滑坡概率：{report.get('landslide_probability')}",
        f"- 不确定性：{report.get('uncertainty')}",
        f"- 人工复核：{report.get('human_review_required')}",
        f"- 风险等级：{report.get('risk_level')}（{report.get('risk_level_reason')}）",
        "",
        "## 报告",
        "",
        str(report.get("report_text", "")),
        "",
        "## 主要证据",
        "",
    ]
    for driver in report.get("drivers") or []:
        lines.append(f"- `{driver.get('evidence_id')}` {driver.get('statement')}")
    if report.get("possible_false_positive_sources"):
        lines += ["", "## 可能误检来源", ""]
        lines += [f"- {item}" for item in report["possible_false_positive_sources"]]
    if report.get("limitations"):
        lines += ["", "## 局限性", ""]
        lines += [f"- {item}" for item in report["limitations"]]
    lines += ["", "## 证据引用", "", ", ".join(report.get("evidence_ids") or [])]
    return "\n".join(lines) + "\n"