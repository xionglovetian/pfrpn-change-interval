"""Uncertainty aggregation and human-review policy for M3 EG-RR.

Maps evidence completeness + M2 verification uncertainty + hazard ambiguity to
the ``uncertainty`` string used by ``report.v1`` and to the
``human_review_required`` flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .contracts import EVIDENCE_GROUPS, M3Input
from .hazard_model import HazardResult


@dataclass
class UncertaintyResult:
    level: str
    evidence_completeness: float
    verification_uncertainty: Optional[float]
    human_review_required: bool
    insufficient_evidence: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "evidence_completeness": round(float(self.evidence_completeness), 6),
            "verification_uncertainty": None
            if self.verification_uncertainty is None
            else round(float(self.verification_uncertainty), 6),
            "human_review_required": bool(self.human_review_required),
            "insufficient_evidence": bool(self.insufficient_evidence),
            "reasons": list(self.reasons),
        }


def evidence_completeness(graph: Mapping[str, Any]) -> float:
    present = {
        str(node.get("group"))
        for node in (graph.get("evidence") or [])
        if node.get("normalized_score") is not None
    }
    return len(present & set(EVIDENCE_GROUPS)) / len(EVIDENCE_GROUPS)


def assess(
    m3: M3Input,
    graph: Mapping[str, Any],
    hazard: HazardResult,
    config: Optional[Mapping[str, Any]] = None,
) -> UncertaintyResult:
    cfg = (config or {}).get("uncertainty") or {}
    report_cfg = (config or {}).get("report") or {}
    min_completeness = float(cfg.get("min_completeness", 0.55))
    high_total = float((cfg.get("levels") or {}).get("high_total", 0.25))
    medium_total = float((cfg.get("levels") or {}).get("medium_total", 0.12))
    review_band = tuple(cfg.get("review_probability_band") or (0.35, 0.65))

    completeness = evidence_completeness(graph)
    verification_uncertainty = m3.verification.total_uncertainty
    insufficient = completeness < min_completeness

    signals: List[float] = [1.0 - completeness]
    if verification_uncertainty is not None:
        signals.append(float(verification_uncertainty))
    combined = max(signals)

    reasons: List[str] = []
    if insufficient:
        reasons.append(
            f"evidence completeness {completeness:.2f} < {min_completeness:.2f}; "
            f"missing groups: {hazard.missing_groups}"
        )
    if verification_uncertainty is not None and verification_uncertainty >= medium_total:
        reasons.append(f"verification uncertainty {verification_uncertainty:.2f}")
    if not (0.0 <= hazard.probability <= 1.0):
        reasons.append("hazard probability outside [0,1]")
    if review_band[0] <= hazard.probability <= review_band[1]:
        reasons.append(
            f"hazard probability {hazard.probability:.2f} lies in ambiguous band "
            f"[{review_band[0]:.2f}, {review_band[1]:.2f}]"
        )

    if insufficient or combined >= high_total:
        level = "high"
    elif combined >= medium_total or m3.verification.review_required:
        level = "medium"
    else:
        level = "low"

    high_hazard = hazard.level in ("high", "very_high")
    always_review = bool(report_cfg.get("always_review_high_hazard", True))
    human_review = bool(
        insufficient
        or level != "low"
        or m3.verification.review_required
        or (high_hazard and always_review)
        or m3.verification.decision == "reject"
    )
    if high_hazard and always_review:
        reasons.append(f"hazard level {hazard.level} requires human review")
    if m3.verification.review_required:
        reasons.append("M2 verification requested review")

    return UncertaintyResult(
        level=level,
        evidence_completeness=completeness,
        verification_uncertainty=verification_uncertainty,
        human_review_required=human_review,
        insufficient_evidence=insufficient,
        reasons=reasons,
    )