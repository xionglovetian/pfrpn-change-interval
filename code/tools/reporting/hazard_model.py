"""Deterministic hazard / susceptibility model for M3 EG-RR.

v1 is an interpretable additive rule model. It is deliberately NOT an LLM:
the LLM only receives its output and may not change it. This satisfies
RESEARCH_ROADMAP sections 7.5 and 7.6.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

LEVELS = ("very_low", "low", "moderate", "high", "very_high")
DEFAULT_GROUP_WEIGHTS = {
    "terrain": 1.35,
    "spectral": 1.10,
    "sar": 0.95,
    "meteorology": 1.15,
    "environment": 0.85,
    "geometry": 0.70,
    "historical": 1.00,
    "model": 1.25,
}


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))


def level_from_score(score: float, thresholds: Sequence[float] = (0.20, 0.40, 0.60, 0.80)) -> str:
    for level, threshold in zip(LEVELS, thresholds):
        if score < threshold:
            return level
    return "very_high"


@dataclass
class Attribution:
    evidence_id: str
    contribution: float
    direction: str = "increase"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "contribution": round(float(self.contribution), 6),
            "direction": self.direction,
        }


@dataclass
class HazardResult:
    score: float
    probability: float
    level: str
    model_name: str = "rule_additive_v1"
    terminology: str = "hazard"
    attributions: List[Attribution] = field(default_factory=list)
    missing_groups: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 6),
            "probability": round(float(self.probability), 6),
            "level": self.level,
            "terminology": self.terminology,
            "model_name": self.model_name,
            "attributions": [a.to_dict() for a in self.attributions],
            "missing_groups": list(self.missing_groups),
        }


class RuleHazardModel:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        config = config or {}
        hm = config.get("hazard_model") or {}
        self.group_weights: Dict[str, float] = dict(hm.get("groups") or DEFAULT_GROUP_WEIGHTS)
        self.base_rate = float(hm.get("base_rate", 0.15))
        self.missing_weight = float(hm.get("missing_evidence_weight", 0.25))
        self.interactions: List[Dict[str, Any]] = list(hm.get("interactions") or [])
        self.thresholds: Sequence[float] = tuple(hm.get("level_thresholds") or (0.20, 0.40, 0.60, 0.80))

    def predict(self, graph: Mapping[str, Any]) -> HazardResult:
        nodes = [n for n in (graph.get("evidence") or []) if n.get("normalized_score") is not None]
        by_id = {n["evidence_id"]: n for n in nodes}

        # Per-group weighted mean.
        group_scores: Dict[str, float] = {}
        group_members: Dict[str, List[Tuple[str, float, float]]] = {}
        for group in self.group_weights:
            members: List[Tuple[str, float, float]] = []
            for node in nodes:
                if node.get("group") != group:
                    continue
                reliability = max(0.0, min(1.0, float(node.get("quality", 0.9)))) * (
                    1.0 - max(0.0, min(1.0, float(node.get("uncertainty", 0.1))))
                )
                if reliability <= 0:
                    continue
                members.append((str(node["evidence_id"]), float(node["normalized_score"]), reliability))
            if members:
                total_weight = sum(w for _, _, w in members)
                group_scores[group] = sum(score * weight for _, score, weight in members) / total_weight
                group_members[group] = members

        configured = list(self.group_weights)
        missing_groups = [g for g in configured if g not in group_scores]
        missing_ratio = len(missing_groups) / max(1, len(configured))

        denominator = sum(abs(self.group_weights[g]) for g in configured) or 1.0
        index = sum(self.group_weights[g] * group_scores.get(g, 0.0) for g in configured) / denominator
        index = index * (1.0 - self.missing_weight * missing_ratio) + self.base_rate * self.missing_weight * missing_ratio

        value = logit(self.base_rate)
        decay = 1.0 - self.missing_weight * missing_ratio
        contributions: Dict[str, float] = {}
        for group, weight in self.group_weights.items():
            if group not in group_scores:
                continue
            group_contribution = weight * (group_scores[group] - 0.5) * 2.0 * decay
            value += group_contribution
            share = group_contribution / max(1, len(group_members[group]))
            for evidence_id, _, reliability in group_members[group]:
                contributions[evidence_id] = contributions.get(evidence_id, 0.0) + share * reliability

        for term in self.interactions:
            keys = list(term.get("evidence") or [])
            values: List[float] = []
            ids: List[str] = []
            for key in keys:
                node = next((n for n in nodes if n.get("name") == key or n.get("evidence_id") == key), None)
                if node is None or node.get("normalized_score") is None:
                    values = []
                    break
                values.append(float(node["normalized_score"]))
                ids.append(str(node["evidence_id"]))
            if len(values) < 2:
                continue
            product = 1.0
            for value_item in values:
                product *= value_item
            term_value = float(term.get("weight", 0.0)) * (4.0 * product - 1.0)
            value += term_value
            for evidence_id in ids:
                contributions[evidence_id] = contributions.get(evidence_id, 0.0) + term_value / len(ids)

        attributions = [
            Attribution(eid, contribution, "increase" if contribution >= 0 else "decrease")
            for eid, contribution in sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ]
        probability = sigmoid(value)
        return HazardResult(
            score=max(0.0, min(1.0, index)),
            probability=probability,
            level=level_from_score(index, self.thresholds),
            attributions=attributions,
            missing_groups=sorted(missing_groups),
            notes=[f"missing_group_ratio={missing_ratio:.2f}"],
        )