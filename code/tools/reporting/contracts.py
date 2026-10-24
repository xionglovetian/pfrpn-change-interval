"""M3 EG-RR input contracts and the adapter from M1/M2 artefacts.

Authority: ``E:\\PF-RPN\\RESEARCH_ROADMAP.md`` sections 3, 5.5, 6.4 and 7.3-7.4.

The adapter is deliberately strict: M3 consumes ``proposal.v1`` +
``verification.v1`` + context sections, and never silently accepts the old
``landslide-framework`` field names. Unknown or contradictory fields raise
``ContractError`` so that interface drift is caught by ``tests/test_reporting.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

SCHEMA_M3_INPUT = "m3_input.v1"
SCHEMA_PROPOSAL = "proposal.v1"
SCHEMA_VERIFICATION = "verification.v1"
SCHEMA_EVIDENCE_GRAPH = "evidence_graph.v1"
SCHEMA_REPORT = "report.v1"

EVIDENCE_GROUPS = (
    "geometry",
    "terrain",
    "spectral",
    "sar",
    "meteorology",
    "environment",
    "historical",
    "model",
)

HARD_NEGATIVE_TYPES = (
    "bare_soil",
    "road_cut",
    "quarry_or_construction",
    "cloud_shadow",
)

REQUIRED_SECTIONS = ("proposal", "verification")
OPTIONAL_SECTIONS = ("gis_context", "spectral_context", "sar_context", "historical_context", "exposure_context")


class ContractError(ValueError):
    """Raised when an upstream artefact violates the frozen interface."""


def _f(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prob(value: Any, name: str) -> Optional[float]:
    number = _f(value)
    if number is None:
        return None
    if not 0.0 <= number <= 1.0:
        raise ContractError(f"{name} must be in [0, 1], got {value!r}")
    return number


def _bbox(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ContractError("bbox_xyxy must be a list of four numbers")
    out = [_f(v) for v in value]
    if any(v is None for v in out):
        raise ContractError(f"bbox_xyxy contains non-numeric values: {value!r}")
    x1, y1, x2, y2 = out  # type: ignore[misc]
    if x2 <= x1 or y2 <= y1:
        raise ContractError(f"bbox_xyxy must satisfy x2>x1 and y2>y1, got {value!r}")
    return out  # type: ignore[return-value]


@dataclass
class ProposalCandidate:
    candidate_id: str
    bbox_xyxy: List[float]
    objectness: Optional[float] = None
    proposal_score: Optional[float] = None
    level: Optional[int] = None
    uncertainty: Optional[float] = None
    roi_feature_ref: Optional[str] = None
    image_id: Optional[str] = None
    event_id: Optional[str] = None
    schema_version: str = SCHEMA_PROPOSAL

    @property
    def bbox_width(self) -> float:
        return float(self.bbox_xyxy[2]) - float(self.bbox_xyxy[0])

    @property
    def bbox_height(self) -> float:
        return float(self.bbox_xyxy[3]) - float(self.bbox_xyxy[1])

    @property
    def bbox_area_px(self) -> float:
        return self.bbox_width * self.bbox_height

    @property
    def bbox_aspect_ratio(self) -> float:
        small = max(1e-6, min(self.bbox_width, self.bbox_height))
        return max(self.bbox_width, self.bbox_height) / small


@dataclass
class Verification:
    candidate_id: str
    is_landslide: Optional[bool] = None
    landslide_probability: Optional[float] = None
    calibrated_probability: Optional[float] = None
    hard_negative_probabilities: Dict[str, float] = field(default_factory=dict)
    mask: Optional[Dict[str, Any]] = None
    boundary_quality: Optional[float] = None
    uncertainty: Dict[str, float] = field(default_factory=dict)
    decision: Optional[str] = None
    review_required: bool = False
    schema_version: str = SCHEMA_VERIFICATION

    @property
    def probability(self) -> Optional[float]:
        return self.calibrated_probability if self.calibrated_probability is not None else self.landslide_probability

    @property
    def total_uncertainty(self) -> Optional[float]:
        if self.uncertainty.get("total") is not None:
            return _f(self.uncertainty.get("total"))
        parts = [_f(v) for v in self.uncertainty.values() if v is not None]
        return max(parts) if parts else None

    def top_hard_negatives(self, threshold: float = 0.05) -> List[str]:
        items = [(k, _f(v) or 0.0) for k, v in self.hard_negative_probabilities.items()]
        items = [pair for pair in items if pair[1] >= threshold]
        items.sort(key=lambda pair: pair[1], reverse=True)
        return [key for key, _ in items]


@dataclass
class M3Input:
    proposal: ProposalCandidate
    verification: Verification
    gis_context: Dict[str, Any] = field(default_factory=dict)
    spectral_context: Dict[str, Any] = field(default_factory=dict)
    sar_context: Dict[str, Any] = field(default_factory=dict)
    historical_context: Dict[str, Any] = field(default_factory=dict)
    exposure_context: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_M3_INPUT

    @property
    def candidate_id(self) -> str:
        return self.proposal.candidate_id

    @property
    def has_exposure(self) -> bool:
        return bool(self.exposure_context)


def parse_proposal(raw: Mapping[str, Any]) -> ProposalCandidate:
    if not isinstance(raw, Mapping):
        raise ContractError("proposal must be an object")
    candidate_id = raw.get("candidate_id")
    if not candidate_id:
        raise ContractError("proposal.candidate_id is required (proposal.v1)")
    bbox = _bbox(raw.get("bbox_xyxy"))
    if bbox is None:
        raise ContractError("proposal.bbox_xyxy is required (proposal.v1)")
    return ProposalCandidate(
        candidate_id=str(candidate_id),
        bbox_xyxy=bbox,
        objectness=_prob(raw.get("objectness"), "proposal.objectness"),
        proposal_score=_prob(raw.get("proposal_score"), "proposal.proposal_score"),
        level=int(raw["level"]) if raw.get("level") is not None else None,
        uncertainty=_prob(raw.get("uncertainty"), "proposal.uncertainty"),
        roi_feature_ref=raw.get("roi_feature_ref"),
        image_id=raw.get("image_id"),
        event_id=raw.get("event_id"),
        schema_version=str(raw.get("schema_version", SCHEMA_PROPOSAL)),
    )


def parse_verification(raw: Mapping[str, Any]) -> Verification:
    if not isinstance(raw, Mapping):
        raise ContractError("verification must be an object")
    candidate_id = raw.get("candidate_id")
    if not candidate_id:
        raise ContractError("verification.candidate_id is required (verification.v1)")
    hard_negatives: Dict[str, float] = {}
    for key, value in (raw.get("hard_negative_probabilities") or {}).items():
        probability = _prob(value, f"verification.hard_negative_probabilities.{key}")
        if probability is not None:
            hard_negatives[str(key)] = probability
    uncertainty_raw = raw.get("uncertainty") or {}
    if not isinstance(uncertainty_raw, Mapping):
        raise ContractError("verification.uncertainty must be an object")
    uncertainty = {
        str(key): value
        for key, value in uncertainty_raw.items()
        if _f(value) is not None
    }
    return Verification(
        candidate_id=str(candidate_id),
        is_landslide=bool(raw["is_landslide"]) if raw.get("is_landslide") is not None else None,
        landslide_probability=_prob(raw.get("landslide_probability"), "verification.landslide_probability"),
        calibrated_probability=_prob(raw.get("calibrated_probability"), "verification.calibrated_probability"),
        hard_negative_probabilities=hard_negatives,
        mask=dict(raw["mask"]) if isinstance(raw.get("mask"), Mapping) else None,
        boundary_quality=_prob(raw.get("boundary_quality"), "verification.boundary_quality"),
        uncertainty=uncertainty,  # type: ignore[arg-type]
        decision=raw.get("decision"),
        review_required=bool(raw.get("review_required", False)),
        schema_version=str(raw.get("schema_version", SCHEMA_VERIFICATION)),
    )


def parse_m3_input(payload: Mapping[str, Any]) -> M3Input:
    """Strict adapter: roadmap section 7.3 -> typed M3Input."""
    if not isinstance(payload, Mapping):
        raise ContractError("M3 input must be a JSON object")
    allowed = set(REQUIRED_SECTIONS) | set(OPTIONAL_SECTIONS)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ContractError(f"unknown M3 input sections: {unknown}")
    for section in REQUIRED_SECTIONS:
        if section not in payload:
            raise ContractError(f"missing required M3 input section: {section}")

    proposal = parse_proposal(payload["proposal"])
    verification = parse_verification(payload["verification"])
    if proposal.candidate_id != verification.candidate_id:
        raise ContractError(
            "candidate_id mismatch: "
            f"proposal={proposal.candidate_id!r} verification={verification.candidate_id!r}"
        )

    def section(name: str) -> Dict[str, Any]:
        value = payload.get(name) or {}
        if not isinstance(value, Mapping):
            raise ContractError(f"{name} must be an object")
        return dict(value)

    return M3Input(
        proposal=proposal,
        verification=verification,
        gis_context=section("gis_context"),
        spectral_context=section("spectral_context"),
        sar_context=section("sar_context"),
        historical_context=section("historical_context"),
        exposure_context=section("exposure_context"),
    )