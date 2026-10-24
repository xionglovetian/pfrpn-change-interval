"""Build ``evidence_graph.v1`` from the frozen M3 input contract.

Evidence is grouped into the 8 groups required by RESEARCH_ROADMAP section
7.2: geometry, terrain, spectral, sar, meteorology, environment, historical,
model. Every node keeps its raw value, unit, source and normalization score so
that a claim can be traced back to an upstream artefact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .contracts import EVIDENCE_GROUPS, M3Input, SCHEMA_EVIDENCE_GRAPH

DEFAULT_SPECS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "geometry": {
        "bbox_area_px": {"mode": "log_linear", "low": 400.0, "high": 400000.0},
        "bbox_aspect_ratio": {"mode": "ramp", "low": 1.0, "high": 6.0},
        "boundary_quality": {"mode": "linear", "low": 0.0, "high": 1.0},
    },
    "terrain": {
        "slope_mean_deg": {"mode": "ramp", "low": 10.0, "high": 40.0},
        "slope_max_deg": {"mode": "ramp", "low": 15.0, "high": 50.0},
        "curvature_mean": {"mode": "ramp", "low": -0.5, "high": 0.5},
        "elevation_mean_m": {"mode": "linear", "low": 200.0, "high": 2500.0},
        "aspect_mean_deg": {"mode": "context"},
    },
    "spectral": {
        "ndvi_drop": {"mode": "ramp", "low": 0.0, "high": 0.35},
        "ndwi_change": {"mode": "ramp", "low": -0.2, "high": 0.3},
        "nbr_drop": {"mode": "ramp", "low": 0.0, "high": 0.40},
    },
    "sar": {
        "vv_change_db": {"mode": "ramp", "low": 0.0, "high": 6.0},
        "vh_change_db": {"mode": "ramp", "low": 0.0, "high": 6.0},
        "coherence_drop": {"mode": "ramp", "low": 0.0, "high": 0.5},
    },
    "meteorology": {"rainfall_7d_mm": {"mode": "ramp", "low": 0.0, "high": 350.0}},
    "environment": {
        "fault_distance_m": {"mode": "inverse_ramp", "low": 0.0, "high": 5000.0},
        "river_distance_m": {"mode": "inverse_ramp", "low": 0.0, "high": 2000.0},
        "road_distance_m": {"mode": "inverse_ramp", "low": 0.0, "high": 1000.0},
    },
    "historical": {
        "historical_landslide_density": {"mode": "ramp", "low": 0.0, "high": 1.0},
        "distance_to_inventory_m": {"mode": "inverse_ramp", "low": 0.0, "high": 3000.0},
    },
    "model": {
        "objectness": {"mode": "linear", "low": 0.0, "high": 1.0},
        "proposal_score": {"mode": "linear", "low": 0.0, "high": 1.0},
        "calibrated_probability": {"mode": "linear", "low": 0.0, "high": 1.0},
        "verification_uncertainty": {"mode": "context"},
    },
}


@dataclass
class EvidenceNode:
    evidence_id: str
    group: str
    name: str
    value: Any
    unit: Optional[str] = None
    normalized_score: Optional[float] = None
    direction: str = "increase"
    quality: float = 0.9
    uncertainty: float = 0.1
    source: str = "unknown"
    method: str = "unknown"
    raw_key: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "group": self.group,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "normalized_score": None if self.normalized_score is None else round(float(self.normalized_score), 6),
            "direction": self.direction,
            "quality": round(float(self.quality), 6),
            "uncertainty": round(float(self.uncertainty), 6),
            "source": self.source,
            "method": self.method,
            "raw_key": self.raw_key,
            "notes": list(self.notes),
        }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize(value: Any, spec: Optional[Mapping[str, Any]]) -> Optional[float]:
    if spec is None or value is None or isinstance(value, bool):
        return None
    mode = str(spec.get("mode", "linear"))
    if mode == "context":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    low, high = spec.get("low"), spec.get("high")
    if mode == "passthrough":
        return _clip01(number)
    if low is None or high is None:
        return None
    low, high = float(low), float(high)
    if mode == "log_linear":
        if number <= 0 or low <= 0 or high <= 0:
            return None
        lo, hi, val = math.log10(low), math.log10(high), math.log10(number)
        score = 0.0 if hi == lo else (val - lo) / (hi - lo)
    else:
        score = 0.0 if high == low else (number - low) / (high - low)
    score = _clip01(score)
    if mode in ("inverse_linear", "inverse_ramp"):
        score = 1.0 - score
    return score


class EvidenceGraphBuilder:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.specs: Dict[str, Dict[str, Dict[str, Any]]] = (
            ((self.config.get("evidence") or {}).get("specs")) or DEFAULT_SPECS
        )
        self._nodes: List[EvidenceNode] = []

    # -------------------------------------------------------------- helpers
    def _spec(self, group: str, name: str) -> Optional[Mapping[str, Any]]:
        return (self.specs.get(group) or {}).get(name)

    def add(
        self,
        evidence_id: str,
        group: str,
        name: str,
        value: Any,
        unit: Optional[str] = None,
        source: str = "unknown",
        method: str = "unknown",
        raw_key: Optional[str] = None,
        quality: float = 0.9,
        uncertainty: float = 0.1,
        direction: Optional[str] = None,
        score_override: Optional[float] = None,
    ) -> Optional[EvidenceNode]:
        if value is None:
            return None
        spec = self._spec(group, name)
        score = score_override if score_override is not None else normalize(value, spec)
        if direction is None:
            mode = str((spec or {}).get("mode", "linear"))
            if mode == "context":
                direction = "context"
            elif "inverse" in mode:
                direction = "decrease"
            else:
                direction = str((spec or {}).get("direction", "increase"))
        node = EvidenceNode(
            evidence_id=evidence_id,
            group=group,
            name=name,
            value=value,
            unit=unit,
            normalized_score=score,
            direction=direction,
            quality=quality,
            uncertainty=uncertainty,
            source=source,
            method=method,
            raw_key=raw_key,
        )
        self._nodes.append(node)
        return node

    def _get(self, group: str, context: Mapping[str, Any], key: str, evidence_id: str, source: str, **kwargs: Any) -> Optional[EvidenceNode]:
        if key not in context or context.get(key) is None:
            return None
        return self.add(
            evidence_id=evidence_id,
            group=group,
            name=key,
            value=context.get(key),
            source=source,
            method=f"{source}.{key}",
            raw_key=key,
            **kwargs,
        )

    # ------------------------------------------------------------- builders
    def build(self, m3: M3Input) -> Dict[str, Any]:
        self._nodes = []
        p, v = m3.proposal, m3.verification

        # geometry (from M1 proposal + M2 boundary quality)
        self.add("EV-GEOM", "geometry", "bbox_area_px", p.bbox_area_px, "px^2", "proposal.v1", "bbox_area", "bbox_xyxy", quality=0.95, uncertainty=float(p.uncertainty or 0.05))
        self.add("EV-GEOM-AR", "geometry", "bbox_aspect_ratio", p.bbox_aspect_ratio, None, "proposal.v1", "bbox_aspect_ratio", "bbox_xyxy", quality=0.9, uncertainty=0.1)
        self.add("EV-BOUNDARY", "geometry", "boundary_quality", v.boundary_quality, None, "verification.v1", "boundary_quality", "boundary_quality", quality=0.9, uncertainty=float(v.total_uncertainty or 0.1))

        # terrain
        gis = m3.gis_context
        self._get("terrain", gis, "slope_mean_deg", "EV-SLOPE", "gis_context", unit="deg", quality=0.9, uncertainty=0.08)
        self._get("terrain", gis, "slope_max_deg", "EV-SLOPE-MAX", "gis_context", unit="deg", quality=0.9, uncertainty=0.08)
        self._get("terrain", gis, "aspect_mean_deg", "EV-ASPECT", "gis_context", unit="deg", quality=0.8, uncertainty=0.1)
        self._get("terrain", gis, "curvature_mean", "EV-CURV", "gis_context", quality=0.85, uncertainty=0.1)
        self._get("terrain", gis, "elevation_mean_m", "EV-ELEV", "gis_context", unit="m", quality=0.9, uncertainty=0.08)

        # spectral
        spec = m3.spectral_context
        self._get("spectral", spec, "ndvi_drop", "EV-NDVI", "spectral_context", quality=0.9, uncertainty=0.08)
        self._get("spectral", spec, "ndwi_change", "EV-NDWI", "spectral_context", quality=0.85, uncertainty=0.1)
        self._get("spectral", spec, "nbr_drop", "EV-NBR", "spectral_context", quality=0.9, uncertainty=0.08)

        # SAR
        sar = m3.sar_context
        vv = self._get("sar", sar, "vv_change_db", "EV-VV", "sar_context", unit="dB", quality=0.85, uncertainty=0.12)
        vh = self._get("sar", sar, "vh_change_db", "EV-VH", "sar_context", unit="dB", quality=0.85, uncertainty=0.12)
        coh = self._get("sar", sar, "coherence_drop", "EV-COH", "sar_context", quality=0.85, uncertainty=0.12)
        sar_scores = [n.normalized_score for n in (vv, vh, coh) if n is not None and n.normalized_score is not None]
        if sar_scores:
            composite = sum(sar_scores) / len(sar_scores)
            self.add(
                "EV-SAR", "sar", "sar_composite", round(composite, 6), None,
                "sar_context", "mean_normalized_sar", None, quality=0.8, uncertainty=0.15,
                score_override=composite,
            )

        # meteorology / environment / historical
        self._get("meteorology", gis, "rainfall_7d_mm", "EV-RAIN", "gis_context", unit="mm", quality=0.9, uncertainty=0.1)
        self._get("environment", gis, "fault_distance_m", "EV-FAULT", "gis_context", unit="m", quality=0.8, uncertainty=0.15)
        self._get("environment", gis, "river_distance_m", "EV-RIVER", "gis_context", unit="m", quality=0.8, uncertainty=0.15)
        self._get("environment", gis, "road_distance_m", "EV-ROAD", "gis_context", unit="m", quality=0.8, uncertainty=0.15)
        hist = m3.historical_context
        self._get("historical", hist, "historical_landslide_density", "EV-HIST-DENSITY", "historical_context", quality=0.75, uncertainty=0.2)
        self._get("historical", hist, "distance_to_inventory_m", "EV-HIST-DIST", "historical_context", unit="m", quality=0.75, uncertainty=0.2)

        # model evidence (M1 + M2)
        self.add("EV-OBJ", "model", "objectness", p.objectness, None, "proposal.v1", "objectness", "objectness", quality=0.95, uncertainty=float(p.uncertainty or 0.05))
        self.add("EV-PROPOSAL-SCORE", "model", "proposal_score", p.proposal_score, None, "proposal.v1", "proposal_score", "proposal_score", quality=0.9, uncertainty=float(p.uncertainty or 0.05))
        self.add("EV-VERIF", "model", "calibrated_probability", v.probability, None, "verification.v1", "calibrated_probability", "calibrated_probability", quality=0.95, uncertainty=float(v.total_uncertainty or 0.05))
        self.add("EV-UNCERT", "model", "verification_uncertainty", v.total_uncertainty, None, "verification.v1", "uncertainty.total", "uncertainty.total", quality=0.9, uncertainty=0.0)

        missing = self._missing(m3)
        edges = self._edges()
        return {
            "schema_version": SCHEMA_EVIDENCE_GRAPH,
            "candidate_id": m3.candidate_id,
            "event_id": p.event_id or (v.candidate_id.split("_")[0] if v.candidate_id else None),
            "image_id": p.image_id,
            "evidence": [n.to_dict() for n in self._nodes],
            "links": edges,
            "missing_evidence": missing,
        }

    def _edges(self) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        for node in self._nodes:
            score = node.normalized_score
            if score is None:
                relation, weight = "contextualizes", None
            else:
                relation = "supports" if score >= 0.5 else "weakens"
                weight = round((score - 0.5) * 2.0, 6)
            edges.append({"source": node.evidence_id, "target": "hypothesis:landslide", "relation": relation, "weight": weight})
        return edges

    def _missing(self, m3: M3Input) -> List[str]:
        missing: List[str] = []
        expected = {
            "gis_context": ("slope_mean_deg", "rainfall_7d_mm", "fault_distance_m", "river_distance_m", "road_distance_m"),
            "spectral_context": ("ndvi_drop", "ndwi_change", "nbr_drop"),
            "sar_context": ("vv_change_db", "vh_change_db", "coherence_drop"),
            "historical_context": ("historical_landslide_density", "distance_to_inventory_m"),
        }
        source = {
            "gis_context": m3.gis_context,
            "spectral_context": m3.spectral_context,
            "sar_context": m3.sar_context,
            "historical_context": m3.historical_context,
        }
        for section, keys in expected.items():
            for key in keys:
                if source[section].get(key) is None:
                    missing.append(f"{section}.{key}")
        return missing

    @property
    def nodes(self) -> List[EvidenceNode]:
        return list(self._nodes)