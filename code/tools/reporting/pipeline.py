"""M3 EG-RR pipeline: proposal.v1 + verification.v1 + context -> report.v1.

Output contract (RESEARCH_ROADMAP 7.4):
    artifacts/reports/<candidate_id>.evidence_graph.json
    artifacts/reports/<candidate_id>.report.json
    artifacts/reports/<candidate_id>.report.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .contracts import M3Input, parse_m3_input
from .evidence_graph import EvidenceGraphBuilder
from .hazard_model import HazardResult, RuleHazardModel
from .llm_interface import EvidenceGroundedReporter, ReportValidator, TemplateReporter, to_markdown
from .uncertainty import UncertaintyResult, assess

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "reporting" / "reporting.yaml"

DEFAULT_KB: List[Dict[str, str]] = [
    {
        "doc_id": "kb_rainfall",
        "title": "降雨型滑坡证据规范",
        "text": "短历时强降雨和前期累计降雨共同触发浅层滑坡；必须与坡度和地表变化证据联合解释，跨区域阈值需要重新标定。",
    },
    {
        "doc_id": "kb_terrain",
        "title": "地形证据规范",
        "text": "坡度、曲率、高程与地形湿度指数共同影响滑坡易发性；单一地形因子不能作为独立结论。",
    },
    {
        "doc_id": "kb_sar",
        "title": "SAR 变化证据规范",
        "text": "VV/VH 变化与相干性下降可指示地表扰动，但裸土、道路切坡、采石和云影会产生相似信号，必须联合判别。",
    },
    {
        "doc_id": "kb_terminology",
        "title": "危险性、易发性与风险的区别",
        "text": "没有人口、基础设施和经济暴露数据时，只能输出危险性或易发性，不能输出完整风险等级。",
    },
]


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    import yaml

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class ReportingPipeline:
    def __init__(self, config: Optional[Mapping[str, Any]] = None, client: Optional[Any] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self.builder = EvidenceGraphBuilder(self.config)
        self.hazard_model = RuleHazardModel(self.config)
        self.reporter = EvidenceGroundedReporter(client=client, config=self.config)
        self.validator = ReportValidator(self.config)
        self.template = TemplateReporter(self.config)

    @classmethod
    def from_yaml(cls, path: Optional[str | Path] = None, client: Optional[Any] = None) -> "ReportingPipeline":
        return cls(load_config(path), client=client)

    # ------------------------------------------------------------------ run
    def run(
        self,
        payload: Mapping[str, Any] | M3Input,
        output_dir: Optional[str | Path] = None,
        write: bool = True,
    ) -> Tuple[Dict[str, Any], HazardResult, UncertaintyResult, Dict[str, Any]]:
        m3 = payload if isinstance(payload, M3Input) else parse_m3_input(payload)
        graph = self.builder.build(m3)
        hazard = self.hazard_model.predict(graph)
        uncertainty = assess(m3, graph, hazard, self.config)
        graph["hazard"] = hazard.to_dict()
        graph["uncertainty"] = uncertainty.to_dict()
        graph["risk_level"] = None
        graph["risk_level_reason"] = "缺少承灾体暴露数据，不能计算完整风险等级"
        graph["provenance"] = {
            "module": "M3-EG-RR",
            "schema_version": graph["schema_version"],
            "config": str(DEFAULT_CONFIG_PATH),
            "input_schema": m3.schema_version,
        }

        retrieved = self._retrieve(m3, graph)
        report = self.reporter.generate(m3, graph, hazard, uncertainty, retrieved)
        report.setdefault("evidence_graph_ref", f"{m3.candidate_id}.evidence_graph.json")

        result = self.validator.validate(report, m3, graph)
        report.setdefault("provenance", {})
        report["provenance"]["validated"] = bool(result["valid"])
        report["provenance"]["validator_issues"] = list(result["issues"])
        report["provenance"]["validator_metrics"] = result["metrics"]

        if write:
            self.write_artifacts(m3.candidate_id, graph, report, output_dir)
        return graph, hazard, uncertainty, report

    def run_file(self, input_path: str | Path, output_dir: Optional[str | Path] = None) -> Dict[str, Any]:
        with Path(input_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return self.run(payload, output_dir=output_dir)[3]

    # --------------------------------------------------------------- writing
    def write_artifacts(
        self,
        candidate_id: str,
        graph: Mapping[str, Any],
        report: Mapping[str, Any],
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Path]:
        out = Path(output_dir) if output_dir else REPO_ROOT / "artifacts" / "reports"
        out.mkdir(parents=True, exist_ok=True)
        graph_path = out / f"{candidate_id}.evidence_graph.json"
        report_path = out / f"{candidate_id}.report.json"
        markdown_path = out / f"{candidate_id}.report.md"
        graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(to_markdown(report), encoding="utf-8")
        return {"evidence_graph": graph_path, "report_json": report_path, "report_md": markdown_path}

    # -------------------------------------------------------------- retrieval
    @staticmethod
    def _retrieve(m3: M3Input, graph: Mapping[str, Any], k: Optional[int] = None) -> List[Dict[str, str]]:
        query_terms = {node.get("name") for node in graph.get("evidence") or []}
        query_terms |= {node.get("group") for node in graph.get("evidence") or []}
        scored: List[Tuple[int, Dict[str, str]]] = []
        for doc in DEFAULT_KB:
            text = doc["title"] + doc["text"]
            score = sum(1 for term in query_terms if term and str(term) in text)
            scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [doc for _, doc in scored[: (k or 4)]]
        return top


def run_reporting(input_path: str | Path, output_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    pipeline = ReportingPipeline.from_yaml()
    return pipeline.run_file(input_path, output_dir=output_dir)