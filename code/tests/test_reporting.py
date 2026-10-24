"""M3 EG-RR tests (RESEARCH_ROADMAP 7.8).

Run from the repository root:
    python -m unittest tests.test_reporting -v
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.reporting.contracts import ContractError, parse_m3_input  # noqa: E402
from tools.reporting.evidence_graph import EvidenceGraphBuilder  # noqa: E402
from tools.reporting.llm_interface import EvidenceGroundedReporter, ReportValidator  # noqa: E402
from tools.reporting.pipeline import ReportingPipeline, load_config  # noqa: E402

MOCK_PATH = REPO_ROOT / "tools" / "reporting" / "examples" / "mock_m3_input.json"
SCHEMA_DIR = REPO_ROOT / "tools" / "reporting" / "schemas"


def load_mock() -> dict:
    return json.loads(MOCK_PATH.read_text(encoding="utf-8"))


class FakeClient:
    model = "fake-llm"

    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, messages):  # noqa: ANN001
        return self.response


class M3ContractTest(unittest.TestCase):
    def test_mock_input_parses(self) -> None:
        m3 = parse_m3_input(load_mock())
        self.assertEqual(m3.candidate_id, "tile_000123_000001")
        self.assertEqual(m3.verification.candidate_id, m3.candidate_id)
        self.assertAlmostEqual(m3.verification.probability, 0.89)

    def test_old_contract_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            parse_m3_input({"instance": {}, "module1_output": {}, "module2_output": {}})

    def test_candidate_id_mismatch_is_rejected(self) -> None:
        payload = load_mock()
        payload["verification"]["candidate_id"] = "other_id"
        with self.assertRaises(ContractError):
            parse_m3_input(payload)

    def test_missing_bbox_is_rejected(self) -> None:
        payload = load_mock()
        del payload["proposal"]["bbox_xyxy"]
        with self.assertRaises(ContractError):
            parse_m3_input(payload)


class M3PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.pipeline = ReportingPipeline(cls.config)

    def test_evidence_graph_has_all_available_groups(self) -> None:
        m3 = parse_m3_input(load_mock())
        graph = EvidenceGraphBuilder(self.config).build(m3)
        groups = {node["group"] for node in graph["evidence"]}
        expected = {"geometry", "terrain", "spectral", "sar", "meteorology", "environment", "model"}
        self.assertTrue(expected.issubset(groups), groups)
        # Roadmap 7.3 does not supply historical context yet.
        self.assertNotIn("historical", groups)

    def test_pipeline_writes_three_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph, hazard, uncertainty, report = self.pipeline.run(load_mock(), output_dir=tmp)
            cid = report["candidate_id"]
            for suffix in ("evidence_graph.json", "report.json", "report.md"):
                self.assertTrue((Path(tmp) / f"{cid}.{suffix}").exists(), suffix)
            self.assertEqual(report["schema_version"], "report.v1")
            self.assertEqual(graph["schema_version"], "evidence_graph.v1")

    def test_report_and_graph_validate_against_schemas(self) -> None:
        try:
            import jsonschema
        except ImportError:  # pragma: no cover
            self.skipTest("jsonschema not installed")
        graph, _, _, report = self.pipeline.run(load_mock(), write=False)
        for schema_name, payload in (
            ("evidence_graph.schema.json", graph),
            ("report.schema.json", report),
        ):
            schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
            jsonschema.validate(instance=payload, schema=schema)

    def test_no_risk_level_without_exposure(self) -> None:
        _, _, _, report = self.pipeline.run(load_mock(), write=False)
        self.assertIsNone(report["risk_level"])
        self.assertTrue(report["risk_level_reason"])

    def test_high_hazard_requires_human_review(self) -> None:
        payload = copy.deepcopy(load_mock())
        payload["gis_context"].update(
            {
                "slope_mean_deg": 45.0,
                "slope_max_deg": 60.0,
                "rainfall_7d_mm": 340.0,
                "fault_distance_m": 50.0,
                "river_distance_m": 30.0,
            }
        )
        payload["spectral_context"] = {"ndvi_drop": 0.5, "ndwi_change": 0.3, "nbr_drop": 0.6}
        _, hazard, _, report = self.pipeline.run(payload, write=False)
        self.assertIn(hazard.level, {"high", "very_high"})
        self.assertTrue(report["human_review_required"])

    def test_report_evidence_ids_are_traceable(self) -> None:
        graph, _, _, report = self.pipeline.run(load_mock(), write=False)
        graph_ids = {node["evidence_id"] for node in graph["evidence"]}
        self.assertTrue(set(report["evidence_ids"]).issubset(graph_ids))
        for claim in report["claims"]:
            self.assertTrue(set(claim["evidence_ids"]).issubset(graph_ids), claim)

    def test_validator_rejects_unknown_evidence(self) -> None:
        m3 = parse_m3_input(load_mock())
        graph = EvidenceGraphBuilder(self.config).build(m3)
        graph["hazard"] = self.pipeline.hazard_model.predict(graph).to_dict()
        report = {
            "schema_version": "report.v1",
            "candidate_id": m3.candidate_id,
            "landslide_probability": 0.89,
            "hazard_score": 0.5,
            "hazard_level": "moderate",
            "risk_level": None,
            "risk_level_reason": "缺少承压体数据",
            "evidence_ids": ["EV-NOT-EXIST"],
            "uncertainty": "medium",
            "possible_false_positive_sources": [],
            "human_review_required": True,
            "report_text": "证据不足。",
            "claims": [{"claim_id": "c", "text": "存在滑坡", "claim_type": "observation", "evidence_ids": ["EV-NOT-EXIST"], "confidence": 0.5}],
        }
        result = ReportValidator(self.config).validate(report, m3, graph)
        self.assertFalse(result["valid"])
        self.assertTrue(any("unknown evidence" in issue for issue in result["issues"]))

    def test_llm_fallback_on_invalid_json(self) -> None:
        m3 = parse_m3_input(load_mock())
        graph = EvidenceGraphBuilder(self.config).build(m3)
        hazard = self.pipeline.hazard_model.predict(graph)
        from tools.reporting.uncertainty import assess

        uncertainty = assess(m3, graph, hazard, self.config)
        reporter = EvidenceGroundedReporter(client=FakeClient("not json"), config=self.config)
        report = reporter.generate(m3, graph, hazard, uncertainty)
        self.assertIn("template_fallback", report["provenance"]["generator"])
        self.assertTrue(report["provenance"]["validator_issues"])


    def test_validator_rejects_unnegated_risk_wording(self) -> None:
        m3 = parse_m3_input(load_mock())
        graph = EvidenceGraphBuilder(self.config).build(m3)
        graph["hazard"] = self.pipeline.hazard_model.predict(graph).to_dict()
        _, _, _, report = self.pipeline.run(load_mock(), write=False)
        report = copy.deepcopy(report)
        report["report_text"] = "该地区存在高风险，需要立即疏散。"
        result = ReportValidator(self.config).validate(report, m3, graph)
        self.assertFalse(result["valid"])
        self.assertTrue(any("risk wording" in issue for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()