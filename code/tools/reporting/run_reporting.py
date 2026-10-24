"""CLI for M3 EG-RR.

Usage:
    python tools/reporting/run_reporting.py --input tools/reporting/examples/mock_m3_input.json
    python tools/reporting/run_reporting.py --input <in.json> --output-dir artifacts/reports
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.reporting.pipeline import ReportingPipeline, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 EG-RR evidence-grounded reporting")
    parser.add_argument("--input", required=True, help="m3_input.v1 JSON file")
    parser.add_argument("--output-dir", default=None, help="artifact directory (default artifacts/reports)")
    parser.add_argument("--config", default=None, help="configs/reporting/reporting.yaml")
    parser.add_argument("--no-write", action="store_true", help="run without writing artifacts")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    pipeline = ReportingPipeline(config)
    with Path(args.input).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _, hazard, uncertainty, report = pipeline.run(
        payload,
        output_dir=None if args.no_write else args.output_dir,
        write=not args.no_write,
    )

    print(f"candidate_id       : {report.get('candidate_id')}")
    print(f"hazard_score/level : {report.get('hazard_score')} / {report.get('hazard_level')}")
    print(f"hazard_probability : {hazard.probability:.4f}")
    print(f"uncertainty        : {report.get('uncertainty')} (completeness={uncertainty.evidence_completeness:.2f})")
    print(f"human_review       : {report.get('human_review_required')}")
    print(f"validated          : {report.get('provenance', {}).get('validated')}")
    issues = report.get("provenance", {}).get("validator_issues") or []
    if issues:
        print("validator_issues   :")
        for issue in issues:
            print(f"  - {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())