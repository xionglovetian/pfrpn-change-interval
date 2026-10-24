"""M3 EG-RR: evidence graph + deterministic hazard model + evidence-grounded report."""

from .contracts import ContractError, M3Input, parse_m3_input
from .llm_interface import EvidenceGroundedReporter, ReportValidator, TemplateReporter, to_markdown
from .pipeline import ReportingPipeline, run_reporting

__all__ = [
    "ContractError",
    "EvidenceGroundedReporter",
    "M3Input",
    "ReportValidator",
    "ReportingPipeline",
    "TemplateReporter",
    "parse_m3_input",
    "run_reporting",
    "to_markdown",
]