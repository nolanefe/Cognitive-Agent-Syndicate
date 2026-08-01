"""Cognitive Agent Syndicate — contract-driven multi-agent delivery pipeline."""

from cognitive_agent_syndicate.config import Settings
from cognitive_agent_syndicate.schemas import (
    AcceptanceCriterion,
    ArchitectureSpec,
    ArtifactBundle,
    ComponentSpec,
    DataModelSpec,
    EndpointSpec,
    GateResult,
    GeneratedFile,
    ReviewCategory,
    ReviewFinding,
    ReviewReport,
    RunReport,
    SystemBrief,
    UsageMetrics,
)

__all__ = [
    "AcceptanceCriterion",
    "ArchitectureSpec",
    "ArtifactBundle",
    "ComponentSpec",
    "DataModelSpec",
    "EndpointSpec",
    "GateResult",
    "GeneratedFile",
    "ReviewCategory",
    "ReviewFinding",
    "ReviewReport",
    "RunReport",
    "Settings",
    "SystemBrief",
    "UsageMetrics",
]

__version__ = "0.1.0"
