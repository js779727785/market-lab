from typing import TYPE_CHECKING, Any

from .base import CandidateModel, ModelArtifact, ModelRun
from .baselines import (
    MarketIndustryBaseline,
    MomentumBaseline,
    RandomBaseline,
    RegularizedLinearModel,
    ReversalBaseline,
)
from .calibration import ProbabilityCalibrator, ResidualIntervalCalibrator
from .registry import ModelRegistry

if TYPE_CHECKING:
    from .tree import LightGBMCandidate


def __getattr__(name: str) -> Any:
    if name == "LightGBMCandidate":
        from .tree import LightGBMCandidate

        return LightGBMCandidate
    raise AttributeError(name)


__all__ = [
    "CandidateModel",
    "LightGBMCandidate",
    "MarketIndustryBaseline",
    "ModelArtifact",
    "ModelRegistry",
    "ModelRun",
    "MomentumBaseline",
    "ProbabilityCalibrator",
    "RandomBaseline",
    "RegularizedLinearModel",
    "ResidualIntervalCalibrator",
    "ReversalBaseline",
]
