"""Finite-precision continuous-CoT research harness.

The package deliberately separates persistent-state capacity from input access,
update depth, and sampled inspection.  See ``FINITE_COT_README.md`` for the
experimental contracts.
"""

from .access import (
    AccessViolation,
    LocalOracleAccess,
    ReadOnlyInputAccess,
    SampledInspectionAccess,
    SealedPrefixAccess,
)
from .quantization import FiniteScalarQuantizer, FiniteStateBottleneck
from .resources import ResourceLedger
from .transcript import DiscreteTranscript

__all__ = [
    "AccessViolation",
    "DiscreteTranscript",
    "FiniteScalarQuantizer",
    "FiniteStateBottleneck",
    "LocalOracleAccess",
    "ReadOnlyInputAccess",
    "ResourceLedger",
    "SampledInspectionAccess",
    "SealedPrefixAccess",
]
