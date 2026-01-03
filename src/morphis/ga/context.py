"""
Geometric Algebra - Context Management

Two-axis context system for geometric algebra: Signature (metric structure)
and Structure (geometric interpretation). Enables context-aware operations
and meaningful validation.

Examples:
    euclidean.flat        - Standard Euclidean GA
    euclidean.conformal   - Conformal GA (CGA)
    degenerate.projective - Projective GA (PGA)
    lorentzian.flat       - Spacetime GA (STA)
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class Signature(Enum):
    """
    Metric signature axis for geometric algebra contexts.

    Determines the metric structure of the underlying vector space.
    """

    EUCLIDEAN = auto()  # All positive: (+, +, +, ...)
    LORENTZIAN = auto()  # One timelike: (+, -, -, -) or (-, +, +, +)
    DEGENERATE = auto()  # One null direction: (0, +, +, ...)

    @classmethod
    def from_tuple(cls, sig: Tuple[int, ...]) -> "Signature":
        """
        Infer signature type from metric signature tuple.

        Returns DEGENERATE if any zeros, LORENTZIAN if any negatives,
        otherwise EUCLIDEAN.
        """
        if any(s == 0 for s in sig):
            return cls.DEGENERATE
        elif any(s < 0 for s in sig):
            return cls.LORENTZIAN
        else:
            return cls.EUCLIDEAN


class Structure(Enum):
    """
    Geometric structure axis for geometric algebra contexts.

    Determines the geometric interpretation and available operations.
    """

    FLAT = auto()  # Standard GA - vectors, planes, volumes
    PROJECTIVE = auto()  # Ideal points, incidence geometry
    CONFORMAL = auto()  # Angles, circles, inversions
    ROUND = auto()  # Combined projective + conformal


@dataclass(frozen=True)
class GeometricContext:
    """
    Two-axis context for geometric algebra blades.

    Combines metric structure (signature) and geometric interpretation
    (structure) to enable context-aware operations and validation.

    Immutable and hashable for use as dictionary keys and in sets.
    """

    signature: Signature
    structure: Structure

    def __repr__(self) -> str:
        return f"{self.signature.name.lower()}.{self.structure.name.lower()}"

    def is_compatible(self, other: "GeometricContext") -> bool:
        """Check if two contexts are compatible for operations."""
        return self.signature == other.signature and self.structure == other.structure

    @classmethod
    def merge(cls, *contexts: Optional["GeometricContext"]) -> Optional["GeometricContext"]:
        """
        Merge multiple contexts, returning None if incompatible.

        Rules:
        - All None -> None
        - Single non-None context -> that context
        - All matching -> that context
        - Any mismatch -> None
        """
        non_none = [c for c in contexts if c is not None]

        if not non_none:
            return None

        first = non_none[0]
        if all(c.is_compatible(first) for c in non_none[1:]):
            return first

        return None


class _ContextNamespace:
    """
    Namespace for context singletons.

    Provides clean API: euclidean.flat, degenerate.projective, etc.
    """

    def __init__(self, signature: Signature):
        self._signature = signature

    @property
    def flat(self) -> GeometricContext:
        """Standard geometric algebra context."""
        return GeometricContext(self._signature, Structure.FLAT)

    @property
    def projective(self) -> GeometricContext:
        """Projective geometric algebra context."""
        return GeometricContext(self._signature, Structure.PROJECTIVE)

    @property
    def conformal(self) -> GeometricContext:
        """Conformal geometric algebra context."""
        return GeometricContext(self._signature, Structure.CONFORMAL)

    @property
    def round(self) -> GeometricContext:
        """Round (projective + conformal) context."""
        return GeometricContext(self._signature, Structure.ROUND)


# Singleton namespaces for clean API
euclidean = _ContextNamespace(Signature.EUCLIDEAN)
lorentzian = _ContextNamespace(Signature.LORENTZIAN)
degenerate = _ContextNamespace(Signature.DEGENERATE)

# Common aliases
PGA = degenerate.projective
CGA = euclidean.conformal
STA = lorentzian.flat
