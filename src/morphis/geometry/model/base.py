"""
Geometric Algebra - Base Model

GAModel is the Pydantic base class for all GA objects (Blade, MultiVector, Frame).
It provides common configuration and enforces the required metric field.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from morphis.geometry.model.metric import Metric


class GAModel(BaseModel):
    """
    Base model for all geometric algebra objects.

    Every GA object MUST have a metric that defines its geometric context.
    The metric provides:
    - The inner product structure (metric tensor g_{ab})
    - The signature type (EUCLIDEAN, LORENTZIAN, DEGENERATE)
    - The structure type (FLAT, PROJECTIVE, CONFORMAL, ROUND)

    Subclasses: Blade, MultiVector, Frame

    Attributes:
        metric: The complete geometric context (required)
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    metric: Metric

    @property
    def dim(self) -> int:
        """Dimension of the underlying vector space."""
        return self.metric.dim
