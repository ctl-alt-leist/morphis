"""
Geometric Algebra - Element Base Classes

Base classes for all geometric algebra objects. Every GA element has a metric
that defines its complete geometric context.

Hierarchy:
    Element (metric, collection)
    ├── GradedElement (+ data: NDArray, grade: int)
    │   ├── Blade
    │   └── Frame (+ span: int)
    └── CompositeElement (+ data: dict[int, GradedElement])
        └── MultiVector
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from morphis.elements.metric import Metric


if TYPE_CHECKING:
    from morphis.elements.blade import Blade


class Element(BaseModel):
    """
    Base class for all geometric algebra elements.

    Every GA element MUST have a metric that defines its geometric context.
    The metric provides:
    - The inner product structure (metric tensor g_{ab})
    - The signature type (EUCLIDEAN, LORENTZIAN, DEGENERATE)
    - The structure type (FLAT, PROJECTIVE, CONFORMAL, ROUND)

    Attributes:
        metric: The complete geometric context (required)
        collection: Shape of the collection dimensions
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    metric: Metric
    collection: tuple[int, ...] | None = None

    @property
    def dim(self) -> int:
        """Dimension of the underlying vector space."""
        return self.metric.dim


class GradedElement(Element):
    """
    Base class for elements with a single grade and array data.

    GradedElements store their geometric content in a NumPy array with shape
    (*collection, *geometric_shape) where geometric_shape depends on grade.

    Subclasses: Blade, Frame

    Attributes:
        data: The underlying array of element data
        grade: The grade (0=scalar, 1=vector, 2=bivector, etc.)
    """

    data: NDArray
    grade: int

    # Prevent numpy from intercepting arithmetic - force use of __rmul__ etc.
    __array_ufunc__ = None


class CompositeElement(Element):
    """
    Base class for elements composed of multiple grades.

    CompositeElements store components as a dictionary mapping grade to
    GradedElement (sparse representation).

    Subclasses: MultiVector

    Attributes:
        data: Dictionary mapping grade to component GradedElement
    """

    data: dict[int, "Blade"]
