"""
Geometric Algebra - Protocol Definitions

Protocols define the interfaces for geometric algebra objects. These enable
duck-typing while providing clear interface contracts for type checking.

GAObject: Any object living in a geometric algebra with a metric.
Transformable: Objects that can be transformed by motors/versors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from numpy.typing import NDArray


if TYPE_CHECKING:
    from morphis.geometry.model.metric import Metric
    from morphis.geometry.model.multivector import MultiVector

T = TypeVar("T", bound="GAObject")
S = TypeVar("S", bound="Transformable")


@runtime_checkable
class GAObject(Protocol):
    """
    Protocol for all geometric algebra objects.

    Every GA object (Blade, MultiVector, Frame) must have:
    - metric: The complete geometric context (tensor + signature + structure)
    - dim: The vector space dimension
    - data: The underlying numerical representation

    The metric provides all information needed for operations:
    - The metric tensor g_{ab} for inner products
    - The signature type (EUCLIDEAN, LORENTZIAN, DEGENERATE)
    - The structure type (FLAT, PROJECTIVE, CONFORMAL, ROUND)
    """

    @property
    def metric(self) -> Metric:
        """The complete geometric context for this object."""
        ...

    @property
    def dim(self) -> int:
        """Dimension of the underlying vector space."""
        ...

    @property
    def data(self) -> NDArray:
        """Underlying numerical data."""
        ...

    def with_metric(self: T, metric: Metric) -> T:
        """Return a copy with a different metric context."""
        ...


@runtime_checkable
class Transformable(Protocol):
    """
    Protocol for objects that can be transformed by motors/versors.

    Transformations in geometric algebra are typically sandwich products:
        x' = M x M^{-1}  or  x' = M x ~M

    Objects satisfying this protocol can be transformed in-place or copied.
    """

    def transform_by(self, motor: MultiVector) -> None:
        """
        Transform this object in-place by a motor/versor.

        The transformation applies the sandwich product M x ~M.

        Args:
            motor: The motor or versor to transform by
        """
        ...

    def copy(self: S) -> S:
        """Create a deep copy of this object."""
        ...
