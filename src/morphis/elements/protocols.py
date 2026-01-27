"""
Geometric Algebra - Protocol Definitions

Protocols define the interfaces for geometric algebra objects. These enable
duck-typing while providing clear interface contracts for type checking.

Graded: Objects with a single grade and array data (Vector, Frame).
Spanning: Objects with a span (Frame).
Transformable: Objects that can be transformed by motors/versors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from numpy.typing import NDArray


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector

G = TypeVar("G", bound="Graded")
S = TypeVar("S", bound="Transformable")


@runtime_checkable
class Graded(Protocol):
    """
    Protocol for objects with a single grade and array data.

    Implemented by: Vector, Frame

    Attributes:
        grade: The grade of the element (0=scalar, 1=vector, 2=bivector, etc.)
        data: The underlying numerical array
    """

    @property
    def grade(self) -> int:
        """The grade of this element."""
        ...

    @property
    def data(self) -> NDArray:
        """Underlying numerical data."""
        ...


@runtime_checkable
class Spanning(Protocol):
    """
    Protocol for objects that span a subspace.

    Implemented by: Frame

    Attributes:
        span: Number of elements (vectors) spanning the subspace
    """

    @property
    def span(self) -> int:
        """Number of spanning elements."""
        ...


@runtime_checkable
class Transformable(Protocol):
    """
    Protocol for objects that can be transformed by motors/versors.

    Transformations in geometric algebra are typically sandwich products:
        x' = M x M^{-1}  or  x' = M x ~M

    Objects satisfying this protocol can be transformed in-place or copied.
    """

    def transform(self, motor: MultiVector) -> None:
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
