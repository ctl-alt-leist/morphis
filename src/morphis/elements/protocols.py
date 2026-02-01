"""
Geometric Algebra - Protocol Definitions

Protocols define the interfaces for geometric algebra objects. These enable
duck-typing while providing clear interface contracts for type checking.

Graded: Objects with a single grade and array data (Vector, Frame).
Spanning: Objects with a span (Frame).
Transformable: Objects that can be transformed by motors/versors.
Indexable: Objects supporting index notation for contraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from numpy.typing import NDArray


if TYPE_CHECKING:
    from morphis.algebra.contraction import IndexedTensor
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


@runtime_checkable
class Indexable(Protocol):
    """
    Protocol for objects supporting index notation for contraction.

    Objects satisfying this protocol support:
    - String indexing for einsum-style contraction: obj["ab"]
    - Standard slicing for array access: obj[0, :, 1]

    Implemented by: Vector, Operator

    The protocol dispatches based on key type:
    - str → _index() → IndexedTensor for contraction
    - other → _slice() → standard array slicing
    """

    @property
    def data(self) -> NDArray:
        """Underlying numerical data."""
        ...

    def _index(self, indices: str) -> "IndexedTensor":
        """
        Create an IndexedTensor wrapper for einsum-style contraction.

        Args:
            indices: String of index labels, one per axis

        Returns:
            IndexedTensor wrapping this object
        """
        ...

    def _slice(self, key: Any) -> Any:
        """
        Slice into the underlying data.

        Args:
            key: Standard numpy indexing key

        Returns:
            Sliced result (type depends on implementation)
        """
        ...
