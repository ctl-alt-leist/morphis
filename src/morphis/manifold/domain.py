"""
Manifolds - Domains

A domain is the coordinate region U of a patch X: U → V. It is a product of
factors, each either a real interval, possibly periodic, or a complex region
w ∈ [a, b] + i [c, d]. Products are written with *, and complex regions with
the imaginary unit:

    Interval(0, 1) * Circle()                    # a cylinder's coordinates
    Interval(0, π / 2) + 1j * Interval(-1, 1)    # one complex coordinate w

Every domain has real coordinates σ^1, ..., σ^k, and a complex factor
contributes two of them, σ = Re w and σ' = Im w, adjacent and in that order.
A grid over the domain is an array of shape (*shape, k), one lot axis per real
coordinate, which is how a patch is sampled.
"""

from __future__ import annotations

from numpy import asarray, linspace, meshgrid, pi, stack
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from morphis.topology import CellComplex


# =============================================================================
# Factors
# =============================================================================


class Interval(BaseModel):
    """
    A real coordinate ranging from start to stop.

    A periodic interval is a circle: stop is identified with start, and a grid
    over it leaves stop out.
    """

    model_config = ConfigDict(frozen=True)

    start: float
    stop: float
    periodic: bool = False

    def __init__(self, start: float, stop: float, periodic: bool = False):
        super().__init__(start=start, stop=stop, periodic=periodic)

    @property
    def length(self) -> float:
        return self.stop - self.start

    @property
    def span(self) -> int:
        return 1

    def samples(self, n: int) -> NDArray:
        """n evenly spaced values, the endpoint left out when periodic."""
        return linspace(self.start, self.stop, n, endpoint=not self.periodic)

    def __mul__(self, other) -> Domain:
        return Domain.of(self) * other

    def __rmul__(self, other) -> _Imaginary:
        if not (isinstance(other, complex) and other == 1j):
            return NotImplemented

        return _Imaginary(self)

    def __add__(self, other) -> ComplexRegion:
        if not isinstance(other, _Imaginary):
            return NotImplemented

        return ComplexRegion(real=self, imag=other.interval)

    def __repr__(self) -> str:
        bracket = ")" if self.periodic else "]"
        return f"[{self.start:g}, {self.stop:g}{bracket}"


class Circle(Interval):
    """A periodic coordinate on [0, length), 2π by default."""

    def __init__(self, length: float = 2 * pi):
        super().__init__(0.0, length, periodic=True)


class _Imaginary:
    """The imaginary part i I of a complex region, waiting for its real part."""

    __slots__ = ("interval",)

    def __init__(self, interval: Interval):
        self.interval = interval

    def __add__(self, other) -> ComplexRegion:
        if not isinstance(other, Interval):
            return NotImplemented

        return ComplexRegion(real=other, imag=self.interval)

    __radd__ = __add__


class ComplexRegion(BaseModel):
    """
    A complex coordinate w = σ + i σ' with σ in real and σ' in imag.

    Written as real + 1j * imag. A map of w is taken to be holomorphic in it.
    """

    model_config = ConfigDict(frozen=True)

    real: Interval
    imag: Interval

    @property
    def span(self) -> int:
        return 2

    def __mul__(self, other) -> Domain:
        return Domain.of(self) * other

    def __repr__(self) -> str:
        return f"{self.real!r} + i {self.imag!r}"


Factor = Interval | ComplexRegion


# =============================================================================
# Domain
# =============================================================================


class Domain(BaseModel):
    """
    A product of real intervals and complex regions.

    Attributes:
        factors: The factors in order; their real coordinates are concatenated
    """

    model_config = ConfigDict(frozen=True)

    factors: tuple[Factor, ...]

    @classmethod
    def of(cls, region: Domain | Factor) -> Domain:
        """The domain of a single factor, or the domain itself."""
        result = region if isinstance(region, Domain) else cls(factors=(region,))

        return result

    def __mul__(self, other) -> Domain:
        if not isinstance(other, (Domain, Interval, ComplexRegion)):
            return NotImplemented

        return Domain(factors=self.factors + Domain.of(other).factors)

    # =========================================================================
    # Coordinates
    # =========================================================================

    @property
    def span(self) -> int:
        """Number of real coordinates."""
        return sum(factor.span for factor in self.factors)

    @property
    def intervals(self) -> tuple[Interval, ...]:
        """The interval of each real coordinate."""
        result = []
        for factor in self.factors:
            result.extend([factor.real, factor.imag] if isinstance(factor, ComplexRegion) else [factor])

        return tuple(result)

    @property
    def periodic(self) -> tuple[bool, ...]:
        """Which real coordinates are periodic."""
        return tuple(interval.periodic for interval in self.intervals)

    @property
    def is_complex(self) -> bool:
        """Whether any factor is a complex coordinate."""
        return any(isinstance(factor, ComplexRegion) for factor in self.factors)

    def arguments(self, sigma: NDArray) -> list[NDArray]:
        """
        Split real coordinates into the arguments of a map, one per factor.

        A real factor receives σ^a; a complex factor receives w = σ^a + i σ^{a+1}.
        """
        sigma = asarray(sigma)
        result = []
        a = 0
        for factor in self.factors:
            if isinstance(factor, ComplexRegion):
                result.append(sigma[..., a] + 1j * sigma[..., a + 1])

            else:
                result.append(sigma[..., a])

            a += factor.span

        return result

    def _shape(self, shape: int | tuple[int, ...]) -> tuple[int, ...]:
        shape = (shape,) * self.span if isinstance(shape, int) else tuple(shape)
        if len(shape) != self.span:
            raise ValueError(f"need one grid size per real coordinate, got {len(shape)} for {self.span}")

        return shape

    def grid(self, shape: int | tuple[int, ...]) -> NDArray:
        """
        A coordinate grid of shape (*shape, span).

        Args:
            shape: Samples along each real coordinate, or one count for all
        """
        shape = self._shape(shape)
        axes = [interval.samples(n) for interval, n in zip(self.intervals, shape, strict=True)]

        result = stack(meshgrid(*axes, indexing="ij"), axis=-1)

        return result

    def cells(self, shape: int | tuple[int, ...]) -> CellComplex:
        """The cell complex of grid(shape), periodic coordinates closed into circles."""
        result = CellComplex.grid(self._shape(shape), periodic=self.periodic)

        return result

    def __repr__(self) -> str:
        return " × ".join(repr(factor) for factor in self.factors)
