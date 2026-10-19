"""
Manifolds - Patches

A patch is a parametrization X: U → V of a k-dimensional submanifold of a flat
space V of any dimension and signature. Every geometric quantity is a function
of the point, computed in the ambient algebra, so one formula serves every
dimension and signature (Hestenes and Sobczyk, chapter 4):

    tangent frame           ∂_a X
    induced metric          h_ab = ∂_a X · ∂_b X
    tangent projection      P(v) = h^{ab} (v · ∂_b X) ∂_a X
    second fundamental form k_ab = P⊥(∂_a ∂_b X)
    mean curvature          k = (1/k) h^{ab} k_ab
    Riemann curvature       R_abcd = k_ac · k_bd - k_ad · k_bc

The last line is the Gauss equation, which gives the intrinsic curvature from
the extrinsic one in any codimension.

A map takes one argument per domain factor: a real array for an interval, a
complex array w for a complex region. It returns the point as a grade-1
Vector, as a sequence or array of its d real components, or, in a space with
modes, as its n complex amplitudes. A map of a complex coordinate is
holomorphic in it, so ∂_{Im w} X = J ∂_{Re w} X holds by construction.

Derivatives are fourth-order central differences of the map, evaluated off the
grid at σ ± h and σ ± 2h, so periodic coordinates need no special care.
"""

from __future__ import annotations

from typing import Callable

from numpy import (
    abs as np_abs,
    asarray,
    broadcast_arrays,
    broadcast_to,
    einsum,
    full,
    iscomplexobj,
    sqrt,
    stack,
    where,
    zeros,
)
from numpy.linalg import det, eigh, eigvalsh, inv, svd
from numpy.typing import NDArray

from morphis.elements.frame import Frame
from morphis.elements.metric import Metric
from morphis.elements.vector import Vector
from morphis.manifold.domain import ComplexRegion, Domain, Factor
from morphis.topology import CellComplex


# Fourth-order central-difference stencils: offsets in units of h, and weights
FIRST_OFFSETS = (-2, -1, 1, 2)
FIRST_WEIGHTS = (1 / 12, -8 / 12, 8 / 12, -1 / 12)
SECOND_OFFSETS = (-2, -1, 0, 1, 2)
SECOND_WEIGHTS = (-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12)


class Patch:
    """
    A parametrized submanifold X: U → V.

    Args:
        map: The parametrization, one argument per domain factor
        domain: The coordinate region U
        metric: The flat space V
        step: Finite-difference step h, in coordinate units

    Attributes:
        span: Intrinsic dimension k, the number of real coordinates
        dim: Ambient dimension d

    Examples:
        >>> torus = Patch(lambda θ, φ: [cos(θ), sin(θ), cos(φ), sin(φ)], Circle() * Circle(), metric(space=4))
        >>> σ = torus.grid(64)
        >>> torus.gaussian_curvature(σ)  # zero: the Clifford torus is flat
    """

    __slots__ = ("map", "domain", "metric", "step", "_bases", "_factors", "_cache")

    def __init__(self, map: Callable, domain: Domain | Factor, metric: Metric, step: float = 1e-3):
        domain = Domain.of(domain)
        if domain.is_complex and metric.n_modes == 0:
            raise ValueError(f"a holomorphic patch needs a space with modes, got {metric!r}")

        self.map = map
        self.domain = domain
        self.metric = metric
        self.step = step

        # Each real coordinate is differentiated along a base coordinate and scaled by a
        # factor: 1 for a real coordinate and for Re w, i for Im w, since ∂_{Im w} = i ∂_w
        bases, factors = [], []
        a = 0
        for factor in domain.factors:
            if isinstance(factor, ComplexRegion):
                bases.extend([a, a])
                factors.extend([1.0, 1j])

            else:
                bases.append(a)
                factors.append(1.0)

            a += factor.span

        self._bases = tuple(bases)
        self._factors = tuple(factors)
        self._cache: dict[str, tuple[bytes, NDArray]] = {}

    @property
    def span(self) -> int:
        return self.domain.span

    @property
    def dim(self) -> int:
        return self.metric.dim

    def __repr__(self) -> str:
        return f"Patch(span={self.span}, {self.metric!r}; {self.domain!r})"

    # =========================================================================
    # Sampling
    # =========================================================================

    def grid(self, shape: int | tuple[int, ...]) -> NDArray:
        """A coordinate grid of shape (*shape, span) over the domain."""
        return self.domain.grid(shape)

    def cells(self, shape: int | tuple[int, ...]) -> CellComplex:
        """The cell complex of grid(shape)."""
        return self.domain.cells(shape)

    def __call__(self, sigma: NDArray) -> Vector:
        """The points X(σ), a grade-1 Vector whose lot is σ's."""
        sigma = self._coordinates(sigma)

        result = Vector(self._components(self._raw(sigma)), grade=1, metric=self.metric)

        return result

    def _coordinates(self, sigma) -> NDArray:
        sigma = asarray(sigma, dtype=float)
        if sigma.ndim == 0 or sigma.shape[-1] != self.span:
            raise ValueError(f"coordinates need a last axis of size {self.span}, got shape {sigma.shape}")

        return sigma

    def _raw(self, sigma: NDArray) -> NDArray:
        """The map's output as an array: d real components, or n complex amplitudes."""
        output = self.map(*self.domain.arguments(sigma))

        if isinstance(output, Vector):
            data = output.data

        elif isinstance(output, (list, tuple)):
            data = stack(broadcast_arrays(*(asarray(c) for c in output)), axis=-1)

        else:
            data = asarray(output)

        is_amplitudes = iscomplexobj(data)
        width = self.metric.n_modes if is_amplitudes else self.dim
        if is_amplitudes and self.metric.n_modes == 0:
            raise TypeError(f"the map returned complex values, but {self.metric!r} declares no modes")

        if data.shape[-1] != width:
            reading = "complex amplitudes" if is_amplitudes else "real components"
            raise ValueError(f"the map returned {data.shape[-1]} {reading}, expected {width}")

        if self.domain.is_complex and not is_amplitudes:
            raise TypeError("a map of a complex coordinate must return complex amplitudes")

        result = broadcast_to(data, sigma.shape[:-1] + (width,))

        return result

    def _components(self, raw: NDArray) -> NDArray:
        """Real components of raw output, realizing amplitudes as v^m + i v^ṁ."""
        result = Vector.from_amplitudes(raw, self.metric).data if iscomplexobj(raw) else asarray(raw, dtype=float)

        return result

    # =========================================================================
    # Derivatives
    # =========================================================================

    def _cached(self, name: str, sigma: NDArray, compute: Callable[[NDArray], NDArray]) -> NDArray:
        """Remember the last evaluation of each derivative, since every quantity starts from them."""
        key = sigma.tobytes() + str(sigma.shape).encode()
        entry = self._cache.get(name)
        if entry is None or entry[0] != key:
            entry = (key, compute(sigma))
            self._cache[name] = entry

        return entry[1]

    def _shifted(self, sigma: NDArray, shifts: list[dict[int, float]]) -> NDArray:
        """Evaluate the raw map at σ displaced by each shift {coordinate: offset in h}, in one call."""
        displaced = []
        for shift in shifts:
            point = sigma.copy()
            for b, offset in shift.items():
                point[..., b] += offset * self.step

            displaced.append(point)

        result = self._raw(stack(displaced))

        return result

    def _first(self, sigma: NDArray) -> NDArray:
        """∂_a X as real components, shape (*lot, k, d)."""
        bases = sorted(set(self._bases))
        shifts = [{b: c} for b in bases for c in FIRST_OFFSETS]
        values = self._shifted(sigma, shifts).reshape((len(bases), len(FIRST_OFFSETS)) + sigma.shape[:-1] + (-1,))
        along = {
            b: einsum("c, c... -> ...", asarray(FIRST_WEIGHTS), values[n]) / self.step for n, b in enumerate(bases)
        }

        derivatives = [factor * along[b] for b, factor in zip(self._bases, self._factors, strict=True)]
        result = stack([self._components(raw) for raw in derivatives], axis=-2)

        return result

    def _second(self, sigma: NDArray) -> NDArray:
        """∂_a ∂_b X as real components, shape (*lot, k, k, d)."""
        bases = sorted(set(self._bases))
        lot = sigma.shape[:-1]
        along = {}
        for n, b in enumerate(bases):
            pure = self._shifted(sigma, [{b: c} for c in SECOND_OFFSETS])
            along[b, b] = einsum("c, c... -> ...", asarray(SECOND_WEIGHTS), pure) / self.step**2

            for c in bases[n + 1 :]:
                shifts = [{b: p, c: q} for p in FIRST_OFFSETS for q in FIRST_OFFSETS]
                mixed = self._shifted(sigma, shifts).reshape((len(FIRST_OFFSETS),) * 2 + lot + (-1,))
                weights = asarray(FIRST_WEIGHTS)
                along[b, c] = along[c, b] = einsum("p, q, pq... -> ...", weights, weights, mixed) / self.step**2

        k = self.span
        result = zeros(lot + (k, k, self.dim))
        for a in range(k):
            for b in range(k):
                raw = self._factors[a] * self._factors[b] * along[self._bases[a], self._bases[b]]
                result[..., a, b, :] = self._components(raw)

        return result

    def _frame_data(self, sigma) -> NDArray:
        sigma = self._coordinates(sigma)

        return self._cached("first", sigma, self._first)

    def _hessian_data(self, sigma) -> NDArray:
        sigma = self._coordinates(sigma)

        return self._cached("second", sigma, self._second)

    # =========================================================================
    # Tangent Space
    # =========================================================================

    def tangent_frame(self, sigma) -> Frame:
        """The coordinate frame ∂_a X, span k, lot of σ."""
        return Frame(data=self._frame_data(sigma), metric=self.metric)

    def tangent_blade(self, sigma) -> Vector:
        """The tangent k-blade t = ∂_1 X ∧ ... ∧ ∂_k X, built densely with d^k components per point."""
        return self.tangent_frame(sigma).to_vector()

    def induced_metric(self, sigma) -> NDArray:
        """h_ab = ∂_a X · ∂_b X, shape (*lot, k, k)."""
        E = self._frame_data(sigma)

        return einsum("...am, mn, ...bn -> ...ab", E, self.metric.data, E)

    def volume_element(self, sigma) -> NDArray:
        """√|det h|, the volume of the coordinate cell per unit coordinate volume."""
        return sqrt(np_abs(det(self.induced_metric(sigma))))

    def _tangent_coefficients(self, v: NDArray, E: NDArray, h_inverse: NDArray) -> NDArray:
        """Coefficients c^a of P(v) = c^a ∂_a X, broadcast over any extra axes v carries."""
        projections = einsum("...m, mn, ...bn -> ...b", v, self.metric.data, E)

        return einsum("...ab, ...b -> ...a", h_inverse, projections)

    def tangent_part(self, v: Vector, sigma) -> Vector:
        """
        The tangent projection P(v) = h^{ab} (v · ∂_b X) ∂_a X.

        v broadcasts against the lot of σ. The projection needs a nondegenerate h.
        """
        E = self._frame_data(sigma)
        coefficients = self._tangent_coefficients(v.data, E, inv(self.induced_metric(sigma)))

        result = Vector(einsum("...a, ...am -> ...m", coefficients, E), grade=1, metric=self.metric)

        return result

    def normal_part(self, v: Vector, sigma) -> Vector:
        """The normal projection P⊥(v) = v - P(v)."""
        return v - self.tangent_part(v, sigma)

    # =========================================================================
    # Normal Space
    # =========================================================================

    def normal_frame(self, sigma) -> Frame:
        """
        A frame of the normal space, orthonormal in the ambient metric.

        Each normal squares to ±1, or is left unscaled where the normal space is
        null. The frame is determined only up to a rotation of the normal space
        at each point, so it need not vary smoothly over a grid.
        """
        E = self._frame_data(sigma)
        k, d = self.span, self.dim
        covectors = einsum("...am, mn -> ...an", E, self.metric.data)
        _, _, rows = svd(covectors, full_matrices=True)
        normals = rows[..., k:, :]

        gram = einsum("...pm, mn, ...qn -> ...pq", normals, self.metric.data, normals)
        squares, rotation = eigh(gram)
        is_null = np_abs(squares) <= 1e-12 * np_abs(squares).max(axis=-1, keepdims=True).clip(min=1.0)
        scale = where(is_null, 1.0, 1.0 / sqrt(where(is_null, 1.0, np_abs(squares))))
        data = einsum("...pq, ...pm -> ...qm", rotation, normals) * scale[..., None]

        result = Frame(data=data.reshape(E.shape[:-2] + (d - k, d)), metric=self.metric)

        return result

    def normal_blade(self, sigma) -> Vector:
        """The normal (d - k)-blade, built densely."""
        return self.normal_frame(sigma).to_vector()

    # =========================================================================
    # Curvature
    # =========================================================================

    def second_fundamental_form(self, sigma) -> Vector:
        """
        k_ab = P⊥(∂_a ∂_b X), the normal part of the second derivatives.

        Returns a grade-1 Vector with lot (*lot, k, k), symmetric in a and b.
        """
        E = self._frame_data(sigma)
        S = self._hessian_data(sigma)
        h_inverse = inv(self.induced_metric(sigma))
        coefficients = self._tangent_coefficients(S, E[..., None, None, :, :], h_inverse[..., None, None, :, :])
        tangential = einsum("...abc, ...cm -> ...abm", coefficients, E)

        result = Vector(S - tangential, grade=1, metric=self.metric)

        return result

    def mean_curvature(self, sigma) -> Vector:
        """The mean curvature vector k = (1/k) h^{ab} k_ab."""
        h_inverse = inv(self.induced_metric(sigma))
        k_ab = self.second_fundamental_form(sigma).data

        result = Vector(einsum("...ab, ...abm -> ...m", h_inverse, k_ab) / self.span, grade=1, metric=self.metric)

        return result

    def riemann_curvature(self, sigma) -> NDArray:
        """R_abcd = k_ac · k_bd - k_ad · k_bc by the Gauss equation, shape (*lot, k, k, k, k)."""
        k_ab = self.second_fundamental_form(sigma).data
        products = einsum("...acm, mn, ...bdn -> ...abcd", k_ab, self.metric.data, k_ab)

        return products - products.swapaxes(-1, -2)

    def ricci_curvature(self, sigma) -> NDArray:
        """R_bd = h^{ac} R_abcd, shape (*lot, k, k)."""
        h_inverse = inv(self.induced_metric(sigma))

        return einsum("...ac, ...abcd -> ...bd", h_inverse, self.riemann_curvature(sigma))

    def scalar_curvature(self, sigma) -> NDArray:
        """R = h^{bd} R_bd."""
        h_inverse = inv(self.induced_metric(sigma))

        return einsum("...bd, ...bd -> ...", h_inverse, self.ricci_curvature(sigma))

    def gaussian_curvature(self, sigma) -> NDArray:
        """K = R_1212 / det h, for a surface."""
        if self.span != 2:
            raise ValueError(f"Gaussian curvature is defined for surfaces, got span {self.span}")

        result = self.riemann_curvature(sigma)[..., 0, 1, 0, 1] / det(self.induced_metric(sigma))

        return result

    # =========================================================================
    # Causal Character
    # =========================================================================

    def causal_character(self, sigma, tol: float = 1e-9) -> NDArray:
        """
        "spacelike", "timelike", or "null" at each point, from the inertia of h.

        A tangent space is null where h is degenerate, timelike where h has a
        direction of the time sign, and spacelike otherwise. The time sign is
        read from the metric's convention, so the answer is the same in both.
        """
        squares = eigvalsh(self.induced_metric(sigma))
        scale = np_abs(squares).max(axis=-1, keepdims=True).clip(min=1e-300)
        is_null = (np_abs(squares) <= tol * scale).any(axis=-1)

        has_time = self.metric.n_time > 0
        time_sign = 1.0 if self.metric.convention == "mostly minus" else -1.0
        is_timelike = (squares * time_sign > tol * scale).any(axis=-1) if has_time else zeros(squares.shape[:-1], bool)

        result = full(squares.shape[:-1], "spacelike", dtype="<U9")
        result = where(is_timelike, "timelike", result)
        result = where(is_null, "null", result)

        return result
