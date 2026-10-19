"""
Geometric Algebra - Complex Structure

A space declared with modes, metric(modes=n), is ℝ^{2n} with its directions
paired as e_m and e_ṁ. The pairing is the structure bivector

    J = e_m ∧ e_ṁ                    (summed over modes)

and the complex structure is the map it induces by contraction,

    J(v) = v ⌋ J,                    J(e_m) = e_ṁ,  J(e_ṁ) = -e_m

extended to every grade as an outermorphism. Blades are sorted by how they sit
relative to J: complex (J b = b), isotropic (J ⌋ b = 0), Lagrangian (isotropic
of grade n), and, for planes, the continuous Kähler angle cos θ = b · J.

Every function that takes J accepts any compatible structure bivector and
defaults to the one the metric declares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import abs as np_abs, all as np_all, arccos, asarray, clip, einsum, ones, sqrt, zeros
from numpy.typing import NDArray

from morphis.elements.metric import Dot, Metric
from morphis.operations.norms import norm
from morphis.operations.projections import interior_left


if TYPE_CHECKING:
    from morphis.elements.frame import Frame
    from morphis.elements.vector import Vector
    from morphis.operations.operator import Operator


# =============================================================================
# The Structure
# =============================================================================


def structure_bivector(metric: Metric) -> Vector:
    """
    The structure bivector J = e_m ∧ e_ṁ of a space with modes.

    Its unit-normalized top wedge is the pseudoscalar, J^{∧n} / n! = 𝟙.

    Raises:
        ValueError: If the metric declares no modes
    """
    from morphis.elements.vector import Vector

    if metric.n_modes == 0:
        raise ValueError(f"{metric!r} declares no modes; use metric(modes=n) for a space with complex dimensions")

    data = zeros((metric.dim, metric.dim))
    for m in range(1, metric.n_modes + 1):
        a, b = metric.to_internal(m), metric.to_internal(Dot(m))
        data[a, b] = 1.0
        data[b, a] = -1.0

    result = Vector(data, grade=2, metric=metric)

    return result


def contraction_map(B: Vector) -> Operator:
    """
    The vector map v ↦ v ⌋ B induced by a bivector.

    Its matrix is the bivector with one index lowered, (v ⌋ B)^ν = v^μ g_{μλ} B^{λν}.
    """
    from morphis.algebra.specs import VectorSpec
    from morphis.operations.operator import Operator

    if B.grade != 2 or B.lot != ():
        raise ValueError("contraction_map requires a single bivector")

    g = B.metric
    matrix = einsum("ml, ln -> nm", g.data, B.data)
    spec = VectorSpec(grade=1, lot=(), dim=g.dim)

    result = Operator(data=matrix, input_spec=spec, output_spec=spec, metric=g)

    return result


def complex_structure(metric: Metric) -> Operator:
    """
    The complex structure J(v) = v ⌋ J as an Operator.

    Applied to a grade-k vector it acts as the outermorphism, J(u ∧ v) = J u ∧ J v,
    so J² = (-1)^k on grade k. It broadcasts over lots.
    """
    return contraction_map(structure_bivector(metric))


def _structure(b: Vector | Frame, J: Vector | None) -> Vector:
    result = structure_bivector(b.metric) if J is None else J

    return result


def _n_modes(J: Vector) -> int:
    return J.dim // 2


# =============================================================================
# Kähler Angle
# =============================================================================


def kahler_angle(b: Vector | Frame, J: Vector | None = None) -> NDArray:
    """
    The Kähler angle θ of a plane, cos θ = b · J for a unit 2-blade b.

    Equivalently, θ is the angle between J u and v for any orthonormal pair
    spanning the plane. At θ = 0 or π the plane is a conjugate plane (the sign
    is its orientation); at θ = π/2 it is isotropic.

    Args:
        b: A 2-blade, or a Frame of span 2 such as a tangent frame; either may
            carry a lot, and the result has that lot
        J: Structure bivector; defaults to the metric's

    Returns:
        θ in [0, π], lot-shaped
    """
    from morphis.elements.frame import Frame

    J = _structure(b, J)
    g = J.metric

    if isinstance(b, Frame):
        if b.span != 2:
            raise ValueError(f"kahler_angle needs a plane, got a frame of span {b.span}")

        u, v = b.data[..., 0, :], b.data[..., 1, :]
        Ju = einsum("nm, ...m -> ...n", contraction_map(J).data, u)
        uu = einsum("mn, ...m, ...n -> ...", g.data, u, u)
        vv = einsum("mn, ...m, ...n -> ...", g.data, v, v)
        uv = einsum("mn, ...m, ...n -> ...", g.data, u, v)
        cosine = einsum("mn, ...m, ...n -> ...", g.data, Ju, v) / sqrt(uu * vv - uv**2)

    else:
        if b.grade != 2:
            raise ValueError(f"kahler_angle needs a 2-blade, got grade {b.grade}")

        # The bivector dot b · J, positive-definite: Σ_{μ<ν} b^{μν} J_{μν}
        dot_bJ = 0.5 * einsum("ma, nb, ...mn, ab -> ...", g.data, g.data, b.data, J.data)
        cosine = dot_bJ / norm(b)

    result = arccos(clip(cosine, -1.0, 1.0))

    return result


# =============================================================================
# Blade Classes
# =============================================================================


def _vanishes(x: Vector, scale: NDArray, tol: float) -> NDArray:
    """True where every geometric component of x is below tol relative to scale."""
    geo_axes = tuple(range(-x.grade, 0))
    bound = tol * asarray(scale).reshape(asarray(scale).shape + (1,) * x.grade)

    result = np_all(np_abs(x.data) <= bound, axis=geo_axes)

    return result


def is_complex(b: Vector, J: Vector | None = None, tol: float = 1e-9) -> NDArray:
    """
    True where the blade spans a complex subspace, J b = b.

    Complex subspaces have even dimension, so odd grades are never complex.
    The test is relative to the blade's norm.
    """
    J = _structure(b, J)
    image = contraction_map(J) * b

    result = _vanishes(image - b, norm(b), tol)

    return result


def is_isotropic(b: Vector, J: Vector | None = None, tol: float = 1e-9) -> NDArray:
    """
    True where the symplectic form vanishes on the blade's subspace, J ⌋ b = 0.

    Every blade of grade 0 or 1 is isotropic. The test is relative to the
    blade's norm.
    """
    J = _structure(b, J)

    if b.grade < 2:
        result = ones(b.lot, dtype=bool)

    else:
        result = _vanishes(interior_left(J, b), norm(b), tol)

    return result


def is_lagrangian(b: Vector, J: Vector | None = None, tol: float = 1e-9) -> NDArray:
    """True where the blade is isotropic of the maximal grade n."""
    J = _structure(b, J)

    if b.grade != _n_modes(J):
        result = zeros(b.lot, dtype=bool)

    else:
        result = is_isotropic(b, J, tol)

    return result
