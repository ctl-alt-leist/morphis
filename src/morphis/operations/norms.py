"""
Geometric Algebra - Norms

Norm operations on Elements: form (quadratic form), norm, and unit.
The metric is obtained directly from the element's metric attribute.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Element naming convention: v (never a, b, c for elements).
"""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING

from numpy import abs as np_abs, conj, einsum, newaxis, sqrt, where
from numpy.typing import NDArray

from morphis.config import TOLERANCE
from morphis.operations.structure import norm_squared_signature


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector


# =============================================================================
# Form (Quadratic Form)
# =============================================================================


def _form_v(v: Vector) -> NDArray:
    """
    Compute the quadratic form of a Vector: v · v.

    For grade-k vectors, this is the metric contraction with itself.
    Can be negative in non-Euclidean metrics.
    """
    k = v.grade
    g = v.metric

    if k == 0:
        return v.data * v.data

    sig = norm_squared_signature(k)
    metric_args = [g.data] * k
    return einsum(sig, *metric_args, v.data, v.data) / factorial(k)


def _form_mv(v: MultiVector) -> NDArray:
    """
    Compute the quadratic form of a MultiVector: scalar(v * ~v).

    This is the Clifford norm squared, appropriate for versors/rotors.
    For a rotor R, form(R) = 1.
    """
    from morphis.operations.products import geometric, grade_project

    v_rev = v.reverse()
    product = geometric(v, v_rev)
    scalar_part = grade_project(product, 0)
    return scalar_part.data


def form(v: Vector | MultiVector) -> NDArray:
    """
    Compute the quadratic form of an element.

    For Vector: v · v (metric inner product with itself)
    For MultiVector: scalar(v * ~v) (Clifford norm squared)

    Can be negative in non-Euclidean metrics.
    Returns scalar array with shape matching element's lot.
    """
    from morphis.elements.vector import Vector

    if isinstance(v, Vector):
        return _form_v(v)

    return _form_mv(v)


# =============================================================================
# Norm
# =============================================================================


def norm(v: Vector | MultiVector) -> NDArray:
    """
    Compute the norm of an element: sqrt(|form(v)|).

    Always returns non-negative values.
    Returns scalar array with shape matching element's lot.
    """
    return sqrt(np_abs(form(v)))


# =============================================================================
# Unit (Normalization)
# =============================================================================


def _unit_v(v: Vector) -> Vector:
    """Normalize a Vector to unit norm."""
    from morphis.elements.vector import Vector

    n = norm(v)
    n_expanded = n
    for _ in range(v.grade):
        n_expanded = n_expanded[..., newaxis]

    safe_norm = where(n_expanded > TOLERANCE, n_expanded, 1.0)
    result_data = v.data / safe_norm

    return Vector(
        data=result_data,
        grade=v.grade,
        metric=v.metric,
        lot=v.lot,
    )


def _unit_mv(v: MultiVector) -> MultiVector:
    """Normalize a MultiVector such that v * ~v = 1."""
    from morphis.elements.multivector import MultiVector

    n = norm(v)

    # Scale each grade component
    components = {}
    for grade, component in v.data.items():
        n_expanded = n
        for _ in range(grade):
            n_expanded = n_expanded[..., newaxis]

        safe_norm = where(n_expanded > TOLERANCE, n_expanded, 1.0)
        from morphis.elements.vector import Vector

        components[grade] = Vector(
            data=component.data / safe_norm,
            grade=grade,
            metric=component.metric,
            lot=component.lot,
        )

    return MultiVector(data=components, metric=v.metric)


def unit(v: Vector | MultiVector) -> Vector | MultiVector:
    """
    Normalize an element to unit norm.

    For Vector: returns v / norm(v) with norm(result) = 1
    For MultiVector: returns v / norm(v) with result * ~result = 1

    Handles zero elements safely by returning zero.
    """
    from morphis.elements.vector import Vector

    if isinstance(v, Vector):
        return _unit_v(v)

    return _unit_mv(v)


def conjugate(v: Vector) -> Vector:
    """
    Return vector with complex-conjugated coefficients.

    For real vectors, returns a copy (conjugation is identity on reals).
    For complex vectors, applies np.conj to all coefficients.

    This is the coefficient conjugation, not a GA operation. The complex
    numbers represent temporal phasors, not geometric structure.

    Returns Vector with conjugated data.
    """
    from morphis.elements.vector import Vector

    return Vector(
        data=conj(v.data),
        grade=v.grade,
        metric=v.metric,
        lot=v.lot,
    )


def hermitian_norm(v: Vector) -> NDArray:
    """
    Compute Hermitian norm: sqrt of hermitian_form.

    Always returns real non-negative values for positive-definite metrics.
    For real vectors, equivalent to norm.
    For complex vectors (phasors), gives the RMS amplitude.

    Returns real scalar array of norms with shape matching lot.
    """
    return sqrt(hermitian_form(v))


def hermitian_form(v: Vector) -> NDArray:
    """
    Compute Hermitian (sesquilinear) quadratic form:

        |v|^2_H = (1 / k!) conj(v^{m_1 ... m_k}) v^{n_1 ... n_k} g_{m_1 n_1} ... g_{m_k n_k}

    This is the physical magnitude squared, always real for real metrics.
    For real vectors, equivalent to form.
    For complex vectors (phasors), gives the squared RMS amplitude.

    Use this for physical quantities. Use form for algebraic
    (bilinear) inner product computations.

    Returns real scalar array with shape matching lot.
    """
    k = v.grade
    g = v.metric

    if k == 0:
        return (conj(v.data) * v.data).real

    sig = norm_squared_signature(k)
    metric_args = [g.data] * k
    result = einsum(sig, *metric_args, conj(v.data), v.data) / factorial(k)
    return result.real
