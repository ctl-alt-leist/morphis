"""
Geometric Algebra - Products

The geometric product, wedge product, and related operations. The geometric
product unifies the dot and wedge products into a single associative operation.
For vectors u and v:

    uv = u . v + u ^ v

This extends to all grades through systematic contraction and antisymmetrization.
Metrics are obtained from vector attributes and validated for compatibility.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Vector naming convention: u, v, w (never a, b, c for vectors).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import einsum, newaxis, zeros
from numpy.typing import NDArray

from morphis.operations._helpers import broadcast_collection_shape, get_broadcast_collection, get_common_dim
from morphis.operations.structure import (
    generalized_delta,
    geometric_normalization,
    geometric_signature,
    wedge_normalization,
    wedge_signature,
)


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector


# =============================================================================
# Wedge Product
# =============================================================================


def _wedge_vectors(*vectors: Vector) -> Vector:
    """
    Core wedge product implementation for Vectors only.

    Computes the exterior product via antisymmetrization:

        (u ^ v)^{mn} = u^m v^n - u^n v^m

    More generally for k vectors with grades (g_1, ..., g_k):

        B^{m_1 ... m_n} = outer^{a_1 ... a_n} delta^{m_1 ... m_n}_{a_1 ... a_n}

    where n = g_1 + ... + g_k and delta is the generalized Kronecker delta
    encoding antisymmetric structure.

    All vectors must have compatible metrics (validated via Metric.merge).

    Returns Vector of grade sum(grades), or zero vector if sum(grades) > dim.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.vector import Vector

    if not vectors:
        raise ValueError("wedge() requires at least one vector")

    # Merge metrics from all vectors (raises if incompatible)
    metric = Metric.merge(*(u.metric for u in vectors))

    # Single vector: return copy
    if len(vectors) == 1:
        u = vectors[0]
        return Vector(data=u.data.copy(), grade=u.grade, metric=u.metric, collection=u.collection)

    # Dimensional and grade calculations
    d = get_common_dim(*vectors)
    grades = tuple(u.grade for u in vectors)
    n = sum(grades)

    # Grade exceeds dimension: result is zero
    if n > d:
        # Need explicit shape for zero vector
        collection = get_broadcast_collection(*vectors)
        shape = collection + (d,) * n
        return Vector(data=zeros(shape), grade=n, metric=metric)

    # All scalars: just multiply
    if n == 0:
        result = vectors[0].data
        for u in vectors[1:]:
            result = result * u.data
        return Vector(data=result, grade=0, metric=metric)

    # Single einsum with delta contraction - let Vector infer collection from result
    sig = wedge_signature(grades)
    delta = generalized_delta(n, d)
    norm = wedge_normalization(grades)
    result = norm * einsum(sig, *[u.data for u in vectors], delta)

    return Vector(data=result, grade=n, metric=metric)


def wedge(*elements):
    """
    Wedge product: u ^ v ^ ... ^ w

    Generalized wedge product supporting Vector, MultiVector, and Frame operands.

    For Vectors only:
        Returns Vector of grade sum(grades).

    For MultiVectors:
        Distributes over components and returns MultiVector.

    For Frames:
        Distributes over frame vectors: (p ^ q) ^ {u, v, w} = {p ^ q ^ u, p ^ q ^ v, p ^ q ^ w}
        Returns Frame containing the wedged vectors.

    All operands must have compatible metrics.

    Args:
        *elements: Vectors, MultiVectors, or Frames to wedge together

    Returns:
        Vector, MultiVector, or Frame depending on inputs
    """
    from morphis.elements.frame import Frame
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector

    if not elements:
        raise ValueError("wedge() requires at least one element")

    # Check if any element is a Frame
    has_frame = any(isinstance(e, Frame) for e in elements)
    has_multivector = any(isinstance(e, MultiVector) for e in elements)

    # If we have a Frame, distribute over frame vectors
    if has_frame:
        # Find the Frame(s) and non-Frame elements
        frames = [e for e in elements if isinstance(e, Frame)]
        non_frames = [e for e in elements if not isinstance(e, Frame)]

        # For now, support single Frame at the end: (stuff) ^ Frame
        if len(frames) != 1:
            raise NotImplementedError("wedge with multiple Frames not yet supported")

        frame = frames[0]

        # Compute wedge of all non-frame elements first
        if non_frames:
            if len(non_frames) == 1:
                prefix = non_frames[0]
            else:
                prefix = wedge(*non_frames)

            # Distribute over frame vectors
            result_vectors = []
            for idx in range(frame.span):
                vec = frame.vector(idx)
                if isinstance(prefix, (Vector, MultiVector)):
                    result_vectors.append(wedge(prefix, vec))
                else:
                    result_vectors.append(vec)

            return Frame(*result_vectors)
        else:
            # Just the Frame itself - wedge all its vectors together
            vectors = [frame.vector(idx) for idx in range(frame.span)]
            return _wedge_vectors(*vectors)

    # If we have MultiVectors but no Frame
    if has_multivector:
        # Convert all to MultiVectors and use pairwise multiplication
        Ms = []
        for e in elements:
            if isinstance(e, Vector):
                Ms.append(MultiVector(data={e.grade: e}, metric=e.metric))
            elif isinstance(e, MultiVector):
                Ms.append(e)
            else:
                raise TypeError(f"wedge() does not support {type(e)}")

        # Compute pairwise
        result = Ms[0]
        for M in Ms[1:]:
            result = _wedge_mv_mv(result, M)
        return result

    # All Vectors - use the core implementation
    if all(isinstance(e, Vector) for e in elements):
        return _wedge_vectors(*elements)

    raise TypeError("wedge() arguments must be Vector, MultiVector, or Frame")


def _wedge_v_mv(u: Vector, M: MultiVector) -> MultiVector:
    """
    Wedge product of vector with multivector: u ^ M

    Distributes over components: u ^ (A + B + ...) = (u ^ A) + (u ^ B) + ...

    All components must have compatible metrics.

    Returns MultiVector.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    result_components: dict[int, Vector] = {}

    for _k, component in M.data.items():
        product = wedge(u, component)
        result_grade = product.grade

        if result_grade in result_components:
            result_components[result_grade] = result_components[result_grade] + product
        else:
            result_components[result_grade] = product

    return MultiVector(data=result_components, metric=Metric.merge(u.metric, M.metric))


def _wedge_mv_v(M: MultiVector, u: Vector) -> MultiVector:
    """
    Wedge product of multivector with vector: M ^ u

    Distributes over components.

    All components must have compatible metrics.

    Returns MultiVector.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    result_components: dict[int, Vector] = {}

    for _k, component in M.data.items():
        product = wedge(component, u)
        result_grade = product.grade

        if result_grade in result_components:
            result_components[result_grade] = result_components[result_grade] + product
        else:
            result_components[result_grade] = product

    return MultiVector(data=result_components, metric=Metric.merge(M.metric, u.metric))


def _wedge_mv_mv(M: MultiVector, N: MultiVector) -> MultiVector:
    """
    Wedge product of two multivectors: M ^ N

    Computes all pairwise wedge products of components and sums.

    All components must have compatible metrics.

    Returns MultiVector.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    result_components: dict[int, Vector] = {}

    for _k1, vec1 in M.data.items():
        for _k2, vec2 in N.data.items():
            product = wedge(vec1, vec2)
            result_grade = product.grade

            if result_grade in result_components:
                result_components[result_grade] = result_components[result_grade] + product
            else:
                result_components[result_grade] = product

    return MultiVector(data=result_components, metric=Metric.merge(M.metric, N.metric))


# =============================================================================
# Antiwedge (Regressive) Product
# =============================================================================


def _antiwedge_vectors(u: Vector, v: Vector) -> Vector:
    """
    Core antiwedge implementation for two Vectors.

    Computes u ∨ v = complement(complement(u) ∧ complement(v))

    Returns Vector of grade (j + k - d).
    """
    from morphis.operations.duality import right_complement

    u_comp = right_complement(u)
    v_comp = right_complement(v)
    joined = _wedge_vectors(u_comp, v_comp)

    return right_complement(joined)


def antiwedge(*elements):
    """
    Antiwedge (regressive) product: u ∨ v ∨ ... ∨ w

    Generalized antiwedge supporting Vector, MultiVector, and Frame operands.

    The regressive product is dual to the wedge product. It computes the
    intersection of subspaces, just as wedge computes the union (span).

    Defined via duality:

        u ∨ v = complement(complement(u) ∧ complement(v))

    For grade-j vector u and grade-k vector v in d dimensions:
    - Result grade: j + k - d
    - Returns zero vector if j + k < d (subspaces don't intersect generically)

    For Frames:
        Distributes over frame vectors: (p ∨ q) ∨ {u, v, w} = {p ∨ q ∨ u, p ∨ q ∨ v, p ∨ q ∨ w}
        Returns Frame containing the antiwedged vectors.

    Also known as: regressive product, vee product, meet.

    All operands must have compatible metrics.

    Args:
        *elements: Vectors, MultiVectors, or Frames

    Returns:
        Vector, MultiVector, or Frame depending on inputs
    """
    from morphis.elements.frame import Frame
    from morphis.elements.vector import Vector

    if not elements:
        raise ValueError("antiwedge() requires at least one element")

    if len(elements) == 1:
        return elements[0]

    # Check if any element is a Frame
    has_frame = any(isinstance(e, Frame) for e in elements)

    # If we have a Frame, distribute over frame vectors
    if has_frame:
        frames = [e for e in elements if isinstance(e, Frame)]
        non_frames = [e for e in elements if not isinstance(e, Frame)]

        if len(frames) != 1:
            raise NotImplementedError("antiwedge with multiple Frames not yet supported")

        frame = frames[0]

        if non_frames:
            if len(non_frames) == 1:
                prefix = non_frames[0]
            else:
                prefix = antiwedge(*non_frames)

            result_vectors = []
            for idx in range(frame.span):
                vec = frame.vector(idx)
                result_vectors.append(antiwedge(prefix, vec))

            return Frame(*result_vectors)
        else:
            # Just the Frame - antiwedge all its vectors
            vectors = [frame.vector(idx) for idx in range(frame.span)]
            result = vectors[0]
            for v in vectors[1:]:
                result = _antiwedge_vectors(result, v)
            return result

    # All Vectors - compute pairwise
    if all(isinstance(e, Vector) for e in elements):
        result = elements[0]
        for e in elements[1:]:
            result = _antiwedge_vectors(result, e)
        return result

    raise TypeError("antiwedge() arguments must be Vector, MultiVector, or Frame")


# =============================================================================
# Geometric Product (Vector x Vector)
# =============================================================================


def _geometric_v_v(u: Vector, v: Vector) -> MultiVector:
    """
    Geometric product of two vectors: uv = sum of <uv>_r over grades r

    For vectors of grade j and k, produces components at grades
    |j - k|, |j - k| + 2, ..., j + k (same parity as j + k).

    Each grade r corresponds to (j + k - r)/2 metric contractions.

    Both vectors must have compatible metrics (validated via Metric.merge).

    Returns MultiVector containing all nonzero grade components.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(u.metric, v.metric)
    g = metric

    d = get_common_dim(u, v)
    j, k = u.grade, v.grade

    components = {}

    # Compute all grade components
    min_grade = abs(j - k)
    max_grade = min(j + k, d)

    for r in range(min_grade, max_grade + 1, 2):
        # Number of contractions for this grade
        c = (j + k - r) // 2

        if c < 0 or c > min(j, k):
            continue

        # Build einsum signature and compute
        sig = geometric_signature(j, k, c)
        norm = geometric_normalization(j, k, c)

        # Prepare arguments for einsum
        metric_args = [g.data] * c

        if r == 0:
            # Scalar result
            result_data = norm * einsum(sig, *metric_args, u.data, v.data)
            component = Vector(result_data, grade=0, metric=metric)

        elif c == 0:
            # Pure wedge (no contractions)
            delta = generalized_delta(r, d)
            result_data = norm * einsum(sig, u.data, v.data, delta)
            component = Vector(data=result_data, grade=r, metric=metric)

        else:
            # Mixed: contractions + antisymmetrization
            delta = generalized_delta(r, d)
            result_data = norm * einsum(sig, *metric_args, u.data, v.data, delta)
            component = Vector(data=result_data, grade=r, metric=metric)

        components[r] = component

    return MultiVector(data=components, metric=metric)


# =============================================================================
# Geometric Product (MultiVector x MultiVector)
# =============================================================================


def _geometric_mv_mv(M: MultiVector, N: MultiVector) -> MultiVector:
    """
    Geometric product of two multivectors.

    Computes all pairwise geometric products of components and sums.

    Both multivectors must have compatible metrics.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(M.metric, N.metric)

    result_components: dict[int, Vector] = {}

    for _k1, u in M.data.items():
        for _k2, v in N.data.items():
            product = _geometric_v_v(u, v)
            for grade, component in product.data.items():
                if grade in result_components:
                    result_components[grade] = result_components[grade] + component
                else:
                    result_components[grade] = component

    return MultiVector(data=result_components, metric=metric)


# =============================================================================
# Public Interface
# =============================================================================


def geometric(u, v):
    """
    Geometric product: uv = sum of <uv>_r over grades r

    Generalized geometric product supporting Vector, MultiVector, and Frame operands.

    For Vectors/MultiVectors:
        For vectors of grade j and k, produces components at grades
        |j - k|, |j - k| + 2, ..., j + k (same parity as j + k).
        Each grade r corresponds to (j + k - r) / 2 metric contractions.
        Returns MultiVector containing all nonzero grade components.

    For Frames:
        Distributes over frame vectors: M * {u, v, w} = {M * u, M * v, M * w}
        Returns Frame containing the multiplied vectors (extracting grade-1 components).

    Both operands must have compatible metrics (validated via Metric.merge).

    Args:
        u: Vector, MultiVector, or Frame
        v: Vector, MultiVector, or Frame

    Returns:
        MultiVector or Frame depending on inputs
    """
    from morphis.elements.frame import Frame
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector

    # Handle Frame inputs
    if isinstance(u, Frame) or isinstance(v, Frame):
        if isinstance(u, Frame) and isinstance(v, Frame):
            raise NotImplementedError("geometric product Frame * Frame not yet supported")

        if isinstance(v, Frame):
            # u * Frame - distribute over frame vectors
            result_vectors = []
            for idx in range(v.span):
                vec = v.vector(idx)
                product = geometric(u, vec)
                # Extract grade-1 component for Frame
                grade1 = product.grade_select(1)
                if grade1 is not None:
                    result_vectors.append(grade1)
                else:
                    raise ValueError("geometric product did not produce grade-1 component for Frame")
            return Frame(*result_vectors)
        else:
            # Frame * v - distribute over frame vectors
            result_vectors = []
            for idx in range(u.span):
                vec = u.vector(idx)
                product = geometric(vec, v)
                grade1 = product.grade_select(1)
                if grade1 is not None:
                    result_vectors.append(grade1)
                else:
                    raise ValueError("geometric product did not produce grade-1 component for Frame")
            return Frame(*result_vectors)

    # Both vectors
    if isinstance(u, Vector) and isinstance(v, Vector):
        return _geometric_v_v(u, v)

    # Convert vectors to multivectors if needed
    if isinstance(u, Vector):
        u = MultiVector(data={u.grade: u}, metric=u.metric)

    if isinstance(v, Vector):
        v = MultiVector(data={v.grade: v}, metric=v.metric)

    return _geometric_mv_mv(u, v)


# =============================================================================
# Grade Projection
# =============================================================================


def grade_project(M: MultiVector, k: int) -> Vector:
    """
    Extract grade-k component from multivector: <M>_k

    Returns grade-k vector if present, otherwise zero vector.
    """
    from morphis.elements.vector import Vector

    component = M.grade_select(k)

    if component is not None:
        return component

    # Return zero vector of appropriate grade
    d = M.dim
    collection = M.collection
    shape = collection + (d,) * k if k > 0 else collection

    return Vector(data=zeros(shape), grade=k, metric=M.metric, collection=collection)


# =============================================================================
# Component Products
# =============================================================================


def scalar_product(u: Vector, v: Vector) -> NDArray:
    """
    Scalar part of geometric product: <uv>_0

    Returns scalar array with shape collection_shape.
    """
    M = geometric(u, v)
    s = M.grade_select(0)

    if s is not None:
        return s.data

    # No scalar component
    return zeros(broadcast_collection_shape(u, v))


def commutator(u: Vector, v: Vector) -> MultiVector:
    """
    Commutator product: [u, v] = (1 / 2) (uv - vu)

    Extracts antisymmetric part (odd grade differences).
    """
    uv = geometric(u, v)
    vu = geometric(v, u)

    return 0.5 * (uv - vu)


def anticommutator(u: Vector, v: Vector) -> MultiVector:
    """
    Anticommutator product: u * v = (1 / 2) (uv + vu)

    Extracts symmetric part (even grade differences).
    """
    uv = geometric(u, v)
    vu = geometric(v, u)

    return 0.5 * (uv + vu)


# =============================================================================
# Reversion
# =============================================================================


def _reverse_v(u: Vector) -> Vector:
    """
    Reverse a vector: reverse(u) = (-1)^(k (k - 1) / 2) u for grade-k vector.

    Reverses the order of grade-1 factors in the vector.
    """
    from morphis.elements.vector import Vector

    k = u.grade
    sign = (-1) ** (k * (k - 1) // 2)

    return Vector(
        data=sign * u.data,
        grade=k,
        metric=u.metric,
        collection=u.collection,
    )


def _reverse_mv(M: MultiVector) -> MultiVector:
    """
    Reverse each component of a multivector.

    Returns MultiVector with all components reversed.
    """
    from morphis.elements.multivector import MultiVector

    components = {k: _reverse_v(component) for k, component in M.data.items()}

    return MultiVector(data=components, metric=M.metric)


def reverse(u: Vector | MultiVector) -> Vector | MultiVector:
    """
    Reverse: reverse(u) = (-1)^(k (k - 1) / 2) u for grade-k vector.

    For multivectors, reverses each component.

    Reverses the order of grade-1 factors.
    """
    from morphis.elements.vector import Vector

    if isinstance(u, Vector):
        return _reverse_v(u)

    return _reverse_mv(u)


# =============================================================================
# Inverse
# =============================================================================


def _inverse_v(u: Vector) -> Vector:
    """
    Inverse of a vector: u^(-1) = reverse(u) / (u * reverse(u))

    Requires u * reverse(u) to be nonzero scalar.
    """
    from morphis.elements.vector import Vector

    u_rev = _reverse_v(u)
    u_u_rev = _geometric_v_v(u, u_rev)

    # Extract scalar part
    s = u_u_rev.grade_select(0)
    if s is None:
        raise ValueError("Vector square is not scalar - cannot invert")

    # Divide reversed vector by scalar
    s_expanded = s.data
    for _ in range(u.grade):
        s_expanded = s_expanded[..., newaxis]

    return Vector(
        data=u_rev.data / s_expanded,
        grade=u.grade,
        metric=u.metric,
        collection=u.collection,
    )


def _inverse_mv(M: MultiVector) -> MultiVector:
    """
    Inverse of a multivector: M^(-1) = reverse(M) / (M * reverse(M))

    Requires M * reverse(M) to be invertible scalar.
    """
    M_rev = _reverse_mv(M)
    M_M_rev = _geometric_mv_mv(M, M_rev)

    # Extract scalar part
    s = M_M_rev.grade_select(0)
    if s is None:
        raise ValueError("MultiVector product with reverse is not scalar - cannot invert")

    return M_rev * (1.0 / s.data)


def inverse(u: Vector | MultiVector) -> Vector | MultiVector:
    """
    Inverse: u^(-1) = reverse(u) / (u * reverse(u))

    Requires u * reverse(u) to be nonzero scalar.
    """
    from morphis.elements.vector import Vector

    if isinstance(u, Vector):
        return _inverse_v(u)

    return _inverse_mv(u)


# =============================================================================
# Geometric Product with Mixed Types (for operators)
# =============================================================================


def _geometric_v_mv(u: Vector, M: MultiVector) -> MultiVector:
    """
    Geometric product of vector with multivector: u * M

    Distributes over components.

    Returns MultiVector.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(u.metric, M.metric)

    result_components: dict[int, Vector] = {}

    for _k, component in M.data.items():
        product = _geometric_v_v(u, component)

        for grade, vec in product.data.items():
            if grade in result_components:
                result_components[grade] = result_components[grade] + vec
            else:
                result_components[grade] = vec

    return MultiVector(data=result_components, metric=metric)


def _geometric_mv_v(M: MultiVector, u: Vector) -> MultiVector:
    """
    Geometric product of multivector with vector: M * u

    Distributes over components.

    Returns MultiVector.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(M.metric, u.metric)

    result_components: dict[int, Vector] = {}

    for _k, component in M.data.items():
        product = _geometric_v_v(component, u)

        for grade, vec in product.data.items():
            if grade in result_components:
                result_components[grade] = result_components[grade] + vec
            else:
                result_components[grade] = vec

    return MultiVector(data=result_components, metric=metric)
