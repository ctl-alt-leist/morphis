"""
Geometric Algebra - Projective Operations

PGA-specific operations: point and direction embedding, geometric constructors,
distances, incidence predicates, and translation. These operations are specific
to projective geometric algebra and use the degenerate metric diag(0, 1, 1, ..., 1).

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import (
    abs as np_abs,
    all as np_all,
    asarray,
    einsum,
    newaxis,
    ones,
    sqrt,
    where,
    zeros,
)
from numpy.typing import NDArray

from morphis.config import TOLERANCE
from morphis.elements.metric import Metric, pga_metric
from morphis.operations.norms import norm
from morphis.operations.products import geometric


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector


# =============================================================================
# Embedding
# =============================================================================


def point(x: NDArray, metric: Metric | None = None, collection: tuple[int, ...] | None = None) -> Vector:
    """
    Embed a Euclidean point into projective space. Points have unit weight
    (e_0 component = 1):

        p = e_0 + x^1 e_1 + ... + x^d e_d

    Args:
        x: Euclidean coordinates of shape (..., d)
        metric: PGA metric (inferred from x if not provided)
        collection: Shape of the collection dimensions (inferred from x if not provided)

    Returns:
        Grade-1 blade in (d + 1)-dimensional PGA.
    """
    from morphis.elements.vector import Vector

    x = asarray(x)
    d = x.shape[-1]
    dim = d + 1

    if metric is None:
        metric = pga_metric(d)
    elif metric.dim != dim:
        raise ValueError(f"Metric dim {metric.dim} doesn't match point dim+1 {dim}")

    shape = x.shape[:-1] + (dim,)
    p = zeros(shape, dtype=x.dtype)
    p[..., 0] = 1.0
    p[..., 1:] = x

    # Infer collection from x shape if not provided
    if collection is None:
        collection = x.shape[:-1]

    return Vector(data=p, grade=1, metric=metric, collection=collection)


def direction(v: NDArray, metric: Metric | None = None, collection: tuple[int, ...] | None = None) -> Vector:
    """
    Embed a Euclidean direction into projective space. Directions have zero
    weight (e_0 component = 0) and represent points at infinity.

    Args:
        v: Euclidean direction of shape (..., d)
        metric: PGA metric (inferred from v if not provided)
        collection: Shape of the collection dimensions (inferred from v if not provided)

    Returns:
        Grade-1 blade in (d + 1)-dimensional PGA.
    """
    from morphis.elements.vector import Vector

    v = asarray(v)
    d = v.shape[-1]
    dim = d + 1

    if metric is None:
        metric = pga_metric(d)
    elif metric.dim != dim:
        raise ValueError(f"Metric dim {metric.dim} doesn't match direction dim+1 {dim}")

    shape = v.shape[:-1] + (dim,)
    result = zeros(shape, dtype=v.dtype)
    result[..., 1:] = v

    # Infer collection from v shape if not provided
    if collection is None:
        collection = v.shape[:-1]

    return Vector(data=result, grade=1, metric=metric, collection=collection)


# =============================================================================
# Decomposition
# =============================================================================


def weight(p: Vector) -> NDArray:
    """
    Extract the weight (e_0 component) of a projective vector.

    Returns scalar array of weights.
    """
    if p.grade != 1:
        raise ValueError(f"weight() requires grade-1 blade, got grade {p.grade}")
    return p.data[..., 0]


def bulk(p: Vector) -> NDArray:
    """
    Extract the bulk (Euclidean components) of a projective vector.

    Returns array of Euclidean components with shape (*collection_shape, d).
    """
    if p.grade != 1:
        raise ValueError(f"bulk() requires grade-1 blade, got grade {p.grade}")
    return p.data[..., 1:]


def euclidean(p: Vector) -> NDArray:
    """
    Project a projective point to Euclidean coordinates by dividing bulk by
    weight. For directions (weight = 0), returns the bulk directly.

    Returns Euclidean coordinates with shape (*collection_shape, d).
    """
    w = weight(p)[..., newaxis]
    b = bulk(p)
    safe_w = where(np_abs(w) > TOLERANCE, w, 1.0)
    return where(np_abs(w) > TOLERANCE, b / safe_w, b)


# =============================================================================
# Predicates
# =============================================================================


def is_point(p: Vector) -> NDArray:
    """Check if a projective vector represents a point (nonzero weight)."""
    return np_abs(weight(p)) > TOLERANCE


def is_direction(p: Vector) -> NDArray:
    """Check if a projective vector represents a direction (zero weight)."""
    return np_abs(weight(p)) <= TOLERANCE


# =============================================================================
# Geometric Constructors
# =============================================================================


def line(p: Vector, q: Vector) -> Vector:
    """Construct a line through two points as the bivector p ^ q."""
    return p ^ q


def plane(p: Vector, q: Vector, r: Vector) -> Vector:
    """Construct a plane through three points as the trivector p ^ q ^ r."""
    return p ^ q ^ r


def plane_from_point_and_line(p: Vector, l: Vector) -> Vector:
    """Construct a plane through a point and a line as p ^ l."""
    return p ^ l


# =============================================================================
# Distances
# =============================================================================


def distance_point_to_point(p: Vector, q: Vector) -> NDArray:
    """Compute Euclidean distance between two points."""
    metric = Metric.merge(p.metric, q.metric)
    x_p = euclidean(p)
    x_q = euclidean(q)
    diff = x_q - x_p
    g_eucl = metric.data[1:, 1:]
    return sqrt(einsum("ab, ...a, ...b -> ...", g_eucl, diff, diff))


def distance_point_to_line(p: Vector, l: Vector) -> NDArray:
    """Compute distance from a point to a line as |p ^ l| / |l|."""
    Metric.merge(p.metric, l.metric)
    p_wedge_l = p ^ l
    numerator = norm(p_wedge_l)
    denominator = norm(l)
    return numerator / where(denominator > TOLERANCE, denominator, 1.0)


def distance_point_to_plane(p: Vector, h: Vector) -> NDArray:
    """Compute distance from a point to a plane as |p ^ h| / |h|."""
    Metric.merge(p.metric, h.metric)
    p_wedge_h = p ^ h
    numerator = norm(p_wedge_h)
    denominator = norm(h)
    return numerator / where(denominator > TOLERANCE, denominator, 1.0)


# =============================================================================
# Incidence Predicates
# =============================================================================


def are_collinear(p: Vector, q: Vector, r: Vector, tol: float | None = None) -> NDArray:
    """Check if three points are collinear: p ^ q ^ r = 0."""
    if tol is None:
        tol = TOLERANCE
    trivector = p ^ q ^ r
    return np_all(np_abs(trivector.data) < tol, axis=tuple(range(-3, 0)))


def are_coplanar(p: Vector, q: Vector, r: Vector, s: Vector, tol: float | None = None) -> NDArray:
    """Check if four points are coplanar: p ^ q ^ r ^ s = 0."""
    if tol is None:
        tol = TOLERANCE
    quadvector = p ^ q ^ r ^ s
    return np_all(np_abs(quadvector.data) < tol, axis=tuple(range(-4, 0)))


def point_on_line(p: Vector, l: Vector, tol: float | None = None) -> NDArray:
    """Check if a point lies on a line: p ^ l = 0."""
    if tol is None:
        tol = TOLERANCE
    joined = p ^ l
    return np_all(np_abs(joined.data) < tol, axis=tuple(range(-3, 0)))


def point_on_plane(p: Vector, h: Vector, tol: float | None = None) -> NDArray:
    """Check if a point lies on a plane: p ^ h = 0."""
    if tol is None:
        tol = TOLERANCE
    joined = p ^ h
    return np_all(np_abs(joined.data) < tol, axis=tuple(range(-4, 0)))


def line_in_plane(l: Vector, h: Vector, tol: float | None = None) -> NDArray:
    """Check if a line lies in a plane: l ^ h = 0."""
    if tol is None:
        tol = TOLERANCE
    joined = l ^ h
    return np_all(np_abs(joined.data) < tol, axis=tuple(range(-5, 0)))


# =============================================================================
# Translator Constructor
# =============================================================================


def translator(v: Vector) -> MultiVector:
    """
    Create a translator for pure translation (PGA only).

    M = 1 + (1/2) t^m e_{0m}

    The translator is a MultiVector with grades {0, 2} where the bivector
    part uses only degenerate (e_0) components. Apply via sandwich product:
        translated = M * p * ~M

    Args:
        v: Direction Vector (grade-1) representing the translation.
           Should be a PGA direction (zero weight) or will be treated as one.

    Returns:
        MultiVector with grades {0, 2} representing the translation.

    Example:
        # Translate by (1, 0, 0) in 3D PGA
        from morphis.transforms.projective import direction
        v = direction([1, 0, 0])
        M = translator(v)
        p_translated = M * p * ~M
    """
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector

    if v.grade != 1:
        raise ValueError(f"translator requires grade-1 vector, got grade {v.grade}")

    metric = v.metric
    dim = metric.dim
    collection = v.collection

    # Extract Euclidean components (bulk) from the direction vector
    displacement = bulk(v)

    # Scalar part: 1
    shape_0 = collection if collection else ()
    scalar_data = ones(shape_0)

    # Bivector part: (1/2) t^m e_{0m}
    shape_2 = collection + (dim, dim)
    bivector_data = zeros(shape_2)
    for m in range(1, dim):
        bivector_data[..., 0, m] = 0.5 * displacement[..., m - 1]

    components = {
        0: Vector(scalar_data, grade=0, metric=metric, collection=collection),
        2: Vector(data=bivector_data, grade=2, metric=metric, collection=collection),
    }

    return MultiVector(data=components, metric=metric, collection=collection)


# =============================================================================
# Screw Motion Constructor
# =============================================================================


def screw(
    B: Vector,
    angle: float | NDArray,
    translation: Vector,
    center: Vector | None = None,
) -> MultiVector:
    """
    Create a motor for screw motion (rotation + translation along axis).

    The screw motion combines rotation in the plane defined by B with
    translation. If center is provided, rotation is about that point.

    Args:
        B: Bivector defining the rotation plane.
        angle: Rotation angle in radians.
        translation: Direction Vector representing the translation.
        center: Optional point Vector for rotation center (default: origin).

    Returns:
        MultiVector (motor) representing the screw motion.

    Example:
        # Rotation in xy-plane + translation along z
        B = e1 ^ e2
        t = direction([0, 0, 1])
        M = screw(B, angle=pi/2, translation=t)
        p_transformed = M * p * ~M
    """
    from morphis.elements.multivector import MultiVector
    from morphis.transforms.rotations import rotor

    metric = B.metric

    # Create rotor and translator
    R = rotor(B, angle)
    T = translator(translation)

    # Compose: T * R (translate after rotate)
    if center is not None:
        # Create direction vectors for translating to/from center
        center_bulk = bulk(center)
        neg_center = direction(-center_bulk, metric=metric, collection=center.collection)
        pos_center = direction(center_bulk, metric=metric, collection=center.collection)

        T_to = translator(neg_center)
        T_back = translator(pos_center)

        # T_back * T * R * T_to
        temp1 = geometric(R, T_to)
        temp2 = geometric(T, temp1)
        result = geometric(T_back, temp2)
    else:
        result = geometric(T, R)

    # Project to motor grades {0, 2}
    motor_components = {k: v for k, v in result.data.items() if k in {0, 2}}

    return MultiVector(
        data=motor_components,
        metric=metric,
        collection=B.collection,
    )
