"""
Geometric Algebra - Example Script

Demonstrates the use of the geometric algebra module, showing how operations
work on single blades, arrays of blades, and combinations thereof. This script
serves as both documentation and verification that the operations broadcast
correctly across collection dimensions.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
"""

from numpy import array

from morphis.geometry.algebra import (
    dot,
    hodge_dual,
    interior,
    left_complement,
    meet,
    norm,
    normalize,
    project,
    reject,
    right_complement,
    wedge,
)
from morphis.geometry.algebra.projective import (
    are_collinear,
    bulk,
    direction,
    distance_point_to_point,
    euclidean,
    line,
    plane,
    point,
    point_on_line,
    weight,
)
from morphis.geometry.model import bivector_blade, scalar_blade, vector_blade
from morphis.geometry.model.metric import euclidean as euclidean_metric, pga
from morphis.utils.pretty import section, show_array, show_blade, subsection


# =============================================================================
# Section 1: Basic Blade Creation
# =============================================================================


def demo_blade_creation() -> None:
    """Demonstrate creating scalars, vectors, and bivectors."""
    section("1. BASIC BLADE CREATION")

    g = euclidean_metric(3)
    print(f"Euclidean metric (3D): signature={g.signature}")

    subsection("Scalar (Grade 0)")
    s = scalar_blade(2.5, g)
    show_blade("s", s)

    subsection("Vector (Grade 1)")
    v = vector_blade(array([1.0, 2.0, 3.0]), g)
    show_blade("v", v)

    subsection("Bivector (Grade 2)")
    b = bivector_blade(
        array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        g,
    )
    show_blade("B (e1 ^ e2 plane)", b)

    subsection("Scalar Arithmetic")
    v2 = 2.0 * v
    show_blade("2 * v", v2)
    v_neg = -v
    show_blade("-v", v_neg)


# =============================================================================
# Section 2: Blade Arithmetic (Same Grade)
# =============================================================================


def demo_blade_arithmetic() -> None:
    """Demonstrate addition and subtraction of blades."""
    section("2. BLADE ARITHMETIC (SAME GRADE)")

    g = euclidean_metric(3)

    u = vector_blade(array([1.0, 0.0, 0.0]), g)
    v = vector_blade(array([0.0, 1.0, 0.0]), g)

    subsection("Vector Addition")
    show_blade("u", u)
    show_blade("v", v)
    w = u + v
    show_blade("u + v", w)

    subsection("Vector Subtraction")
    d = u - v
    show_blade("u - v", d)


# =============================================================================
# Section 3: Single Blade vs Single Blade Operations
# =============================================================================


def demo_single_vs_single() -> None:
    """Demonstrate operations between two single blades."""
    section("3. SINGLE BLADE vs SINGLE BLADE")

    g = euclidean_metric(3)

    u = vector_blade(array([1.0, 0.0, 0.0]), g)
    v = vector_blade(array([0.0, 1.0, 0.0]), g)
    w = vector_blade(array([0.0, 0.0, 1.0]), g)

    subsection("Wedge Product: u ^ v (operator)")
    uv = u ^ v
    show_blade("u ^ v", uv)

    subsection("Chained Wedge: u ^ v ^ w (operator)")
    uvw_op = u ^ v ^ w
    show_blade("u ^ v ^ w", uvw_op)
    print("  Chained operators evaluate left-to-right: (u ^ v) ^ w")

    subsection("Variadic Wedge: wedge(u, v, w) (optimized)")
    uvw = wedge(u, v, w)
    show_blade("wedge(u, v, w)", uvw)
    print("  Single einsum with ε-symbol: optimal for large collections!")

    subsection("Dot Product")
    p = vector_blade(array([1.0, 2.0, 3.0]), g)
    q = vector_blade(array([4.0, 5.0, 6.0]), g)
    d = dot(p, q)
    show_array("p · q", d)
    print(f"  expected: 1*4 + 2*5 + 3*6 = {1 * 4 + 2 * 5 + 3 * 6}")

    subsection("Norm")
    n = norm(p)
    show_array("|p|", n)
    print(f"  expected: sqrt(1 + 4 + 9) = {(1 + 4 + 9) ** 0.5:.6f}")

    subsection("Interior Product: u ⌋ B")
    u_int_b = interior(u, uv)
    show_blade("u ⌋ (u ∧ v)", u_int_b)
    print("  (contracts u with u in the bivector, leaving v)")


# =============================================================================
# Section 4: Single Blade vs Array of Blades
# =============================================================================


def demo_single_vs_array() -> None:
    """Demonstrate broadcasting: single blade against an array of blades."""
    section("4. SINGLE BLADE vs ARRAY OF BLADES")

    g = euclidean_metric(3)

    subsection("Create single vector and array of vectors")
    single = vector_blade(array([1.0, 0.0, 0.0]), g)
    show_blade("single (e1)", single)

    many_data = array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    many = vector_blade(many_data, g, collection=(4,))
    show_blade("many (4 vectors)", many)

    subsection("Dot product: single . many (broadcasts)")
    dots = dot(single, many)
    show_array("e1 . [e1, e2, e3, e1+e2]", dots)
    print("  expected: [1, 0, 0, 1]")

    subsection("Wedge product: single ^ many (operator)")
    wedges = single ^ many
    show_blade("single ^ many", wedges)
    print("  Note: e1 ^ e1 = 0, giving zero in first position")


# =============================================================================
# Section 5: Array of Blades vs Array of Blades
# =============================================================================


def demo_array_vs_array() -> None:
    """Demonstrate element-wise operations on aligned arrays."""
    section("5. ARRAY OF BLADES vs ARRAY OF BLADES")

    g = euclidean_metric(3)

    subsection("Two arrays of vectors (same shape)")
    u_data = array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    v_data = array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ])
    u = vector_blade(u_data, g, collection=(3,))
    v = vector_blade(v_data, g, collection=(3,))
    show_blade("u (3 vectors)", u)
    show_blade("v (3 vectors)", v)

    subsection("Element-wise dot product")
    dots = dot(u, v)
    show_array("u[k] . v[k]", dots)
    print("  expected: [0, 0, 0] (all orthogonal pairs)")

    subsection("Element-wise wedge product (operator)")
    wedges = u ^ v
    show_blade("u ^ v", wedges)

    subsection("Element-wise norms")
    norms = norm(u)
    show_array("|u[k]|", norms)


# =============================================================================
# Section 6: Array Operations (Normalization, Projection)
# =============================================================================


def demo_array_operations() -> None:
    """Demonstrate operations that work across array elements."""
    section("6. OPERATIONS ON ARRAYS OF BLADES")

    g = euclidean_metric(3)

    subsection("Normalize array of vectors")
    v_data = array([
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    v = vector_blade(v_data, g, collection=(3,))
    show_blade("v (unnormalized)", v)

    v_norm = normalize(v)
    show_blade("normalize(v)", v_norm)
    show_array("check |v|", norm(v_norm))

    subsection("Project vectors onto a direction")
    axis = vector_blade(array([1.0, 1.0, 0.0]), g)
    projections = project(v, axis)
    show_blade("project(v, [1,1,0])", projections)

    subsection("Reject vectors from a direction")
    rejections = reject(v, axis)
    show_blade("reject(v, [1,1,0])", rejections)


# =============================================================================
# Section 7: Complements and Duality
# =============================================================================


def demo_complements() -> None:
    """Demonstrate complement and Hodge dual operations."""
    section("7. COMPLEMENTS AND DUALITY")

    g = euclidean_metric(3)

    u = vector_blade(array([1.0, 0.0, 0.0]), g)
    v = vector_blade(array([0.0, 1.0, 0.0]), g)
    B = u ^ v

    subsection("Right complement of a vector")
    u_comp = right_complement(u)
    show_blade("ū (right complement)", u_comp)
    print("  Maps grade-1 to grade-2 in 3D")

    subsection("Left complement of a vector")
    u_left = left_complement(u)
    show_blade("_u (left complement)", u_left)

    subsection("Hodge dual of a bivector")
    dual_B = hodge_dual(B)
    show_blade("⋆(u ∧ v)", dual_B)
    print("  In 3D Euclidean, ⋆(u ∧ v) ~ w")


# =============================================================================
# Section 8: Meet Operation
# =============================================================================


def demo_meet() -> None:
    """Demonstrate the meet (intersection) of subspaces."""
    section("8. MEET (INTERSECTION)")

    g = euclidean_metric(3)

    u = vector_blade(array([1.0, 0.0, 0.0]), g)
    v = vector_blade(array([0.0, 1.0, 0.0]), g)
    w = vector_blade(array([0.0, 0.0, 1.0]), g)

    # Using ^ operator for clean plane definitions
    A = u ^ v  # xy-plane
    B = u ^ w  # xz-plane

    subsection("Intersection of two planes")
    show_blade("A = u ^ v (xy-plane)", A)
    show_blade("B = u ^ w (xz-plane)", B)

    intersection = meet(A, B)
    show_blade("meet(A, B)", intersection)
    print("  The x-axis is where xy-plane meets xz-plane")


# =============================================================================
# Section 9: Projective Geometric Algebra
# =============================================================================


def demo_projective() -> None:
    """Demonstrate PGA-specific operations."""
    section("9. PROJECTIVE GEOMETRIC ALGEBRA (PGA)")

    g = pga(3)
    print(f"PGA metric (3D): signature={g.signature}")
    print("  diag(0, 1, 1, 1) - degenerate in e0")

    subsection("Embed Euclidean points")
    p1 = point(array([0.0, 0.0, 0.0]))
    p2 = point(array([1.0, 0.0, 0.0]))
    p3 = point(array([0.0, 1.0, 0.0]))
    show_blade("origin", p1)
    show_blade("(1,0,0)", p2)

    subsection("Point decomposition")
    show_array("weight(p2)", weight(p2))
    show_array("bulk(p2)", bulk(p2))
    show_array("euclidean(p2)", euclidean(p2))

    subsection("Embed a direction (point at infinity)")
    d = direction(array([1.0, 1.0, 0.0]))
    show_blade("direction [1,1,0]", d)
    show_array("weight (should be 0)", weight(d))

    subsection("Line through two points")
    l = line(p1, p2)
    show_blade("line(origin, (1,0,0))", l)

    subsection("Plane through three points")
    h = plane(p1, p2, p3)
    show_blade("plane(origin, x, y)", h)

    subsection("Distance between points")
    dist = distance_point_to_point(p1, p2)
    show_array("dist(origin, (1,0,0))", dist)

    subsection("Collinearity test")
    p4 = point(array([2.0, 0.0, 0.0]))
    are_coll = are_collinear(p1, p2, p4)
    show_array("collinear(origin, (1,0,0), (2,0,0))", are_coll)


# =============================================================================
# Section 10: PGA with Arrays
# =============================================================================


def demo_projective_arrays() -> None:
    """Demonstrate PGA operations on arrays of points."""
    section("10. PGA WITH ARRAYS OF POINTS")

    subsection("Array of points")
    coords = array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    points = point(coords, collection=(4,))
    show_blade("4 points", points)

    subsection("Single point to array distances")
    origin = point(array([0.0, 0.0, 0.0]))
    dists = distance_point_to_point(origin, points)
    show_array("dist(origin, points[k])", dists)
    print("  expected: [0, 1, 1, sqrt(2)]")

    subsection("Lines from origin to each point")
    lines = line(origin, points)
    show_blade("line(origin, points[k])", lines)

    subsection("Point on line test (array)")
    test_point = point(array([0.5, 0.0, 0.0]))
    on_line = point_on_line(test_point, lines)
    show_array("(0.5,0,0) on lines?", on_line)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all demonstrations."""
    print()
    print("=" * 70)
    print("  GEOMETRIC ALGEBRA DEMONSTRATION")
    print("  Showcasing blade operations and broadcasting behavior")
    print("=" * 70)

    demo_blade_creation()
    demo_blade_arithmetic()
    demo_single_vs_single()
    demo_single_vs_array()
    demo_array_vs_array()
    demo_array_operations()
    demo_complements()
    demo_meet()
    demo_projective()
    demo_projective_arrays()

    section("DEMONSTRATION COMPLETE")
    print()


if __name__ == "__main__":
    main()
