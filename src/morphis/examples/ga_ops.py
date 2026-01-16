"""
Geometric Algebra - Example Script

Demonstrates the use of the geometric algebra module, showing how operations
work on single blades, arrays of blades, and combinations thereof. This script
serves as both documentation and verification that the operations broadcast
correctly across collection dimensions.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
"""

from morphis.elements import Blade, metric
from morphis.operations import (
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
from morphis.transforms import (
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
from morphis.utils.pretty import section, show_array, show_blade, subsection


# =============================================================================
# Section 1: Basic Blade Creation
# =============================================================================


def demo_blade_creation() -> None:
    """Demonstrate creating scalars, vectors, and bivectors."""
    section("1. BASIC BLADE CREATION")

    g = metric(3)
    print(f"Euclidean metric (3D): signature={g.signature}")

    subsection("Scalar (Grade 0)")
    s = Blade(2.5, grade=0, metric=g)
    show_blade("s", s)

    subsection("Vector (Grade 1)")
    v = Blade([1.0, 2.0, 3.0], grade=1, metric=g)
    show_blade("v", v)

    subsection("Bivector (Grade 2)")
    b = Blade(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        grade=2,
        metric=g,
    )
    show_blade("B (e1 ^ e2 plane)", b)

    subsection("Scalar Arithmetic")
    show_blade("2 * v", 2.0 * v)
    show_blade("-v", -v)


# =============================================================================
# Section 2: Blade Arithmetic (Same Grade)
# =============================================================================


def demo_blade_arithmetic() -> None:
    """Demonstrate addition and subtraction of blades."""
    section("2. BLADE ARITHMETIC (SAME GRADE)")

    g = metric(3)

    u = Blade([1.0, 0.0, 0.0], grade=1, metric=g)
    v = Blade([0.0, 1.0, 0.0], grade=1, metric=g)

    subsection("Vector Addition")
    show_blade("u", u)
    show_blade("v", v)
    show_blade("u + v", u + v)

    subsection("Vector Subtraction")
    show_blade("u - v", u - v)


# =============================================================================
# Section 3: Single Blade vs Single Blade Operations
# =============================================================================


def demo_single_vs_single() -> None:
    """Demonstrate operations between two single blades."""
    section("3. SINGLE BLADE vs SINGLE BLADE")

    g = metric(3)

    u = Blade([1.0, 0.0, 0.0], grade=1, metric=g)
    v = Blade([0.0, 1.0, 0.0], grade=1, metric=g)
    w = Blade([0.0, 0.0, 1.0], grade=1, metric=g)

    subsection("Wedge Product: u ^ v (operator)")
    uv = u ^ v
    show_blade("u ^ v", uv)

    subsection("Chained Wedge: u ^ v ^ w (operator)")
    show_blade("u ^ v ^ w", u ^ v ^ w)
    print("  Chained operators evaluate left-to-right: (u ^ v) ^ w")

    subsection("Variadic Wedge: wedge(u, v, w) (optimized)")
    show_blade("wedge(u, v, w)", wedge(u, v, w))
    print("  Single einsum with ε-symbol: optimal for large collections!")

    subsection("Dot Product")
    p = Blade([1.0, 2.0, 3.0], grade=1, metric=g)
    q = Blade([4.0, 5.0, 6.0], grade=1, metric=g)
    show_array("p · q", dot(p, q))
    print(f"  expected: 1*4 + 2*5 + 3*6 = {1 * 4 + 2 * 5 + 3 * 6}")

    subsection("Norm")
    show_array("|p|", norm(p))
    print(f"  expected: sqrt(1 + 4 + 9) = {(1 + 4 + 9) ** 0.5:.6f}")

    subsection("Interior Product: u ⌋ B")
    show_blade("u ⌋ (u ∧ v)", interior(u, uv))
    print("  (contracts u with u in the bivector, leaving v)")


# =============================================================================
# Section 4: Single Blade vs Array of Blades
# =============================================================================


def demo_single_vs_array() -> None:
    """Demonstrate broadcasting: single blade against an array of blades."""
    section("4. SINGLE BLADE vs ARRAY OF BLADES")

    g = metric(3)

    subsection("Create single vector and array of vectors")
    single = Blade([1.0, 0.0, 0.0], grade=1, metric=g)
    show_blade("single (e1)", single)

    many = Blade(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        grade=1,
        metric=g,
        collection=(4,),
    )
    show_blade("many (4 vectors)", many)

    subsection("Dot product: single . many (broadcasts)")
    show_array("e1 . [e1, e2, e3, e1 + e2]", dot(single, many))
    print("  expected: [1, 0, 0, 1]")

    subsection("Wedge product: single ^ many (operator)")
    show_blade("single ^ many", single ^ many)
    print("  Note: e1 ^ e1 = 0, giving zero in first position")


# =============================================================================
# Section 5: Array of Blades vs Array of Blades
# =============================================================================


def demo_array_vs_array() -> None:
    """Demonstrate element-wise operations on aligned arrays."""
    section("5. ARRAY OF BLADES vs ARRAY OF BLADES")

    g = metric(3)

    subsection("Two arrays of vectors (same shape)")
    u_data = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    v_data = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ]
    u = Blade(u_data, grade=1, metric=g, collection=(3,))
    v = Blade(v_data, grade=1, metric=g, collection=(3,))
    show_blade("u (3 vectors)", u)
    show_blade("v (3 vectors)", v)

    subsection("Element-wise dot product")
    show_array("u[k] . v[k]", dot(u, v))
    print("  expected: [0, 0, 0] (all orthogonal pairs)")

    subsection("Element-wise wedge product (operator)")
    show_blade("u ^ v", u ^ v)

    subsection("Element-wise norms")
    show_array("|u[k]|", norm(u))


# =============================================================================
# Section 6: Array Operations (Normalization, Projection)
# =============================================================================


def demo_array_operations() -> None:
    """Demonstrate operations that work across array elements."""
    section("6. OPERATIONS ON ARRAYS OF BLADES")

    g = metric(3)

    subsection("Normalize array of vectors")
    v_data = [
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    v = Blade(v_data, grade=1, metric=g, collection=(3,))
    show_blade("v (unnormalized)", v)

    v_norm = normalize(v)
    show_blade("normalize(v)", v_norm)
    show_array("check |v|", norm(v_norm))

    subsection("Project vectors onto a direction")
    axis = Blade([1.0, 1.0, 0.0], grade=1, metric=g)
    show_blade("project(v, [1,1,0])", project(v, axis))

    subsection("Reject vectors from a direction")
    show_blade("reject(v, [1,1,0])", reject(v, axis))


# =============================================================================
# Section 7: Complements and Duality
# =============================================================================


def demo_complements() -> None:
    """Demonstrate complement and Hodge dual operations."""
    section("7. COMPLEMENTS AND DUALITY")

    g = metric(3)

    u = Blade([1.0, 0.0, 0.0], grade=1, metric=g)
    v = Blade([0.0, 1.0, 0.0], grade=1, metric=g)
    B = u ^ v

    subsection("Right complement of a vector")
    show_blade("ū (right complement)", right_complement(u))
    print("  Maps grade-1 to grade-2 in 3D")

    subsection("Left complement of a vector")
    show_blade("_u (left complement)", left_complement(u))

    subsection("Hodge dual of a bivector")
    show_blade("⋆(u ∧ v)", hodge_dual(B))
    print("  In 3D Euclidean, ⋆(u ∧ v) ~ w")


# =============================================================================
# Section 8: Meet Operation
# =============================================================================


def demo_meet() -> None:
    """Demonstrate the meet (intersection) of subspaces."""
    section("8. MEET (INTERSECTION)")

    g = metric(3)

    u = Blade([1.0, 0.0, 0.0], grade=1, metric=g)
    v = Blade([0.0, 1.0, 0.0], grade=1, metric=g)
    w = Blade([0.0, 0.0, 1.0], grade=1, metric=g)

    # Using ^ operator for clean plane definitions
    A = u ^ v  # xy-plane
    B = u ^ w  # xz-plane

    subsection("Intersection of two planes")
    show_blade("A = u ^ v (xy-plane)", A)
    show_blade("B = u ^ w (xz-plane)", B)

    show_blade("meet(A, B)", meet(A, B))
    print("  The x-axis is where xy-plane meets xz-plane")


# =============================================================================
# Section 9: Projective Geometric Algebra
# =============================================================================


def demo_projective() -> None:
    """Demonstrate PGA-specific operations."""
    section("9. PROJECTIVE GEOMETRIC ALGEBRA (PGA)")

    # Can use "pga" as alias for "projective" structure
    g = metric(3, "euclidean", "pga")
    print(f"PGA metric (3D): signature={g.signature}")
    print("  diag(0, 1, 1, 1) - degenerate in e0")

    subsection("Embed Euclidean points")
    p1 = point([0.0, 0.0, 0.0])
    p2 = point([1.0, 0.0, 0.0])
    p3 = point([0.0, 1.0, 0.0])
    show_blade("origin", p1)
    show_blade("(1,0,0)", p2)

    subsection("Point decomposition")
    show_array("weight(p2)", weight(p2))
    show_array("bulk(p2)", bulk(p2))
    show_array("euclidean(p2)", euclidean(p2))

    subsection("Embed a direction (point at infinity)")
    d = direction([1.0, 1.0, 0.0])
    show_blade("direction [1,1,0]", d)
    show_array("weight (should be 0)", weight(d))

    subsection("Line through two points")
    show_blade("line(origin, (1,0,0))", line(p1, p2))

    subsection("Plane through three points")
    show_blade("plane(origin, x, y)", plane(p1, p2, p3))

    subsection("Distance between points")
    show_array("dist(origin, (1,0,0))", distance_point_to_point(p1, p2))

    subsection("Collinearity test")
    p4 = point([2.0, 0.0, 0.0])
    show_array("collinear(origin, (1,0,0), (2,0,0))", are_collinear(p1, p2, p4))


# =============================================================================
# Section 10: PGA with Arrays
# =============================================================================


def demo_projective_arrays() -> None:
    """Demonstrate PGA operations on arrays of points."""
    section("10. PGA WITH ARRAYS OF POINTS")

    subsection("Array of points")
    coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    points = point(coords, collection=(4,))
    show_blade("4 points", points)

    subsection("Single point to array distances")
    origin = point([0.0, 0.0, 0.0])
    show_array("dist(origin, points[k])", distance_point_to_point(origin, points))
    print("  expected: [0, 1, 1, sqrt(2)]")

    subsection("Lines from origin to each point")
    lines = line(origin, points)
    show_blade("line(origin, points[k])", lines)

    subsection("Point on line test (array)")
    test_point = point([0.5, 0.0, 0.0])
    show_array("(0.5,0,0) on lines?", point_on_line(test_point, lines))


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
