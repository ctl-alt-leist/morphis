"""
Exterior Algebra Operations

Demonstrates exterior algebra operations on blades:
- Wedge product (exterior product)
- Dot product and contractions
- Duality (Hodge dual, complements)
- Projections and rejections
- Meet and join (subspace intersection/union)

Run: uv run python -m morphis.examples.exterior
"""

from morphis.elements import Vector, metric
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
from morphis.utils.pretty import section, subsection


# =============================================================================
# Section 1: Basic Vector Creation
# =============================================================================


def demo_vector_creation() -> None:
    """Demonstrate creating scalars, vectors, and bivectors."""
    section("1. BASIC BLADE CREATION")

    g = metric(3)
    print(f"Euclidean metric (3D): signature={g.signature}")

    subsection("Scalar (Grade 0)")
    s = Vector(2.5, grade=0, metric=g)
    print("s:")
    print(s)

    subsection("Vector (Grade 1)")
    v = Vector([1.0, 2.0, 3.0], grade=1, metric=g)
    print("v:")
    print(v)

    subsection("Bivector (Grade 2)")
    b = Vector(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        grade=2,
        metric=g,
    )
    print("b (e1 ^ e2 plane):")
    print(b)

    subsection("Scalar Arithmetic")
    print("2 * v:")
    print(2.0 * v)
    print()
    print("-v:")
    print(-v)


# =============================================================================
# Section 2: Vector Arithmetic (Same Grade)
# =============================================================================


def demo_vector_arithmetic() -> None:
    """Demonstrate addition and subtraction of blades."""
    section("2. BLADE ARITHMETIC (SAME GRADE)")

    g = metric(3)

    u = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
    v = Vector([0.0, 1.0, 0.0], grade=1, metric=g)

    subsection("Vector Addition")
    print("u:")
    print(u)
    print()
    print("v:")
    print(v)
    print()
    print("u + v:")
    print(u + v)

    subsection("Vector Subtraction")
    print("u - v:")
    print(u - v)


# =============================================================================
# Section 3: Single Vector vs Single Vector Operations
# =============================================================================


def demo_single_vs_single() -> None:
    """Demonstrate operations between two single blades."""
    section("3. SINGLE BLADE vs SINGLE BLADE")

    g = metric(3)

    u = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
    v = Vector([0.0, 1.0, 0.0], grade=1, metric=g)
    w = Vector([0.0, 0.0, 1.0], grade=1, metric=g)

    subsection("Wedge Product: u ^ v (operator)")
    uv = u ^ v
    print("u ^ v:")
    print(uv)

    subsection("Chained Wedge: u ^ v ^ w (operator)")
    print("u ^ v ^ w:")
    print(u ^ v ^ w)
    print("  Chained operators evaluate left-to-right: (u ^ v) ^ w")

    subsection("Variadic Wedge: wedge(u, v, w) (optimized)")
    print("wedge(u, v, w):")
    print(wedge(u, v, w))
    print("  Single einsum with ε-symbol: optimal for large collections!")

    subsection("Dot Product")
    p = Vector([1.0, 2.0, 3.0], grade=1, metric=g)
    q = Vector([4.0, 5.0, 6.0], grade=1, metric=g)
    print(f"p · q = {dot(p, q):.4g}")
    print(f"  expected: 1*4 + 2*5 + 3*6 = {1 * 4 + 2 * 5 + 3 * 6}")

    subsection("Norm")
    print(f"|p| = {norm(p):.6f}")
    print(f"  expected: sqrt(1 + 4 + 9) = {(1 + 4 + 9) ** 0.5:.6f}")

    subsection("Interior Product: u ⌋ B")
    print("u ⌋ (u ∧ v):")
    print(interior(u, uv))
    print("  (contracts u with u in the bivector, leaving v)")


# =============================================================================
# Section 4: Single Vector vs Array of Vectors
# =============================================================================


def demo_single_vs_array() -> None:
    """Demonstrate broadcasting: single blade against an array of blades."""
    section("4. SINGLE BLADE vs ARRAY OF BLADES")

    g = metric(3)

    subsection("Create single vector and array of vectors")
    single = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
    print("single (e1):")
    print(single)

    many = Vector(
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
    print()
    print("many (4 vectors):")
    print(many)

    subsection("Dot product: single . many (broadcasts)")
    print(f"e1 . [e1, e2, e3, e1 + e2] = {dot(single, many)}")
    print("  expected: [1, 0, 0, 1]")

    subsection("Wedge product: single ^ many (operator)")
    print("single ^ many:")
    print(single ^ many)
    print("  Note: e1 ^ e1 = 0, giving zero in first position")


# =============================================================================
# Section 5: Array of Vectors vs Array of Vectors
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
    u = Vector(u_data, grade=1, metric=g, collection=(3,))
    v = Vector(v_data, grade=1, metric=g, collection=(3,))
    print("u (3 vectors):")
    print(u)
    print()
    print("v (3 vectors):")
    print(v)

    subsection("Element-wise dot product")
    print(f"u[k] . v[k] = {dot(u, v)}")
    print("  expected: [0, 0, 0] (all orthogonal pairs)")

    subsection("Element-wise wedge product (operator)")
    print("u ^ v:")
    print(u ^ v)

    subsection("Element-wise norms")
    print(f"|u[k]| = {norm(u)}")


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
    v = Vector(v_data, grade=1, metric=g, collection=(3,))
    print("v (unnormalized):")
    print(v)

    v_norm = normalize(v)
    print()
    print("normalize(v):")
    print(v_norm)
    print(f"check |v| = {norm(v_norm)}")

    subsection("Project vectors onto a direction")
    axis = Vector([1.0, 1.0, 0.0], grade=1, metric=g)
    print("project(v, [1,1,0]):")
    print(project(v, axis))

    subsection("Reject vectors from a direction")
    print("reject(v, [1,1,0]):")
    print(reject(v, axis))


# =============================================================================
# Section 7: Complements and Duality
# =============================================================================


def demo_complements() -> None:
    """Demonstrate complement and Hodge dual operations."""
    section("7. COMPLEMENTS AND DUALITY")

    g = metric(3)

    u = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
    v = Vector([0.0, 1.0, 0.0], grade=1, metric=g)
    b = u ^ v

    subsection("Right complement of a vector")
    print("ū (right complement):")
    print(right_complement(u))
    print("  Maps grade-1 to grade-2 in 3D")

    subsection("Left complement of a vector")
    print("_u (left complement):")
    print(left_complement(u))

    subsection("Hodge dual of a bivector")
    print("⋆(u ∧ v):")
    print(hodge_dual(b))
    print("  In 3D Euclidean, ⋆(u ∧ v) ~ w")


# =============================================================================
# Section 8: Meet Operation
# =============================================================================


def demo_meet() -> None:
    """Demonstrate the meet (intersection) of subspaces."""
    section("8. MEET (INTERSECTION)")

    g = metric(3)

    u = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
    v = Vector([0.0, 1.0, 0.0], grade=1, metric=g)
    w = Vector([0.0, 0.0, 1.0], grade=1, metric=g)

    # Using ^ operator for clean plane definitions
    A = u ^ v  # xy-plane
    B = u ^ w  # xz-plane

    subsection("Intersection of two planes")
    print("A = u ^ v (xy-plane):")
    print(A)
    print()
    print("B = u ^ w (xz-plane):")
    print(B)
    print()
    print("meet(A, B):")
    print(meet(A, B))
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
    print("origin:")
    print(p1)
    print()
    print("(1,0,0):")
    print(p2)

    subsection("Point decomposition")
    print(f"weight(p2) = {weight(p2):.4g}")
    print(f"bulk(p2) = {bulk(p2)}")
    print(f"euclidean(p2) = {euclidean(p2)}")

    subsection("Embed a direction (point at infinity)")
    d = direction([1.0, 1.0, 0.0])
    print("direction [1,1,0]:")
    print(d)
    print(f"weight (should be 0) = {weight(d):.4g}")

    subsection("Line through two points")
    print("line(origin, (1,0,0)):")
    print(line(p1, p2))

    subsection("Plane through three points")
    print("plane(origin, x, y):")
    print(plane(p1, p2, p3))

    subsection("Distance between points")
    print(f"dist(origin, (1,0,0)) = {distance_point_to_point(p1, p2):.4g}")

    subsection("Collinearity test")
    p4 = point([2.0, 0.0, 0.0])
    print(f"collinear(origin, (1,0,0), (2,0,0)) = {are_collinear(p1, p2, p4)}")


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
    print("4 points:")
    print(points)

    subsection("Single point to array distances")
    origin = point([0.0, 0.0, 0.0])
    print(f"dist(origin, points[k]) = {distance_point_to_point(origin, points)}")
    print("  expected: [0, 1, 1, sqrt(2)]")

    subsection("Lines from origin to each point")
    lines = line(origin, points)
    print("line(origin, points[k]):")
    print(lines)

    subsection("Point on line test (array)")
    test_point = point([0.5, 0.0, 0.0])
    print(f"(0.5,0,0) on lines? = {point_on_line(test_point, lines)}")


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

    demo_vector_creation()
    demo_vector_arithmetic()
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
