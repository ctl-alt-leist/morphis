"""
Clifford Algebra Operations

Demonstrates the Clifford (geometric) product and related transformations:
- Geometric product of vectors and blades
- Grade decomposition (scalar, bivector, etc.)
- Reversion and inversion
- Reflections and rotations

Run: uv run python -m morphis.examples.clifford
"""

from math import cos, pi, sin

from numpy import zeros

from morphis.elements import Metric, Vector, metric
from morphis.operations import anticommutator, commutator, geometric, grade_project, inverse, norm, reverse
from morphis.utils.pretty import section, subsection


# =============================================================================
# Helper: Basis vectors
# =============================================================================


def basis_vector(idx: int, metric: Metric) -> Vector:
    """Create basis vector e_idx for the given metric."""
    data = zeros(metric.dim)
    data[idx] = 1.0
    return Vector(data, grade=1, metric=metric)


# =============================================================================
# Section 1: Geometric Product Basics
# =============================================================================


def demo_geometric_basics() -> None:
    """Demonstrate the fundamental property: uv = u.v + u^v."""
    section("1. GEOMETRIC PRODUCT BASICS")

    g = metric(3)

    subsection("Geometric product decomposes into dot + wedge")
    u = Vector([1.0, 0.0, 0.0], grade=1, metric=g)  # e1
    v = Vector([0.0, 1.0, 0.0], grade=1, metric=g)  # e2

    print("For orthogonal vectors e1 and e2:")
    print("  e1 . e2 = 0 (orthogonal)")
    print("  e1 ^ e2 = e12 (bivector)")
    print()

    UV = geometric(u, v)
    print("e1 * e2:")
    print(UV)

    scalar_part = grade_project(UV, 0)
    print()
    print(f"  Scalar part (dot product): {scalar_part.data}")
    print("  Bivector part (wedge): nonzero in e12 plane")

    subsection("Parallel vectors: uv = u.v (pure scalar)")
    u = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
    v = Vector([3.0, 0.0, 0.0], grade=1, metric=g)  # v = 3u

    print("e1 * (3 e1):")
    print(geometric(u, v))
    print("  -> Pure scalar: 3 (no bivector part)")

    subsection("General case: u at angle to v")
    theta = pi / 4
    u = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
    v = Vector([cos(theta), sin(theta), 0.0], grade=1, metric=g)

    UV = geometric(u, v)
    print("e1 * v (at pi/4):")
    print(UV)

    scalar_part = grade_project(UV, 0)
    bivector_part = grade_project(UV, 2)
    bivector_norm = norm(bivector_part)
    print()
    print(f"  Scalar part = cos(pi/4) = {scalar_part.data:.6f}")
    print(f"  Bivector magnitude = sin(pi/4) = {bivector_norm:.6f}")


# =============================================================================
# Section 2: Vector Contraction
# =============================================================================


def demo_vector_contraction() -> None:
    """Demonstrate v^2 = |v|^2 for vectors."""
    section("2. VECTOR CONTRACTION (v^2 = |v|^2)")

    g = metric(3)

    subsection("Unit vector squares to 1")
    e1 = basis_vector(0, g)
    print("e1 * e1:")
    print(geometric(e1, e1))
    print("  -> Scalar 1 (unit vector in Euclidean space)")

    subsection("General vector: v^2 = |v|^2")
    v = Vector([3.0, 4.0, 0.0], grade=1, metric=g)
    V_sq = geometric(v, v)
    print("v * v where v = [3, 4, 0]:")
    print(V_sq)
    print(f"  -> Scalar {grade_project(V_sq, 0).data} = 3^2 + 4^2")


# =============================================================================
# Section 3: Anticommutativity
# =============================================================================


def demo_anticommutativity() -> None:
    """Demonstrate orthogonal vectors anticommute."""
    section("3. ANTICOMMUTATIVITY (orthogonal vectors)")

    g = metric(3)

    e1 = basis_vector(0, g)
    e2 = basis_vector(1, g)

    subsection("e1 * e2 vs e2 * e1")
    E1E2 = geometric(e1, e2)
    E2E1 = geometric(e2, e1)

    print("e1 * e2:")
    print(E1E2)
    print()
    print("e2 * e1:")
    print(E2E1)

    b1 = grade_project(E1E2, 2)
    b2 = grade_project(E2E1, 2)
    print()
    print(f"  Bivector of e1 * e2: {b1.data[0, 1]:.1f} in [0,1] slot")
    print(f"  Bivector of e2 * e1: {b2.data[0, 1]:.1f} in [0,1] slot")
    print("  -> Negatives! Orthogonal vectors anticommute.")


# =============================================================================
# Section 4: Reversion
# =============================================================================


def demo_reversion() -> None:
    """Demonstrate the reverse operation."""
    section("4. REVERSION")

    g = metric(3)

    print("Reverse sign pattern: (-1)^{k(k-1)/2}")
    print("  Grade 0 (scalar):   +1")
    print("  Grade 1 (vector):   +1")
    print("  Grade 2 (bivector): -1")
    print("  Grade 3 (trivector):-1")

    subsection("Vector reverse (unchanged)")
    v = Vector([1.0, 2.0, 3.0], grade=1, metric=g)
    print("v:")
    print(v)
    print()
    print("reverse(v):")
    print(reverse(v))
    print("  -> Same (grade-1 has sign +1)")

    subsection("Bivector reverse (negated)")
    e1 = basis_vector(0, g)
    e2 = basis_vector(1, g)
    b = e1 ^ e2
    print("b = e1 ^ e2:")
    print(b)
    print()
    print("reverse(b):")
    print(reverse(b))
    print("  -> Negated (grade-2 has sign -1)")

    subsection("Reverse of product: rev(AB) = rev(B) rev(A)")
    u = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
    v = Vector([0.0, 1.0, 0.0], grade=1, metric=g)

    UV = geometric(u, v)
    uv_scalar = grade_project(UV, 0)

    print("For u * v = <u * v>_0 + <u * v>_2:")
    print(f"  Scalar part unchanged: {uv_scalar.data}")
    print("  Bivector part negated")


# =============================================================================
# Section 5: Inverse
# =============================================================================


def demo_inverse() -> None:
    """Demonstrate blade inverses."""
    section("5. INVERSE")

    g = metric(3)

    subsection("Vector inverse: v^{-1} = v / |v|^2")
    v = Vector([3.0, 4.0, 0.0], grade=1, metric=g)  # |v|^2 = 25
    v_inv = inverse(v)
    print("v = [3, 4, 0]:")
    print(v)
    print()
    print("v^{-1}:")
    print(v_inv)
    print("  Expected: v / 25 = [0.12, 0.16, 0]")

    subsection("Verification: v^{-1} * v = 1")
    P = geometric(v_inv, v)
    scalar = grade_project(P, 0)
    print(f"v^{{-1}} * v = {scalar.data:.4g}")

    subsection("Bivector inverse")
    e1 = basis_vector(0, g)
    e2 = basis_vector(1, g)
    b = e1 ^ e2
    b_inv = inverse(b)
    print("b = e1 ^ e2:")
    print(b)
    print()
    print("b^{-1}:")
    print(b_inv)

    P = geometric(b_inv, b)
    scalar = grade_project(P, 0)
    print(f"b^{{-1}} * b = {scalar.data:.4g}")
    print("  -> Bivector inverse satisfies b^{-1} * b = 1")


# =============================================================================
# Section 6: Commutator and Anticommutator
# =============================================================================


def demo_commutators() -> None:
    """Demonstrate commutator and anticommutator products."""
    section("6. COMMUTATOR AND ANTICOMMUTATOR")

    g = metric(3)

    e1 = basis_vector(0, g)
    e2 = basis_vector(1, g)

    subsection("Commutator: [u, v] = (uv - vu) / 2")
    print("[e1, e2]:")
    print(commutator(e1, e2))
    print("  -> Antisymmetric part (bivector)")

    subsection("Anticommutator: {u, v} = (uv + vu) / 2")
    print("{e1, e2}:")
    print(anticommutator(e1, e2))
    print("  -> Symmetric part (scalar = dot product)")

    subsection("For orthogonal vectors:")
    print("  [e1, e2] = e1 ^ e2  (pure wedge)")
    print("  {e1, e2} = 0          (no dot)")


# =============================================================================
# Section 7: Grade Structure
# =============================================================================


def demo_grade_structure() -> None:
    """Demonstrate grade structure of geometric products."""
    section("7. GRADE STRUCTURE")

    g = metric(3)

    print("Grade selection rule for blade product:")
    print("  grades(u_j * v_k) = |j-k|, |j-k|+2, ..., j+k")

    subsection("Vector * Vector (grades 1+1 -> 0, 2)")
    e1 = basis_vector(0, g)
    e2 = basis_vector(1, g)
    print(f"  e1 * e2 has grades {list(geometric(e1, e2).data.keys())}")
    print("  Grade 0: scalar (dot product)")
    print("  Grade 2: bivector (wedge product)")

    subsection("Vector * Bivector (grades 1+2 -> 1, 3)")
    b = e1 ^ e2
    print(f"  e1 * (e1 ^ e2) has grades {list(geometric(e1, b).data.keys())}")
    print("  Grade 1: vector (contraction)")
    print("  Grade 3: trivector (extension, if possible)")

    subsection("Bivector * Bivector (grades 2+2 -> 0, 2, 4)")
    e3 = basis_vector(2, g)
    b1 = e1 ^ e2
    b2 = e2 ^ e3
    print(f"  (e1 ^ e2) * (e2 ^ e3) has grades {list(geometric(b1, b2).data.keys())}")
    print("  Grade 0: scalar")
    print("  Grade 2: bivector")
    print("  (Grade 4 not possible in 3D)")


# =============================================================================
# Section 8: Geometric Interpretation
# =============================================================================


def demo_geometric_interpretation() -> None:
    """Show geometric meaning of the geometric product."""
    section("8. GEOMETRIC INTERPRETATION")

    g = metric(2)

    print("The geometric product encodes both magnitude and orientation:")
    print("  u * v = |u||v| (cos(theta) + I sin(theta))")
    print("  where I is the unit bivector in the u-v plane")

    subsection("Bivector as rotation generator")
    e1 = basis_vector(0, g)
    e2 = basis_vector(1, g)

    # e1 ^ e2 gives a bivector
    I = e1 ^ e2
    print("I = e1 ^ e2 (bivector):")
    print(I)

    # I * e1 rotates e1 toward e2
    print()
    print("I * e1:")
    print(grade_project(geometric(I, e1), 1))
    print("  -> Vector component points in e2 direction")

    subsection("Bivector squared")
    scalar = grade_project(geometric(I, I), 0)
    print(f"I * I = {scalar.data:.4g}")
    print("  -> Scalar result (bivector contracted with itself)")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all demonstrations."""
    print()
    print("=" * 70)
    print("  GEOMETRIC PRODUCT DEMONSTRATION")
    print("  The fundamental operation of geometric algebra")
    print("=" * 70)

    demo_geometric_basics()
    demo_vector_contraction()
    demo_anticommutativity()
    demo_reversion()
    demo_inverse()
    demo_commutators()
    demo_grade_structure()
    demo_geometric_interpretation()

    section("DEMONSTRATION COMPLETE")
    print()


if __name__ == "__main__":
    main()
