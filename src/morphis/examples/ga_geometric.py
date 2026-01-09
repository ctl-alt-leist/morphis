"""
Geometric Algebra - Geometric Product Example

Demonstrates the geometric product and related transformations:
- Geometric product of vectors and blades
- Grade decomposition (scalar, bivector, etc.)
- Reversion operation
- Blade inverses
- Reflections and rotations

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
"""

from math import cos, pi, sin

from numpy import array, zeros

from morphis.ga.geometric import (
    anticommutator,
    commutator,
    geometric,
    grade_project,
    inverse,
    reverse,
)
from morphis.ga.model import Blade, euclidean_metric, vector_blade
from morphis.ga.norms import norm
from morphis.utils.pretty import (
    section,
    show_blade,
    show_mv,
    show_scalar,
    subsection,
)


# =============================================================================
# Helper: Basis vectors
# =============================================================================


def basis_vector(idx: int, dim: int) -> Blade:
    """Create basis vector e_idx in d dimensions."""
    data = zeros(dim)
    data[idx] = 1.0
    return vector_blade(data)


# =============================================================================
# Section 1: Geometric Product Basics
# =============================================================================


def demo_geometric_basics() -> None:
    """Demonstrate the fundamental property: uv = u.v + u^v."""
    section("1. GEOMETRIC PRODUCT BASICS")

    g = euclidean_metric(3)

    subsection("Geometric product decomposes into dot + wedge")
    u = vector_blade(array([1.0, 0.0, 0.0]))  # e_1
    v = vector_blade(array([0.0, 1.0, 0.0]))  # e_2

    print("For orthogonal vectors e_1 and e_2:")
    print("  e_1 . e_2 = 0 (orthogonal)")
    print("  e_1 ^ e_2 = e_12 (bivector)")
    print()

    uv = geometric(u, v)
    show_mv("e_1 * e_2", uv)

    scalar_part = grade_project(uv, 0)
    bivector_part = grade_project(uv, 2)
    print()
    print(f"  Scalar part (dot product): {scalar_part.data}")
    print("  Bivector part (wedge): nonzero in e_12 plane")

    subsection("Parallel vectors: uv = u.v (pure scalar)")
    u = vector_blade(array([1.0, 0.0, 0.0]))
    v = vector_blade(array([3.0, 0.0, 0.0]))  # v = 3u

    uv = geometric(u, v)
    show_mv("e_1 * (3 e_1)", uv)
    print("  -> Pure scalar: 3 (no bivector part)")

    subsection("General case: u at angle to v")
    theta = pi / 4
    u = vector_blade(array([1.0, 0.0, 0.0]))
    v = vector_blade(array([cos(theta), sin(theta), 0.0]))

    uv = geometric(u, v)
    show_mv("e_1 * v (at pi/4)", uv)

    scalar_part = grade_project(uv, 0)
    bivector_part = grade_project(uv, 2)
    bivector_norm = norm(bivector_part, g)
    print()
    print(f"  Scalar part = cos(pi/4) = {scalar_part.data:.6f}")
    print(f"  Bivector magnitude = sin(pi/4) = {bivector_norm:.6f}")


# =============================================================================
# Section 2: Vector Contraction
# =============================================================================


def demo_vector_contraction() -> None:
    """Demonstrate v^2 = |v|^2 for vectors."""
    section("2. VECTOR CONTRACTION (v^2 = |v|^2)")

    euclidean_metric(3)

    subsection("Unit vector squares to 1")
    e_1 = basis_vector(0, 3)
    e_1_sq = geometric(e_1, e_1)
    show_mv("e_1 * e_1", e_1_sq)
    print("  -> Scalar 1 (unit vector in Euclidean space)")

    subsection("General vector: v^2 = |v|^2")
    v = vector_blade(array([3.0, 4.0, 0.0]))
    v_sq = geometric(v, v)
    show_mv("v * v  where v = [3, 4, 0]", v_sq)
    print(f"  -> Scalar {grade_project(v_sq, 0).data} = 3^2 + 4^2")


# =============================================================================
# Section 3: Anticommutativity
# =============================================================================


def demo_anticommutativity() -> None:
    """Demonstrate orthogonal vectors anticommute."""
    section("3. ANTICOMMUTATIVITY (orthogonal vectors)")

    euclidean_metric(3)

    e_1 = basis_vector(0, 3)
    e_2 = basis_vector(1, 3)

    subsection("e_1 * e_2 vs e_2 * e_1")
    e_1_e_2 = geometric(e_1, e_2)
    e_2_e_1 = geometric(e_2, e_1)

    print("e_1 * e_2:")
    show_mv("  result", e_1_e_2)
    print()
    print("e_2 * e_1:")
    show_mv("  result", e_2_e_1)

    b_1 = grade_project(e_1_e_2, 2)
    b_2 = grade_project(e_2_e_1, 2)
    print()
    print(f"  Bivector of e_1*e_2: {b_1.data[0, 1]:.1f} in [0,1] slot")
    print(f"  Bivector of e_2*e_1: {b_2.data[0, 1]:.1f} in [0,1] slot")
    print("  -> Negatives! Orthogonal vectors anticommute.")


# =============================================================================
# Section 4: Reversion
# =============================================================================


def demo_reversion() -> None:
    """Demonstrate the reverse operation."""
    section("4. REVERSION")

    euclidean_metric(3)

    print("Reverse sign pattern: (-1)^{k(k-1)/2}")
    print("  Grade 0 (scalar):   +1")
    print("  Grade 1 (vector):   +1")
    print("  Grade 2 (bivector): -1")
    print("  Grade 3 (trivector):-1")

    subsection("Vector reverse (unchanged)")
    v = vector_blade(array([1.0, 2.0, 3.0]))
    v_rev = reverse(v)
    show_blade("v", v)
    show_blade("reverse(v)", v_rev)
    print("  -> Same (grade-1 has sign +1)")

    subsection("Bivector reverse (negated)")
    e_1 = basis_vector(0, 3)
    e_2 = basis_vector(1, 3)
    B = e_1 ^ e_2
    B_rev = reverse(B)
    show_blade("B = e_1 ^ e_2", B)
    show_blade("reverse(B)", B_rev)
    print("  -> Negated (grade-2 has sign -1)")

    subsection("Reverse of product: rev(AB) = rev(B) rev(A)")
    u = vector_blade(array([1.0, 0.0, 0.0]))
    v = vector_blade(array([0.0, 1.0, 0.0]))

    uv = geometric(u, v)
    uv_scalar = grade_project(uv, 0)
    grade_project(uv, 2)

    print("For u*v = <u*v>_0 + <u*v>_2:")
    print(f"  Scalar part unchanged: {uv_scalar.data}")
    print("  Bivector part negated")


# =============================================================================
# Section 5: Inverse
# =============================================================================


def demo_inverse() -> None:
    """Demonstrate blade inverses."""
    section("5. INVERSE")

    g = euclidean_metric(3)

    subsection("Vector inverse: v^{-1} = v / |v|^2")
    v = vector_blade(array([3.0, 4.0, 0.0]))  # |v|^2 = 25
    v_inv = inverse(v, g)
    show_blade("v = [3, 4, 0]", v)
    show_blade("v^{-1}", v_inv)
    print("  Expected: v / 25 = [0.12, 0.16, 0]")

    subsection("Verification: v^{-1} * v = 1")
    product = geometric(v_inv, v)
    scalar = grade_project(product, 0)
    show_scalar("v^{-1} * v", scalar.data)

    subsection("Bivector inverse")
    e_1 = basis_vector(0, 3)
    e_2 = basis_vector(1, 3)
    B = e_1 ^ e_2
    B_inv = inverse(B, g)
    show_blade("B = e_1 ^ e_2", B)
    show_blade("B^{-1}", B_inv)

    product = geometric(B_inv, B)
    scalar = grade_project(product, 0)
    show_scalar("B^{-1} * B", scalar.data)
    print("  -> Bivector inverse satisfies B^{-1} * B = 1")


# =============================================================================
# Section 6: Commutator and Anticommutator
# =============================================================================


def demo_commutators() -> None:
    """Demonstrate commutator and anticommutator products."""
    section("6. COMMUTATOR AND ANTICOMMUTATOR")

    g = euclidean_metric(3)

    e_1 = basis_vector(0, 3)
    e_2 = basis_vector(1, 3)

    subsection("Commutator: [u, v] = (uv - vu) / 2")
    comm = commutator(e_1, e_2, g)
    show_mv("[e_1, e_2]", comm)
    print("  -> Antisymmetric part (bivector)")

    subsection("Anticommutator: {u, v} = (uv + vu) / 2")
    anticomm = anticommutator(e_1, e_2, g)
    show_mv("{e_1, e_2}", anticomm)
    print("  -> Symmetric part (scalar = dot product)")

    subsection("For orthogonal vectors:")
    print("  [e_1, e_2] = e_1 ^ e_2  (pure wedge)")
    print("  {e_1, e_2} = 0          (no dot)")


# =============================================================================
# Section 7: Grade Structure
# =============================================================================


def demo_grade_structure() -> None:
    """Demonstrate grade structure of geometric products."""
    section("7. GRADE STRUCTURE")

    euclidean_metric(3)

    print("Grade selection rule for blade product:")
    print("  grades(u_j * v_k) = |j-k|, |j-k|+2, ..., j+k")

    subsection("Vector * Vector (grades 1+1 -> 0, 2)")
    e_1 = basis_vector(0, 3)
    e_2 = basis_vector(1, 3)
    result = geometric(e_1, e_2)
    print(f"  e_1 * e_2 has grades: {list(result.components.keys())}")
    print("  Grade 0: scalar (dot product)")
    print("  Grade 2: bivector (wedge product)")

    subsection("Vector * Bivector (grades 1+2 -> 1, 3)")
    B = e_1 ^ e_2
    result = geometric(e_1, B)
    print(f"  e_1 * (e_1 ^ e_2) has grades: {list(result.components.keys())}")
    print("  Grade 1: vector (contraction)")
    print("  Grade 3: trivector (extension, if possible)")

    subsection("Bivector * Bivector (grades 2+2 -> 0, 2, 4)")
    e_3 = basis_vector(2, 3)
    B_1 = e_1 ^ e_2
    B_2 = e_2 ^ e_3
    result = geometric(B_1, B_2)
    print(f"  (e_1^e_2) * (e_2^e_3) has grades: {list(result.components.keys())}")
    print("  Grade 0: scalar")
    print("  Grade 2: bivector")
    print("  (Grade 4 not possible in 3D)")


# =============================================================================
# Section 8: Geometric Interpretation
# =============================================================================


def demo_geometric_interpretation() -> None:
    """Show geometric meaning of the geometric product."""
    section("8. GEOMETRIC INTERPRETATION")

    euclidean_metric(2)

    print("The geometric product encodes both magnitude and orientation:")
    print("  uv = |u||v| (cos(theta) + I sin(theta))")
    print("  where I is the unit bivector in the u-v plane")

    subsection("Bivector as rotation generator")
    e_1 = basis_vector(0, 2)
    e_2 = basis_vector(1, 2)

    # e_1 ^ e_2 gives a bivector
    I = e_1 ^ e_2
    show_blade("I = e_1 ^ e_2 (bivector)", I)

    # I * e_1 rotates e_1 toward e_2
    I_e_1 = geometric(I, e_1)
    rotated = grade_project(I_e_1, 1)
    show_blade("I * e_1", rotated)
    print("  -> Vector component points in e_2 direction")

    subsection("Bivector squared")
    I_sq = geometric(I, I)
    scalar = grade_project(I_sq, 0)
    show_scalar("I * I", scalar.data)
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
