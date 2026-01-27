"""
Duality Operations

Demonstrates complement and Hodge dual operations:
- Right and left complements (metric-independent)
- Hodge dual (metric-dependent)
- Meet operation via complements
- Grade mapping under duality

Run: uv run python -m morphis.examples.duality
"""

from morphis.elements import basis_vectors, euclidean_metric, pseudoscalar
from morphis.operations import (
    antiwedge,
    geometric,
    grade_project,
    hodge_dual,
    left_complement,
    meet,
    norm,
    right_complement,
)
from morphis.utils.pretty import section, subsection


# =============================================================================
# Section 1: The Pseudoscalar
# =============================================================================


def demo_pseudoscalar() -> None:
    """Demonstrate the unit pseudoscalar."""
    section("1. THE PSEUDOSCALAR")

    g = euclidean_metric(3)
    e1, e2, e3 = basis_vectors(g)

    subsection("Construction from basis")
    print("The pseudoscalar is the wedge of all basis vectors:")
    print("  I = e1 ^ e2 ^ e3")

    I = pseudoscalar(g)
    print("I:")
    print(I)
    print(f"  grade: {I.grade} (equals dimension)")

    subsection("Pseudoscalar squared")
    I_sq = geometric(I, I)
    scalar = grade_project(I_sq, 0)
    print(f"I * I = {scalar.data:.4g}")
    print("  -> In 3D Euclidean: I^2 = -1")

    subsection("Pseudoscalar via wedge")
    I_wedge = e1 ^ e2 ^ e3
    diff = norm(I - I_wedge)
    print(f"  e1 ^ e2 ^ e3 equals pseudoscalar(g): diff = {diff:.2e}")


# =============================================================================
# Section 2: Complements
# =============================================================================


def demo_complements() -> None:
    """Demonstrate right and left complements."""
    section("2. COMPLEMENTS (METRIC-INDEPENDENT)")

    g = euclidean_metric(3)
    e1, e2, _e3 = basis_vectors(g)

    print("Complements map grade k -> grade (d-k)")
    print("They use only the Levi-Civita symbol, not the metric.")

    subsection("Right complement of a vector")
    v = e1
    v_right = right_complement(v)
    print("v = e1:")
    print(v)
    print()
    print("right_complement(v):")
    print(v_right)
    print(f"  grade: {v.grade} -> {v_right.grade}")

    subsection("Left complement of a vector")
    v_left = left_complement(v)
    print("left_complement(v):")
    print(v_left)
    print(f"  grade: {v.grade} -> {v_left.grade}")

    subsection("Relationship: left = sign * right")
    print("  left(u) = (-1)^{grade * antigrade} * right(u)")
    print("  For grade 1 in 3D: antigrade = 2")
    print("  Sign = (-1)^{1*2} = +1")

    subsection("Complement of a bivector")
    b = e1 ^ e2
    b_right = right_complement(b)
    print("b = e1 ^ e2:")
    print(b)
    print()
    print("right_complement(b):")
    print(b_right)
    print(f"  grade: {b.grade} -> {b_right.grade}")
    print("  -> Bivector in 3D maps to vector (its 'normal')")

    subsection("Involution: double complement = identity")
    b_double = right_complement(right_complement(b))
    diff = norm(b - b_double)
    print(f"  right(right(b)) - b = {diff:.2e}")


# =============================================================================
# Section 3: Hodge Dual
# =============================================================================


def demo_hodge_dual() -> None:
    """Demonstrate the Hodge dual operation."""
    section("3. HODGE DUAL (METRIC-DEPENDENT)")

    g = euclidean_metric(3)
    e1, e2, e3 = basis_vectors(g)

    print("Hodge dual = complement + metric")
    print("  *u = G(right_complement(u))")
    print("where G raises/lowers indices via the metric.")

    subsection("Hodge dual of basis vectors")
    for i, e in enumerate([e1, e2, e3], 1):
        e_dual = hodge_dual(e)
        print(f"*e{i}:")
        print(e_dual)
        print()
    print("  -> Each vector maps to the bivector perpendicular to it")

    subsection("Hodge dual of a bivector")
    b = e1 ^ e2
    b_dual = hodge_dual(b)
    print("b = e1 ^ e2:")
    print(b)
    print()
    print("*b:")
    print(b_dual)
    print("  -> e1^e2 plane has normal e3")

    subsection("Double Hodge dual")
    print("  **u = (-1)^{k(d-k)} * sgn(g) * u")
    print("  For grade 1 in 3D Euclidean: (-1)^{1*2} * 1 = +1")
    v = e1
    v_double = hodge_dual(hodge_dual(v))
    diff = norm(v - v_double)
    print(f"  *(*e1) - e1 = {diff:.2e}")

    subsection("Cross product as Hodge dual of wedge")
    a = e1
    b = e2
    cross = hodge_dual(a ^ b)
    print("*(e1 ^ e2):")
    print(cross)
    print("  -> This is e3, the cross product e1 x e2")


# =============================================================================
# Section 4: Meet Operation
# =============================================================================


def demo_meet() -> None:
    """Demonstrate the meet (intersection) operation."""
    section("4. MEET (INTERSECTION VIA DUALITY)")

    g = euclidean_metric(3)
    e1, e2, e3 = basis_vectors(g)

    print("The meet finds the intersection of subspaces:")
    print("  A ∨ B = right(right(A) ^ right(B))")

    subsection("Meet of two planes (bivectors)")
    # Two planes in 3D intersect in a line (represented as a vector)
    plane1 = e1 ^ e2  # xy-plane
    plane2 = e2 ^ e3  # yz-plane

    print("plane1 = e1 ^ e2 (xy-plane):")
    print(plane1)
    print()
    print("plane2 = e2 ^ e3 (yz-plane):")
    print(plane2)

    # Their intersection is the y-axis
    intersection = meet(plane1, plane2)
    print()
    print("meet(plane1, plane2):")
    print(intersection)
    print("  -> Intersection is the y-axis (e2 direction)")

    subsection("Equivalent: antiwedge")
    intersection2 = antiwedge(plane1, plane2)
    diff = norm(intersection - intersection2)
    print(f"  antiwedge(A, B) = meet(A, B): diff = {diff:.2e}")

    subsection("Grade of meet")
    print(f"  grade(plane1) = {plane1.grade}")
    print(f"  grade(plane2) = {plane2.grade}")
    print(f"  grade(meet) = {intersection.grade}")
    print("  Formula: grade(A ∨ B) = grade(A) + grade(B) - d")
    print(f"  Check: 2 + 2 - 3 = {2 + 2 - 3}")


# =============================================================================
# Section 5: Summary
# =============================================================================


def demo_summary() -> None:
    """Summarize metric-independent vs metric-dependent operations."""
    section("5. SUMMARY: METRIC DEPENDENCE")

    print("METRIC-INDEPENDENT operations:")
    print("  - Wedge product (^)")
    print("  - Left/right complement")
    print("  - Meet/antiwedge")
    print()
    print("METRIC-DEPENDENT operations:")
    print("  - Interior product (<<, >>)")
    print("  - Hodge dual (*)")
    print("  - Norm and normalization")
    print("  - Projection and rejection")
    print()
    print("The complement uses only the Levi-Civita symbol.")
    print("The Hodge dual adds metric information via index raising.")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all demonstrations."""
    print()
    print("=" * 70)
    print("  DUALITY OPERATIONS DEMONSTRATION")
    print("  Complements, Hodge dual, and the meet operation")
    print("=" * 70)

    demo_pseudoscalar()
    demo_complements()
    demo_hodge_dual()
    demo_meet()
    demo_summary()

    section("DEMONSTRATION COMPLETE")
    print()


if __name__ == "__main__":
    main()
