"""
Projective Geometric Algebra (PGA) Transforms

Demonstrates PGA for rigid motions in Euclidean space:
- Points and directions (vectors at infinity)
- Lines and planes from points
- Translators and rotors
- Motors (combined rotation + translation)
- Distance calculations

Run: uv run python -m morphis.examples.transforms
"""

from numpy import pi

from morphis.elements import basis_vectors, pga_metric
from morphis.operations import geometric, norm
from morphis.transforms import (
    are_collinear,
    direction,
    distance_point_to_line,
    distance_point_to_point,
    euclidean,
    is_direction,
    is_point,
    line,
    plane,
    point,
    point_on_line,
    rotor,
    transform,
    translator,
    weight,
)
from morphis.utils.pretty import section, subsection


# =============================================================================
# Section 1: Points and Directions
# =============================================================================


def demo_points_directions() -> None:
    """Demonstrate point and direction embedding."""
    section("1. POINTS AND DIRECTIONS IN PGA")

    h = pga_metric(3)  # 3D PGA -> 4D ambient space

    print("PGA embeds d-dimensional Euclidean space in (d+1)-dimensional")
    print("Clifford algebra with metric diag(0, 1, 1, ..., 1).")
    print()
    print("  e0: ideal direction (e0^2 = 0)")
    print("  e1, e2, e3: Euclidean directions")

    subsection("Creating a point")
    p = point([1.0, 2.0, 3.0], metric=h)
    print("p = point([1, 2, 3]):")
    print(p)
    print("  Point = e0 + x1*e1 + x2*e2 + x3*e3")
    print(f"  weight(p) = {weight(p)} (finite point)")

    subsection("Creating a direction (point at infinity)")
    d = direction([1.0, 0.0, 0.0], metric=h)
    print("d = direction([1, 0, 0]):")
    print(d)
    print("  Direction = 0*e0 + d1*e1 + d2*e2 + d3*e3")
    print(f"  weight(d) = {weight(d)} (at infinity)")

    subsection("Testing point vs direction")
    print(f"  is_point(p) = {is_point(p)}")
    print(f"  is_direction(d) = {is_direction(d)}")
    print(f"  is_point(d) = {is_point(d)}")

    subsection("Extracting Euclidean coordinates")
    coords = euclidean(p)
    print(f"  euclidean(p) = {coords}")


# =============================================================================
# Section 2: Lines and Planes
# =============================================================================


def demo_lines_planes() -> None:
    """Demonstrate line and plane construction."""
    section("2. LINES AND PLANES")

    h = pga_metric(3)

    subsection("Line through two points")
    p1 = point([0.0, 0.0, 0.0], metric=h)
    p2 = point([1.0, 0.0, 0.0], metric=h)

    L = line(p1, p2)
    print("L = line(origin, [1,0,0]):")
    print(L)
    print(f"  grade: {L.grade} (line is a bivector in PGA)")

    subsection("Equivalent: wedge of two points")
    L_wedge = p1 ^ p2
    diff = norm(L - L_wedge)
    print(f"  p1 ^ p2 equals line(p1, p2): diff = {diff:.2e}")

    subsection("Plane through three points")
    p3 = point([0.0, 1.0, 0.0], metric=h)

    P = plane(p1, p2, p3)
    print("P = plane(origin, [1,0,0], [0,1,0]):")
    print(P)
    print(f"  grade: {P.grade} (plane is a trivector in PGA)")

    subsection("Equivalent: wedge of three points")
    P_wedge = p1 ^ p2 ^ p3
    diff = norm(P - P_wedge)
    print(f"  p1 ^ p2 ^ p3 equals plane(p1, p2, p3): diff = {diff:.2e}")


# =============================================================================
# Section 3: Incidence
# =============================================================================


def demo_incidence() -> None:
    """Demonstrate incidence predicates."""
    section("3. INCIDENCE PREDICATES")

    h = pga_metric(3)

    subsection("Collinearity test")
    p1 = point([0.0, 0.0, 0.0], metric=h)
    p2 = point([1.0, 0.0, 0.0], metric=h)
    p3 = point([2.0, 0.0, 0.0], metric=h)
    p4 = point([0.0, 1.0, 0.0], metric=h)

    print("  Points on x-axis: origin, [1,0,0], [2,0,0]")
    print(f"  are_collinear(p1, p2, p3) = {are_collinear(p1, p2, p3)}")

    print()
    print("  Adding [0,1,0]:")
    print(f"  are_collinear(p1, p2, p4) = {are_collinear(p1, p2, p4)}")

    subsection("Point on line test")
    L = line(p1, p2)
    print("  Line through origin and [1,0,0]:")
    print(f"  point_on_line([2,0,0], L) = {point_on_line(p3, L)}")
    print(f"  point_on_line([0,1,0], L) = {point_on_line(p4, L)}")


# =============================================================================
# Section 4: Translation
# =============================================================================


def demo_translation() -> None:
    """Demonstrate the translator versor."""
    section("4. TRANSLATION")

    h = pga_metric(3)
    _e0, _e1, _e2, _e3 = basis_vectors(h)  # noqa: F841

    print("Translator T = exp(-t/2) = 1 - (1/2) t")
    print("where t is a degenerate bivector e0 ^ direction.")

    subsection("Create translator")
    d = direction([2.0, 0.0, 0.0], metric=h)
    T = translator(d)
    print("T = translator([2,0,0]):")
    print(T)
    print(f"  Translator has grades {T.grades}")

    subsection("Apply translation to a point")
    p = point([1.0, 0.0, 0.0], metric=h)
    p_translated = transform(p, T)

    print(f"  Original: euclidean(p) = {euclidean(p)}")
    print(f"  After T:  euclidean(T*p*~T) = {euclidean(p_translated)}")
    print("  -> Moved by [2, 0, 0]")

    subsection("Translation is additive")
    d1 = direction([1.0, 0.0, 0.0], metric=h)
    d2 = direction([0.0, 1.0, 0.0], metric=h)
    T1 = translator(d1)
    T2 = translator(d2)

    # Compose translations
    T_composed = geometric(T2, T1)

    p = point([0.0, 0.0, 0.0], metric=h)
    p_result = transform(transform(p, T1), T2)
    p_direct = transform(p, T_composed)

    print(f"  T1 then T2: {euclidean(p_result)}")
    print(f"  T2*T1:      {euclidean(p_direct)}")


# =============================================================================
# Section 5: Rotation about a point
# =============================================================================


def demo_rotation() -> None:
    """Demonstrate rotation using rotors."""
    section("5. ROTATION IN PGA")

    h = pga_metric(3)
    _e0, e1, e2, _e3 = basis_vectors(h)

    subsection("Rotation about origin")
    # Bivector for xy-plane rotation (in Euclidean subspace)
    b = e1 ^ e2
    b_unit = b / norm(b)

    R = rotor(b_unit, pi / 2)
    print("R = rotor(e1^e2, pi/2):")
    print(R)

    # Rotate a point
    p = point([1.0, 0.0, 0.0], metric=h)
    p_rotated = transform(p, R)

    print(f"  Original: {euclidean(p)}")
    print(f"  Rotated:  {euclidean(p_rotated)}")
    print("  -> [1,0,0] becomes [0,1,0] (90° in xy-plane)")

    subsection("Rotation preserves distance from origin")
    d_before = distance_point_to_point(p, point([0, 0, 0], metric=h))
    d_after = distance_point_to_point(p_rotated, point([0, 0, 0], metric=h))
    print(f"  Distance before: {d_before:.4f}")
    print(f"  Distance after:  {d_after:.4f}")


# =============================================================================
# Section 6: Distances
# =============================================================================


def demo_distances() -> None:
    """Demonstrate distance calculations."""
    section("6. DISTANCES")

    h = pga_metric(3)

    subsection("Point to point distance")
    p1 = point([0.0, 0.0, 0.0], metric=h)
    p2 = point([3.0, 4.0, 0.0], metric=h)

    d = distance_point_to_point(p1, p2)
    print(f"  d(origin, [3,4,0]) = {d:.4f}")
    print("  -> sqrt(3^2 + 4^2) = 5")

    subsection("Point to line distance")
    # Line along x-axis
    L = line(point([0, 0, 0], metric=h), point([1, 0, 0], metric=h))

    # Point above the line
    p = point([0.0, 3.0, 0.0], metric=h)

    d = distance_point_to_line(p, L)
    print("  Line: x-axis")
    print("  Point: [0, 3, 0]")
    print(f"  Distance: {d:.4f}")


# =============================================================================
# Section 7: Summary
# =============================================================================


def demo_summary() -> None:
    """Summarize PGA elements and operations."""
    section("7. PGA SUMMARY")

    print("ELEMENTS:")
    print("  Point    = e0 + x^m e_m     (weight = 1)")
    print("  Direction= x^m e_m          (weight = 0)")
    print("  Line     = p ^ q            (grade 2)")
    print("  Plane    = p ^ q ^ r        (grade 3)")
    print()
    print("VERSORS:")
    print("  Translator T = 1 - (1/2) e0 ^ direction")
    print("  Rotor R = exp(-B*theta/2)")
    print("  Motor M = T * R or R * T")
    print()
    print("ACTION:")
    print("  x' = M * x * ~M  (sandwich product)")
    print()
    print("PROPERTIES:")
    print("  - Rigid motions preserve distances")
    print("  - Composition via geometric product")
    print("  - Smooth interpolation via exp/log")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all demonstrations."""
    print()
    print("=" * 70)
    print("  PROJECTIVE GEOMETRIC ALGEBRA (PGA)")
    print("  Rigid motions: translations, rotations, and motors")
    print("=" * 70)

    demo_points_directions()
    demo_lines_planes()
    demo_incidence()
    demo_translation()
    demo_rotation()
    demo_distances()
    demo_summary()

    section("DEMONSTRATION COMPLETE")
    print()


if __name__ == "__main__":
    main()
