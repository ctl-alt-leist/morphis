"""
Geometric Algebra - Complex Phasors Example

Demonstrates complex-valued blades for phasor representation:
- Creating complex blades from magnitude and phase
- Conjugation and Hermitian norms
- Operations on complex blades (wedge, hodge)
- Physical interpretation for AC electromagnetic fields

The imaginary unit in phasors represents temporal phase rotation,
distinct from the geometric pseudoscalar.
"""

from numpy import array, exp, pi, real

from morphis.elements import Vector, metric
from morphis.operations import (
    conjugate,
    hermitian_norm_squared,
    hodge_dual,
    norm_squared,
    wedge,
)
from morphis.utils.pretty import section, subsection


# =============================================================================
# Section 1: Complex Vector Creation
# =============================================================================


def demo_complex_creation() -> None:
    """Demonstrate creating complex-valued blades."""
    section("1. COMPLEX BLADE CREATION")

    g = metric(3)

    subsection("Direct complex array")
    v = Vector([1 + 0j, 2 + 1j, 0 - 1j], grade=1, metric=g)
    print("v (complex vector):")
    print(v)
    print(f"  dtype: {v.data.dtype}")

    subsection("Phasor from magnitude and phase")
    amplitude = array([1.0, 0.5, 0.25])
    phase = pi / 4  # 45 degrees
    v_phasor = Vector(amplitude * exp(1j * phase), grade=1, metric=g)
    print("v_phasor:")
    print(v_phasor)
    print(f"  phase: {phase:.4f} rad = {phase * 180 / pi:.1f} deg")

    subsection("Complex scalar (overall phasor)")
    s = Vector(2 * exp(1j * pi / 6), grade=0, metric=g)
    print("scalar phasor:")
    print(s)
    print(f"  |s| = {abs(s.data):.4f}")
    print(f"  arg(s) = {phase:.4f} rad")

    subsection("Scale real blade by complex scalar")
    e1 = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
    e1_rotated = e1 * exp(1j * pi / 3)
    print("e1 * exp(i*pi/3):")
    print(e1_rotated)
    print("  -> Real blade becomes complex")


# =============================================================================
# Section 2: Conjugation
# =============================================================================


def demo_conjugation() -> None:
    """Demonstrate complex conjugation."""
    section("2. CONJUGATION")

    g = metric(3)

    subsection("Conjugate function and method")
    v = Vector([1 + 2j, 3 - 4j, 5j], grade=1, metric=g)
    print("v:")
    print(v)
    print()
    print("conjugate(v):")
    print(conjugate(v))
    print()
    print("v.conj():")
    print(v.conj())

    subsection("Double conjugation is identity")
    print("v.conj().conj():")
    print(v.conj().conj())
    print("  -> Same as original v")

    subsection("Conjugation on bivector")
    e1 = Vector([1.0, 0, 0], grade=1, metric=g)
    e2 = Vector([0, 1.0, 0], grade=1, metric=g)
    B = (e1 ^ e2) * (1 + 1j)
    print("B = (e1 ^ e2) * (1 + i):")
    print(B)
    print()
    print("B.conj():")
    print(B.conj())


# =============================================================================
# Section 3: Bilinear vs Hermitian Norms
# =============================================================================


def demo_norms() -> None:
    """Demonstrate the difference between bilinear and Hermitian norms."""
    section("3. BILINEAR vs HERMITIAN NORMS")

    g = metric(3)

    subsection("Real blade: both norms agree")
    v_real = Vector([3.0, 4.0, 0.0], grade=1, metric=g)
    print("v (real):")
    print(v_real)
    print(f"  norm_squared(v) = {norm_squared(v_real)}")
    print(f"  hermitian_norm_squared(v) = {hermitian_norm_squared(v_real)}")
    print("  -> Identical for real blades")

    subsection("Pure phasor: bilinear gives complex, Hermitian gives real")
    phase = pi / 4
    v_phasor = Vector([3.0, 4.0, 0.0], grade=1, metric=g) * exp(1j * phase)
    print("v_phasor = [3,4,0] * exp(i*pi/4):")
    print(v_phasor)

    ns = norm_squared(v_phasor)
    hns = hermitian_norm_squared(v_phasor)
    print()
    print(f"  norm_squared(v_phasor) = {ns}")
    print("    -> Complex! Phase doubled: exp(i*pi/2) = i")
    print()
    print(f"  hermitian_norm_squared(v_phasor) = {hns}")
    print("    -> Real! |amplitude|^2 = 25")

    subsection("Mixed-phase blade: the critical case")
    v_mixed = Vector([1, 1j, 0], grade=1, metric=g)
    print("v_mixed = [1, i, 0]:")
    print(v_mixed)

    ns = norm_squared(v_mixed)
    hns = hermitian_norm_squared(v_mixed)
    print()
    print(f"  norm_squared(v_mixed) = {ns}")
    print("    -> Bilinear: 1^2 + i^2 = 1 - 1 = 0 (cancellation!)")
    print()
    print(f"  hermitian_norm_squared(v_mixed) = {hns}")
    print("    -> Hermitian: |1|^2 + |i|^2 = 1 + 1 = 2 (correct)")

    subsection("Summary: use Hermitian norm for physical amplitudes")
    print("  norm_squared      -> algebraic (GA inner product)")
    print("  hermitian_norm_squared -> physical (RMS amplitude)")


# =============================================================================
# Section 4: Operations on Complex Vectors
# =============================================================================


def demo_operations() -> None:
    """Demonstrate GA operations work naturally with complex blades."""
    section("4. OPERATIONS ON COMPLEX BLADES")

    g = metric(3)

    subsection("Wedge product of phasors")
    phase1 = pi / 4
    phase2 = pi / 6
    u = Vector([1, 0, 0], grade=1, metric=g) * exp(1j * phase1)
    v = Vector([0, 1, 0], grade=1, metric=g) * exp(1j * phase2)

    uv = u ^ v
    print("u ^ v:")
    print(uv)
    print(f"  Combined phase: {phase1 + phase2:.4f} rad")
    print(f"  Result phase: {phase1 + phase2:.4f} rad (phases add)")

    subsection("Hodge dual of complex blade")
    e3 = Vector([0, 0, 1], grade=1, metric=g) * exp(1j * pi / 3)
    print("e3 * exp(i*pi/3):")
    print(e3)
    print()
    print("hodge(e3):")
    print(e3.hodge())
    print("  -> Phase preserved in dual")

    subsection("Method chaining: conj then hodge")
    result = e3.conj().hodge()
    print("e3.conj().hodge():")
    print(result)


# =============================================================================
# Section 5: Physical Application - EM Phasors
# =============================================================================


def demo_em_phasors() -> None:
    """Demonstrate phasor representation for electromagnetic fields."""
    section("5. ELECTROMAGNETIC PHASORS")

    g = metric(3)

    print("In AC electromagnetics, fields are phasors:")
    print("  E(r, t) = Re[E_tilde(r) * exp(-i*omega*t)]")
    print("  B(r, t) = Re[B_tilde(r) * exp(-i*omega*t)]")
    print()
    print("The complex amplitudes E_tilde, B_tilde are blades in morphis.")

    subsection("Example: plane wave propagating in z")
    # E = E_0 * exp(i*kz) in x-direction
    # B = B_0 * exp(i*kz) in y-direction (related by impedance)
    E0 = 1.0
    B0 = E0 / 3e8  # B = E/c in vacuum
    phase = pi / 4  # arbitrary phase at z=0

    E_tilde = Vector([E0 * exp(1j * phase), 0, 0], grade=1, metric=g)
    B_tilde = Vector([0, B0 * exp(1j * phase), 0], grade=1, metric=g)

    print("E_tilde (electric phasor):")
    print(E_tilde)
    print()
    print("B_tilde (magnetic phasor):")
    print(B_tilde)

    subsection("Time-averaged Poynting vector")
    print("  S = (1/2) Re(E ^ B*)")

    # Compute E ^ conj(B)
    S_complex = wedge(E_tilde, B_tilde.conj())
    print("E ^ B*:")
    print(S_complex)

    # For a bivector, hodge dual gives vector
    S_dual = hodge_dual(S_complex)
    S_avg = 0.5 * real(S_dual.data)
    print()
    print(f"  S_avg (vector form) = {S_avg}")
    print("  -> Points in z-direction (wave propagation)")

    subsection("Field energy density")
    print("  u = (1/2) * (epsilon * |E|^2 + (1/mu) * |B|^2)")
    E_mag_sq = hermitian_norm_squared(E_tilde)
    B_mag_sq = hermitian_norm_squared(B_tilde)
    print(f"  |E|^2 = {E_mag_sq}")
    print(f"  |B|^2 = {B_mag_sq}")
    print("  -> Hermitian norm gives real amplitudes for energy")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all demonstrations."""
    print()
    print("=" * 70)
    print("  COMPLEX PHASORS DEMONSTRATION")
    print("  Vectors with complex coefficients for AC field analysis")
    print("=" * 70)

    demo_complex_creation()
    demo_conjugation()
    demo_norms()
    demo_operations()
    demo_em_phasors()

    section("DEMONSTRATION COMPLETE")
    print()


if __name__ == "__main__":
    main()
