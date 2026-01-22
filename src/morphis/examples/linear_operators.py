"""
Geometric Algebra - Linear Operators Example

Demonstrates Operator for structured linear maps between blade spaces:
- Creating operators with BladeSpec for input/output structure
- Forward application: P = L * q
- Adjoint: L^H with proper conjugate transpose
- Least squares inversion with regularization
- SVD decomposition for analysis
- Complex-valued operators for phasor systems

Example system: scalar sources q mapping to bivector fields P via transfer operator L.
Index structure: P^{ab}_m = L^{ab}_{mn} q_n
"""

import numpy as np
from numpy import exp, pi

from morphis.algebra import BladeSpec
from morphis.elements import Blade, Operator, euclidean
from morphis.utils.pretty import section, show_array, show_blade, subsection


# =============================================================================
# Section 1: Operator Construction
# =============================================================================


def demo_operator_construction() -> None:
    """Demonstrate creating Operator with BladeSpec."""
    section("1. OPERATOR CONSTRUCTION")

    d = 3  # Vector space dimension
    M = 4  # Output collection size (field points)
    N = 5  # Input collection size (source points)

    subsection("Define input and output structure")
    print(f"  Input:  scalar sources q_n, shape ({N},)")
    print(f"  Output: bivector fields P^{{ab}}_m, shape ({M}, {d}, {d})")
    print()
    print("  BladeSpec describes the structure:")
    print("    grade: 0=scalar, 1=vector, 2=bivector, ...")
    print("    collection: number of batch dimensions")
    print("    dim: underlying vector space dimension")

    input_spec = BladeSpec(grade=0, collection=1, dim=d)
    output_spec = BladeSpec(grade=2, collection=1, dim=d)
    print()
    print(f"  input_spec  = {input_spec}")
    print(f"  output_spec = {output_spec}")

    subsection("Create transfer operator L^{ab}_{mn}")
    # Operator data layout: (*out_geo, *out_coll, *in_coll, *in_geo)
    # For scalar->bivector: (d, d, M, N)
    np.random.seed(42)
    L_data = np.random.randn(d, d, M, N)
    # Antisymmetrize for valid bivector output
    L_data = (L_data - L_data.transpose(1, 0, 2, 3)) / 2

    L = Operator(
        data=L_data,
        input_spec=input_spec,
        output_spec=output_spec,
        metric=euclidean(d),
    )
    print(f"  L.shape = {L.shape}")
    print(f"  L.input_shape = {L.input_shape}")
    print(f"  L.output_shape = {L.output_shape}")

    subsection("Operator properties")
    print(f"  L.dim = {L.dim}")
    print(f"  L.input_collection = {L.input_collection}")
    print(f"  L.output_collection = {L.output_collection}")


# =============================================================================
# Section 2: Forward Application
# =============================================================================


def demo_forward_application() -> None:
    """Demonstrate applying operator to compute P = L * q."""
    section("2. FORWARD APPLICATION")

    d, M, N = 3, 4, 5
    np.random.seed(42)
    L_data = np.random.randn(d, d, M, N)
    L_data = (L_data - L_data.transpose(1, 0, 2, 3)) / 2

    L = Operator(
        data=L_data,
        input_spec=BladeSpec(grade=0, collection=1, dim=d),
        output_spec=BladeSpec(grade=2, collection=1, dim=d),
        metric=euclidean(d),
    )

    subsection("Create source distribution q")
    q = Blade(np.array([1.0, 0.5, -0.3, 0.8, -0.2]), grade=0, metric=euclidean(d))
    show_blade("q (scalar sources)", q)
    print(f"  shape: {q.shape}")

    subsection("Apply operator: P = L * q")
    P = L * q
    show_blade("P (bivector field)", P)
    print(f"  shape: {P.shape}")
    print(f"  grade: {P.grade}")

    subsection("Equivalent syntax forms")
    P1 = L.apply(q)
    P2 = L * q
    P3 = L(q)
    print("  L.apply(q), L * q, L(q) all produce the same result")
    print(f"  Max difference: {np.max(np.abs(P1.data - P2.data) + np.abs(P2.data - P3.data)):.2e}")

    subsection("Verify antisymmetry of bivector output")
    # P^{ab} = -P^{ba}
    antisym_error = np.max(np.abs(P.data + P.data.transpose(0, 2, 1)))
    print(f"  P + P^T = {antisym_error:.2e} (should be ~0)")


# =============================================================================
# Section 3: Adjoint Operator
# =============================================================================


def demo_adjoint() -> None:
    """Demonstrate adjoint L^H with inner product property."""
    section("3. ADJOINT OPERATOR")

    d, M, N = 3, 4, 5
    np.random.seed(42)
    L_data = np.random.randn(d, d, M, N)

    L = Operator(
        data=L_data,
        input_spec=BladeSpec(grade=0, collection=1, dim=d),
        output_spec=BladeSpec(grade=2, collection=1, dim=d),
        metric=euclidean(d),
    )

    subsection("Compute adjoint")
    L_H = L.adjoint()
    print(f"  L maps: {L.input_shape} -> {L.output_shape}")
    print(f"  L^H maps: {L_H.input_shape} -> {L_H.output_shape}")
    print()
    print("  Adjoint swaps input/output specs")

    subsection("Alternative syntax: L.H")
    print("  L.H is shorthand for L.adjoint()")
    print(f"  Max diff: {np.max(np.abs(L.H.data - L.adjoint().data)):.2e}")

    subsection("Adjoint involution: (L^H)^H = L")
    L_HH = L.H.H
    print(f"  Max |L - (L^H)^H|: {np.max(np.abs(L.data - L_HH.data)):.2e}")

    subsection("Inner product property: <Lq, P> = <q, L^H P>")
    q = Blade(np.random.randn(N), grade=0, metric=euclidean(d))
    P = Blade(np.random.randn(M, d, d), grade=2, metric=euclidean(d))

    Lq = L * q
    LhP = L.H * P

    inner1 = np.sum(Lq.data.conj() * P.data)
    inner2 = np.sum(q.data.conj() * LhP.data)

    print(f"  <Lq, P>   = {inner1:.6f}")
    print(f"  <q, L^H P> = {inner2:.6f}")
    print(f"  Difference: {abs(inner1 - inner2):.2e}")


# =============================================================================
# Section 4: Least Squares Inversion
# =============================================================================


def demo_least_squares() -> None:
    """Demonstrate solving q = L.solve(P) for source estimation."""
    section("4. LEAST SQUARES INVERSION")

    d, M, N = 3, 20, 5  # Overdetermined: more measurements than sources
    np.random.seed(42)
    L_data = np.random.randn(d, d, M, N)
    L_data = (L_data - L_data.transpose(1, 0, 2, 3)) / 2

    L = Operator(
        data=L_data,
        input_spec=BladeSpec(grade=0, collection=1, dim=d),
        output_spec=BladeSpec(grade=2, collection=1, dim=d),
        metric=euclidean(d),
    )

    subsection("Generate synthetic data")
    q_true = Blade(np.array([1.0, -0.5, 0.3, 0.8, -0.2]), grade=0, metric=euclidean(d))
    P = L * q_true
    show_blade("q_true (ground truth)", q_true)
    print(f"  P shape: {P.shape} (overdetermined: {M} > {N})")

    subsection("Recover sources via least squares")
    q_est = L.solve(P, method="lstsq")
    show_blade("q_est (recovered)", q_est)

    error = np.linalg.norm(q_est.data - q_true.data)
    print(f"  Recovery error: {error:.2e}")

    subsection("With measurement noise")
    noise_level = 0.01
    P_noisy = Blade(P.data + noise_level * np.random.randn(*P.shape), grade=2, metric=euclidean(d))

    q_est_noisy = L.solve(P_noisy, method="lstsq")
    error_noisy = np.linalg.norm(q_est_noisy.data - q_true.data)
    print(f"  Noise level: {noise_level}")
    print(f"  Recovery error with noise: {error_noisy:.4f}")

    subsection("Tikhonov regularization for stability")
    q_reg = L.solve(P_noisy, method="lstsq", alpha=1e-3)
    error_reg = np.linalg.norm(q_reg.data - q_true.data)
    print(f"  With alpha=1e-3: {error_reg:.4f}")
    print("  Regularization trades bias for variance reduction")


# =============================================================================
# Section 5: SVD Decomposition
# =============================================================================


def demo_svd() -> None:
    """Demonstrate SVD for operator analysis."""
    section("5. SVD DECOMPOSITION")

    d, M, N = 3, 10, 5
    np.random.seed(42)
    L_data = np.random.randn(d, d, M, N)

    L = Operator(
        data=L_data,
        input_spec=BladeSpec(grade=0, collection=1, dim=d),
        output_spec=BladeSpec(grade=2, collection=1, dim=d),
        metric=euclidean(d),
    )

    subsection("Compute SVD: L = U * diag(S) * Vt")
    U, S, Vt = L.svd()
    print(f"  U maps: {U.input_shape} -> {U.output_shape}")
    print(f"  Vt maps: {Vt.input_shape} -> {Vt.output_shape}")
    print(f"  S has {len(S)} singular values")

    subsection("Singular values (sorted descending)")
    show_array("S", S)
    print(f"  Condition number: {S[0] / S[-1]:.2f}")

    subsection("Verify reconstruction")
    q = Blade(np.random.randn(N), grade=0, metric=euclidean(d))

    # Original: P = L * q
    P_orig = L * q

    # Via SVD: P = U * (S * (Vt * q))
    vt_q = Vt * q
    s_vt_q = Blade(S * vt_q.data, grade=0, metric=euclidean(d))
    P_svd = U * s_vt_q

    recon_error = np.max(np.abs(P_orig.data - P_svd.data))
    print(f"  Max |P_orig - P_svd|: {recon_error:.2e}")

    subsection("Low-rank approximation")
    print("  Truncate to top k singular values for compression")
    print("  Useful for noise filtering or dimensionality reduction")


# =============================================================================
# Section 6: Pseudoinverse
# =============================================================================


def demo_pseudoinverse() -> None:
    """Demonstrate Moore-Penrose pseudoinverse."""
    section("6. PSEUDOINVERSE")

    d, M, N = 3, 10, 5
    np.random.seed(42)
    L_data = np.random.randn(d, d, M, N)

    L = Operator(
        data=L_data,
        input_spec=BladeSpec(grade=0, collection=1, dim=d),
        output_spec=BladeSpec(grade=2, collection=1, dim=d),
        metric=euclidean(d),
    )

    subsection("Compute pseudoinverse")
    L_pinv = L.pinv()
    print(f"  L maps: {L.input_shape} -> {L.output_shape}")
    print(f"  L^+ maps: {L_pinv.input_shape} -> {L_pinv.output_shape}")

    subsection("Pseudoinverse identity: L * L^+ * L = L")
    q = Blade(np.random.randn(N), grade=0, metric=euclidean(d))

    Lq = L * q
    LpLq = L_pinv * Lq
    LLpLq = L * LpLq

    identity_error = np.max(np.abs(Lq.data - LLpLq.data))
    print(f"  Max |Lq - L * L^+ * Lq|: {identity_error:.2e}")

    subsection("Solve via pseudoinverse")
    q_pinv = L.solve(Lq, method="pinv")
    pinv_error = np.linalg.norm(q_pinv.data - q.data)
    print(f"  Recovery error: {pinv_error:.2e}")


# =============================================================================
# Section 7: Complex Operators (Phasor Systems)
# =============================================================================


def demo_complex_operators() -> None:
    """Demonstrate complex-valued operators for phasor systems."""
    section("7. COMPLEX OPERATORS (PHASOR SYSTEMS)")

    print("For frequency-domain analysis, sources and fields are phasors:")
    print("  q(t) = Re[q_tilde * exp(-i*omega*t)]")
    print("  P(t) = Re[P_tilde * exp(-i*omega*t)]")
    print()
    print("The transfer operator L can include frequency-dependent phase shifts.")

    d, M, N = 3, 10, 5
    np.random.seed(42)

    subsection("Create complex transfer operator")
    # Complex operator with magnitude and phase
    L_mag = np.random.randn(d, d, M, N)
    L_phase = np.random.randn(d, d, M, N) * 0.5
    L_data = L_mag * exp(1j * L_phase)

    L = Operator(
        data=L_data,
        input_spec=BladeSpec(grade=0, collection=1, dim=d),
        output_spec=BladeSpec(grade=2, collection=1, dim=d),
        metric=euclidean(d),
    )
    print(f"  L.data.dtype = {L.data.dtype}")

    subsection("Complex source phasors")
    q_mag = np.array([1.0, 0.5, 0.3, 0.8, 0.2])
    q_phase = np.array([0, pi / 4, pi / 2, -pi / 4, pi / 3])
    q_tilde = Blade(q_mag * exp(1j * q_phase), grade=0, metric=euclidean(d))
    show_blade("q_tilde (source phasors)", q_tilde)

    subsection("Forward application preserves complex structure")
    P_tilde = L * q_tilde
    print(f"  P_tilde.dtype = {P_tilde.data.dtype}")
    print(f"  P_tilde.shape = {P_tilde.shape}")

    subsection("Adjoint uses conjugate transpose")
    print("  For complex operators, adjoint = conjugate transpose")
    print("  This is critical for correct inner product properties")

    subsection("Least squares with complex data")
    q_recovered = L.solve(P_tilde, method="lstsq")
    recovery_error = np.linalg.norm(q_recovered.data - q_tilde.data)
    print(f"  Recovery error: {recovery_error:.2e}")


# =============================================================================
# Section 8: Vector-to-Vector Operators
# =============================================================================


def demo_vector_operator() -> None:
    """Demonstrate vector-to-vector operator (like rotation/transformation)."""
    section("8. VECTOR-TO-VECTOR OPERATORS")

    d, M, N = 3, 4, 5
    np.random.seed(42)

    subsection("Create vector-to-vector operator")
    # Shape: (*out_geo, *out_coll, *in_coll, *in_geo) = (d, M, N, d)
    T_data = np.random.randn(d, M, N, d)

    T = Operator(
        data=T_data,
        input_spec=BladeSpec(grade=1, collection=1, dim=d),
        output_spec=BladeSpec(grade=1, collection=1, dim=d),
        metric=euclidean(d),
    )
    print(f"  T.shape = {T.shape}")
    print(f"  T.input_shape = {T.input_shape}")
    print(f"  T.output_shape = {T.output_shape}")

    subsection("Apply to vector collection")
    v = Blade(np.random.randn(N, d), grade=1, metric=euclidean(d))
    w = T * v
    print(f"  Input v: shape {v.shape}, grade {v.grade}")
    print(f"  Output w: shape {w.shape}, grade {w.grade}")

    subsection("Index structure: w^a_m = T^{a}_{mn,b} v^b_n")
    print("  The operator contracts over input collection (n) and input geometric (b)")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all demonstrations."""
    print()
    print("=" * 70)
    print("  LINEAR OPERATORS DEMONSTRATION")
    print("  Structured linear maps between geometric algebra spaces")
    print("=" * 70)

    demo_operator_construction()
    demo_forward_application()
    demo_adjoint()
    demo_least_squares()
    demo_svd()
    demo_pseudoinverse()
    demo_complex_operators()
    demo_vector_operator()

    section("DEMONSTRATION COMPLETE")
    print()


if __name__ == "__main__":
    main()
