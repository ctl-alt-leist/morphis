"""
Blade Factorization

Factor k-blades into their constituent grade-1 blades (vectors).
Any k-blade B can be factored as B = v₁ ∧ v₂ ∧ ... ∧ vₖ where the vectors
span the k-dimensional subspace represented by B.

Note: Factorization is not unique - any k vectors spanning the same subspace
will work. These functions return ONE valid factorization.
"""

from numpy import abs as np_abs, zeros
from numpy.linalg import svd

from morphis.ga.model import Blade


def factor_bivector(b: Blade) -> tuple[Blade, Blade]:
    """
    Factor a bivector into two vectors: b = u ∧ v.

    Uses SVD to find the two vectors that span the plane defined by the bivector.

    Args:
        b: A grade-2 blade (bivector)

    Returns:
        Tuple of two grade-1 Blades (u, v) such that b = u ∧ v (up to scale)

    Raises:
        ValueError: If b is not grade 2
    """
    if b.grade != 2:
        raise ValueError(f"Expected grade 2, got {b.grade}")

    data = b.data
    dim = b.dim
    context = b.context

    # For numerical stability, use SVD on the antisymmetric matrix
    U, S, Vt = svd(data)

    # The two largest singular values correspond to the plane
    if S[0] < 1e-10:
        # Zero bivector - return zero vectors
        u_data = zeros(dim)
        v_data = zeros(dim)
    else:
        # Scale by sqrt of singular value to distribute magnitude
        scale = S[0] ** 0.5
        u_data = U[:, 0] * scale
        v_data = Vt[0, :] * scale

    u = Blade(data=u_data, grade=1, dim=dim, cdim=0, context=context)
    v = Blade(data=v_data, grade=1, dim=dim, cdim=0, context=context)

    return u, v


def factor_trivector(t: Blade) -> tuple[Blade, Blade, Blade]:
    """
    Factor a trivector into three vectors: t = u ∧ v ∧ w.

    Args:
        t: A grade-3 blade (trivector)

    Returns:
        Tuple of three grade-1 Blades (u, v, w) such that t = u ∧ v ∧ w (up to scale)

    Raises:
        ValueError: If t is not grade 3
    """
    if t.grade != 3:
        raise ValueError(f"Expected grade 3, got {t.grade}")

    data = t.data
    dim = t.dim
    context = t.context

    # For 3D trivector T^{abc} = α * ε^{abc}, the spanning vectors are
    # just scaled basis vectors
    if dim >= 3:
        alpha = data[0, 1, 2]
        if np_abs(alpha) > 1e-10:
            scale = np_abs(alpha) ** (1.0 / 3.0)
            u_data = zeros(dim)
            v_data = zeros(dim)
            w_data = zeros(dim)
            u_data[0] = scale
            v_data[1] = scale
            w_data[2] = scale

            u = Blade(data=u_data, grade=1, dim=dim, cdim=0, context=context)
            v = Blade(data=v_data, grade=1, dim=dim, cdim=0, context=context)
            w = Blade(data=w_data, grade=1, dim=dim, cdim=0, context=context)

            return u, v, w

    # Fallback: find spanning vectors from non-zero components
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                if np_abs(data[a, b, c]) > 1e-10:
                    scale = np_abs(data[a, b, c]) ** (1.0 / 3.0)
                    u_data = zeros(dim)
                    v_data = zeros(dim)
                    w_data = zeros(dim)
                    u_data[a] = scale
                    v_data[b] = scale
                    w_data[c] = scale

                    u = Blade(data=u_data, grade=1, dim=dim, cdim=0, context=context)
                    v = Blade(data=v_data, grade=1, dim=dim, cdim=0, context=context)
                    w = Blade(data=w_data, grade=1, dim=dim, cdim=0, context=context)

                    return u, v, w

    # Zero trivector
    u_data = zeros(dim)
    v_data = zeros(dim)
    w_data = zeros(dim)

    u = Blade(data=u_data, grade=1, dim=dim, cdim=0, context=context)
    v = Blade(data=v_data, grade=1, dim=dim, cdim=0, context=context)
    w = Blade(data=w_data, grade=1, dim=dim, cdim=0, context=context)

    return u, v, w


def factor_quadvector(q: Blade) -> tuple[Blade, Blade, Blade, Blade]:
    """
    Factor a quadvector (4-blade) into four vectors: q = u ∧ v ∧ w ∧ x.

    Args:
        q: A grade-4 blade (quadvector)

    Returns:
        Tuple of four grade-1 Blades (u, v, w, x) such that q = u ∧ v ∧ w ∧ x (up to scale)

    Raises:
        ValueError: If q is not grade 4
    """
    if q.grade != 4:
        raise ValueError(f"Expected grade 4, got {q.grade}")

    data = q.data
    dim = q.dim
    context = q.context

    # For 4D quadvector Q^{abcd} = α * ε^{abcd}, spanning vectors are scaled basis
    if dim >= 4:
        alpha = data[0, 1, 2, 3]
        if np_abs(alpha) > 1e-10:
            scale = np_abs(alpha) ** (1.0 / 4.0)
            u_data = zeros(dim)
            v_data = zeros(dim)
            w_data = zeros(dim)
            x_data = zeros(dim)
            u_data[0] = scale
            v_data[1] = scale
            w_data[2] = scale
            x_data[3] = scale

            u = Blade(data=u_data, grade=1, dim=dim, cdim=0, context=context)
            v = Blade(data=v_data, grade=1, dim=dim, cdim=0, context=context)
            w = Blade(data=w_data, grade=1, dim=dim, cdim=0, context=context)
            x = Blade(data=x_data, grade=1, dim=dim, cdim=0, context=context)

            return u, v, w, x

    # Fallback: find from non-zero components
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    if np_abs(data[a, b, c, d]) > 1e-10:
                        scale = np_abs(data[a, b, c, d]) ** (1.0 / 4.0)
                        u_data = zeros(dim)
                        v_data = zeros(dim)
                        w_data = zeros(dim)
                        x_data = zeros(dim)
                        u_data[a] = scale
                        v_data[b] = scale
                        w_data[c] = scale
                        x_data[d] = scale

                        u = Blade(data=u_data, grade=1, dim=dim, cdim=0, context=context)
                        v = Blade(data=v_data, grade=1, dim=dim, cdim=0, context=context)
                        w = Blade(data=w_data, grade=1, dim=dim, cdim=0, context=context)
                        x = Blade(data=x_data, grade=1, dim=dim, cdim=0, context=context)

                        return u, v, w, x

    # Zero quadvector
    u_data = zeros(dim)
    v_data = zeros(dim)
    w_data = zeros(dim)
    x_data = zeros(dim)

    u = Blade(data=u_data, grade=1, dim=dim, cdim=0, context=context)
    v = Blade(data=v_data, grade=1, dim=dim, cdim=0, context=context)
    w = Blade(data=w_data, grade=1, dim=dim, cdim=0, context=context)
    x = Blade(data=x_data, grade=1, dim=dim, cdim=0, context=context)

    return u, v, w, x


def spanning_vectors(b: Blade) -> tuple[Blade, ...]:
    """
    Factor a blade into its constituent vectors.

    For a k-blade b = v₁ ∧ v₂ ∧ ... ∧ vₖ, returns (v₁, v₂, ..., vₖ).

    Args:
        b: A blade of any grade (0-4 currently supported)

    Returns:
        Tuple of k grade-1 Blades that wedge to produce the original blade

    Raises:
        NotImplementedError: For grades > 4
    """
    if b.grade == 0:
        return ()
    elif b.grade == 1:
        # Vector: return a copy as a single-element tuple
        return (Blade(data=b.data.copy(), grade=1, dim=b.dim, cdim=b.cdim, context=b.context),)
    elif b.grade == 2:
        return factor_bivector(b)
    elif b.grade == 3:
        return factor_trivector(b)
    elif b.grade == 4:
        return factor_quadvector(b)
    else:
        raise NotImplementedError(f"Factorization not implemented for grade {b.grade}")
