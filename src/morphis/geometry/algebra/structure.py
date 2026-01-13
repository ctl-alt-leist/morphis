"""
Geometric Algebra - Algebraic Structure

Structure constants and einsum signatures for geometric algebra operations.
Includes permutation functions, antisymmetric symbols, Levi-Civita tensors,
generalized Kronecker deltas, and einsum signature builders.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
"""

from itertools import permutations
from math import factorial

from numpy import transpose, zeros
from numpy.typing import NDArray


INDICES = "abcdefghmnpqrstuvwxyz"


# =============================================================================
# Permutation Functions
# =============================================================================


def permutation_sign(perm: tuple[int, ...]) -> int:
    """
    Compute the sign of a permutation (+1 for even, -1 for odd). Uses the
    cycle-counting algorithm: count transpositions needed to sort.

    Returns +1 or -1.
    """
    perm = list(perm)
    n = len(perm)
    sign = 1

    for m in range(n):
        while perm[m] != m:
            target = perm[m]
            perm[m], perm[target] = perm[target], perm[m]
            sign *= -1

    return sign


def antisymmetrize(tensor: NDArray, k: int, cdim: int = 0) -> NDArray:
    """
    Antisymmetrize a tensor over its last k axes. Computes the projection onto
    the antisymmetric subspace via sum over all permutations weighted by sign:

        T^{[m_1 ... m_k]} = (1 / k!) Sigma_s sgn(s) T^{m_s(1) ... m_s(k)}

    The 1 / k! normalization is NOT applied here; caller handles normalization.

    Returns antisymmetrized tensor of same shape.
    """
    if k <= 1:
        return tensor.copy()

    ndim = tensor.ndim
    collection_axes = list(range(cdim))
    geometric_axes = list(range(cdim, ndim))
    result = zeros(tensor.shape, dtype=tensor.dtype)

    for perm in permutations(range(k)):
        sign = permutation_sign(perm)
        permuted_geo = [geometric_axes[p] for p in perm]
        new_axes = collection_axes + permuted_geo
        result = result + sign * transpose(tensor, new_axes)

    return result


# =============================================================================
# Antisymmetric Structure Constants
# =============================================================================


_ANTISYMMETRIC_SYMBOL_CACHE: dict[tuple[int, int], NDArray] = {}


def antisymmetric_symbol(k: int, d: int) -> NDArray:
    """
    Compute the k-index antisymmetric symbol eps^{m_1 ... m_k} in d dimensions.
    This is the structure constant of the exterior algebra:

        eps^{m_1 ... m_k} = +1  if (m_1, ..., m_k) is even permutation of distinct indices
                         = -1  if (m_1, ..., m_k) is odd permutation of distinct indices
                         =  0  if any indices repeat

    Shape is (d,) * k. When k = d, this is the Levi-Civita symbol.

    Returns the antisymmetric symbol tensor.
    """
    key = (k, d)

    if key not in _ANTISYMMETRIC_SYMBOL_CACHE:
        result = zeros([d] * k)

        for perm in permutations(range(d), k):
            complement = tuple(i for i in range(d) if i not in perm)
            result[perm] = permutation_sign(perm + complement)

        _ANTISYMMETRIC_SYMBOL_CACHE[key] = result

    return _ANTISYMMETRIC_SYMBOL_CACHE[key]


def levi_civita(d: int) -> NDArray:
    """
    Get the Levi-Civita symbol eps^{m_1 ... m_d} for d dimensions. This is the
    fully antisymmetric symbol with d indices in d-dimensional space:

        eps^{m_1 ... m_d} = +1  for even permutations of (0, 1, ..., d - 1)
                         = -1  for odd permutations
                        =  0  if any indices repeat

    Shape is (d,) * d. This is a special case of antisymmetric_symbol(d, d).

    Returns the Levi-Civita tensor.
    """
    return antisymmetric_symbol(d, d)


_GENERALIZED_DELTA_CACHE: dict[tuple[int, int], NDArray] = {}


def generalized_delta(k: int, d: int) -> NDArray:
    """
    Compute the generalized Kronecker delta d^{m_1 ... m_k}_{n_1 ... n_k} in d
    dimensions. This is the antisymmetric projection tensor:

        d^{m_1 ... m_k}_{n_1 ... n_k} = (1 / k!) Sigma_s sgn(s) d^{m_1}_{n_s(1)} ... d^{m_k}_{n_s(k)}

    Shape is (d,) * (2 k), where the first k indices are upper (result) and the
    last k are lower (input). Contracting with a k-tensor antisymmetrizes it:

        T^{[m_1 ... m_k]} = T^{n_1 ... n_k} d^{m_1 ... m_k}_{n_1 ... n_k}

    Warning: This tensor has d^{2 k} elements, which grows quickly. For k = 3,
    d = 4, this is 4^6 = 4096 elements. Use antisymmetrize() for large k.

    Returns the generalized Kronecker delta tensor.
    """
    key = (k, d)

    if key not in _GENERALIZED_DELTA_CACHE:
        shape = [d] * (2 * k)
        result = zeros(shape)

        for upper_indices in permutations(range(d), k):
            upper_complement = tuple(i for i in range(d) if i not in upper_indices)
            for lower_indices in permutations(range(d), k):
                if set(upper_indices) == set(lower_indices):
                    lower_complement = tuple(i for i in range(d) if i not in lower_indices)
                    upper_sign = permutation_sign(upper_indices + upper_complement)
                    lower_sign = permutation_sign(lower_indices + lower_complement)
                    sign = upper_sign * lower_sign
                    result[upper_indices + lower_indices] = sign / factorial(k)

        _GENERALIZED_DELTA_CACHE[key] = result

    return _GENERALIZED_DELTA_CACHE[key]


# =============================================================================
# Wedge Product Signatures
# =============================================================================


_WEDGE_SIGNATURE_CACHE: dict[tuple[int, ...], str] = {}


def wedge_signature(grades: tuple[int, ...]) -> str:
    """
    Einsum signature for wedge product including delta contraction.

    Combines outer product and antisymmetrization into a single einsum:
    - For grades (1, 1): "...a, ...b, cdab -> ...cd"
    - For grades (1, 2): "...a, ...bc, defabc -> ...def"
    - For grades (1, 1, 1): "...a, ...b, ...c, defabc -> ...def"

    The signature contracts blade indices with the lower indices of
    generalized_delta, yielding antisymmetrized output indices.

    Returns the cached einsum signature string.
    """
    if grades not in _WEDGE_SIGNATURE_CACHE:
        n = sum(grades)

        if n == 0:
            # All scalars: just multiply
            sig = ", ".join("..." for _ in grades) + " -> ..."
        else:
            # Allocate blade indices
            blade_indices = []
            offset = 0
            for g in grades:
                if g > 0:
                    blade_indices.append(INDICES[offset : offset + g])
                    offset += g
                else:
                    blade_indices.append("")

            all_input = INDICES[:n]
            output_indices = INDICES[n : 2 * n]

            # Build signature parts
            blade_parts = [f"...{idx}" if idx else "..." for idx in blade_indices]
            delta_part = f"{output_indices}{all_input}"

            sig = ", ".join(blade_parts) + f", {delta_part} -> ...{output_indices}"

        _WEDGE_SIGNATURE_CACHE[grades] = sig

    return _WEDGE_SIGNATURE_CACHE[grades]


def wedge_normalization(grades: tuple[int, ...]) -> float:
    """
    Compute the normalization factor for wedge product.

    This is the multinomial coefficient: n! / (g_1! * g_2! * ... * g_k!)
    where n = sum(grades).

    The factor compensates for:
    - The 1/n! in generalized_delta
    - Overcounting when antisymmetrizing already-antisymmetric inputs

    Returns the normalization factor.
    """
    n = sum(grades)
    if n == 0:
        return 1.0

    denom = 1
    for g in grades:
        if g > 0:
            denom *= factorial(g)

    return factorial(n) / denom


# =============================================================================
# Other Einsum Signature Builders
# =============================================================================


_INTERIOR_LEFT_SIGNATURE_CACHE: dict[tuple[int, int], str] = {}


def interior_left_signature(j: int, k: int) -> str:
    """
    Einsum signature for left interior product (left contraction) of grade j
    into grade k. Contracts all j indices of u with the first j indices of v.
    Result is grade (k - j).

    For j = 1, k = 2 returns "am, ...a, ...mn -> ...n".

    Returns the signature string.
    """
    key = (j, k)

    if key not in _INTERIOR_LEFT_SIGNATURE_CACHE:
        if j == 0:
            if k == 0:
                sig = "..., ... -> ..."
            else:
                sig = "..., ..." + INDICES[:k] + " -> ..." + INDICES[:k]
        else:
            u_indices = INDICES[:j]
            v_contracted = INDICES[j : 2 * j]
            v_remaining = INDICES[2 * j : 2 * j + (k - j)]
            v_indices = v_contracted + v_remaining
            metric_parts = ", ".join(f"{u_indices[m]}{v_contracted[m]}" for m in range(j))
            result_indices = v_remaining if v_remaining else ""
            sig = f"{metric_parts}, ...{u_indices}, ...{v_indices} -> ...{result_indices}"

        _INTERIOR_LEFT_SIGNATURE_CACHE[key] = sig

    return _INTERIOR_LEFT_SIGNATURE_CACHE[key]


# Alias for backwards compatibility
interior_signature = interior_left_signature


_INTERIOR_RIGHT_SIGNATURE_CACHE: dict[tuple[int, int], str] = {}


def interior_right_signature(j: int, k: int) -> str:
    """
    Einsum signature for right interior product (right contraction) of grade j
    by grade k. Contracts all k indices of v with the last k indices of u.
    Result is grade (j - k).

    For j = 2, k = 1 returns "bm, ...ab, ...m -> ...a".

    Returns the signature string.
    """
    key = (j, k)

    if key not in _INTERIOR_RIGHT_SIGNATURE_CACHE:
        if k == 0:
            if j == 0:
                sig = "..., ... -> ..."
            else:
                sig = "..." + INDICES[:j] + ", ... -> ..." + INDICES[:j]
        else:
            # u has j indices, v has k indices
            # Contract last k indices of u with all k indices of v
            u_remaining = INDICES[: j - k]
            u_contracted = INDICES[j - k : j]
            v_indices = INDICES[j : j + k]
            u_indices = u_remaining + u_contracted

            # Metric contractions pair u_contracted[i] with v_indices[i]
            metric_parts = ", ".join(f"{u_contracted[m]}{v_indices[m]}" for m in range(k))
            result_indices = u_remaining if u_remaining else ""
            sig = f"{metric_parts}, ...{u_indices}, ...{v_indices} -> ...{result_indices}"

        _INTERIOR_RIGHT_SIGNATURE_CACHE[key] = sig

    return _INTERIOR_RIGHT_SIGNATURE_CACHE[key]


_COMPLEMENT_SIGNATURE_CACHE: dict[tuple[int, int], str] = {}


def complement_signature(k: int, d: int) -> str:
    """
    Einsum signature for right complement using the Levi-Civita symbol. Maps
    grade k to grade (d - k). For k = 1, d = 4 returns "...a, abcd -> ...bcd".

    Returns the cached einsum signature string.
    """
    key = (k, d)

    if key not in _COMPLEMENT_SIGNATURE_CACHE:
        if k == 0:
            sig = "..., " + INDICES[:d] + " -> ..." + INDICES[:d]
        else:
            blade_indices = INDICES[:k]
            result_indices = INDICES[k:d]
            eps_indices = INDICES[:d]
            sig = f"...{blade_indices}, {eps_indices} -> ...{result_indices}"

        _COMPLEMENT_SIGNATURE_CACHE[key] = sig

    return _COMPLEMENT_SIGNATURE_CACHE[key]


_NORM_SQUARED_SIGNATURE_CACHE: dict[int, str] = {}


def norm_squared_signature(k: int) -> str:
    """
    Einsum signature for blade norm squared. For k = 1 returns
    "ab, ...a, ...b -> ...". For k = 2 returns "am, bn, ...ab, ...mn -> ...".

    Returns the cached einsum signature string.
    """
    if k not in _NORM_SQUARED_SIGNATURE_CACHE:
        if k == 0:
            sig = "..., ... -> ..."
        else:
            first = INDICES[:k]
            second = INDICES[k : 2 * k]
            metric_parts = ", ".join(f"{first[m]}{second[m]}" for m in range(k))
            sig = f"{metric_parts}, ...{first}, ...{second} -> ..."

        _NORM_SQUARED_SIGNATURE_CACHE[k] = sig

    return _NORM_SQUARED_SIGNATURE_CACHE[k]


# =============================================================================
# Geometric Product Signatures
# =============================================================================


_GEOMETRIC_SIGNATURE_CACHE: dict[tuple[int, int, int], str] = {}


def geometric_signature(j: int, k: int, c: int) -> str:
    """
    Einsum signature for geometric product with c contractions.

    Args:
        j: grade of first blade
        k: grade of second blade
        c: number of contractions (index pairs to contract with metric)

    Result grade r = j + k - 2c.

    The contraction pattern follows standard GA convention: contract the LAST c
    indices of the first blade with the FIRST c indices of the second blade,
    paired in reverse order. For bivector B * bivector B (j=k=c=2):
        g^{bc} g^{ad} B^{ab} B^{cd} -> scalar

    Returns the cached einsum signature string.
    """
    key = (j, k, c)

    if key not in _GEOMETRIC_SIGNATURE_CACHE:
        # Indices for first blade
        u_indices = INDICES[:j]

        # Indices for second blade
        v_indices = INDICES[j : j + k]

        # Indices that will be contracted:
        # - Last c indices of u (these are the "inner" indices adjacent to v)
        # - First c indices of v (these are the "inner" indices adjacent to u)
        # Paired in reverse order for correct geometric product sign
        u_remain = u_indices[: j - c]  # First j-c indices remain
        u_contract = u_indices[j - c :]  # Last c indices contract
        v_contract = v_indices[:c]  # First c indices contract
        v_remain = v_indices[c:]  # Last k-c indices remain

        # Result indices (in order: u remainder, then v remainder)
        result_indices = u_remain + v_remain
        r = len(result_indices)

        # Output indices (for generalized delta)
        output_indices = INDICES[j + k : j + k + r]

        # Build metric contraction parts in reverse order:
        # u_contract[-1] with v_contract[0], u_contract[-2] with v_contract[1], etc.
        if c > 0:
            metric_parts = ", ".join(f"{u_contract[c - 1 - i]}{v_contract[i]}" for i in range(c))
        else:
            metric_parts = ""

        # Build full signature
        if c == 0 and r == 0:
            # Scalar * scalar: just multiply
            sig = "..., ... -> ..."
        elif c == 0:
            # No contractions: pure wedge (r > 0)
            sig = f"...{u_indices}, ...{v_indices}, {output_indices}{result_indices} -> ...{output_indices}"
        elif r == 0:
            # Full contraction to scalar
            sig = f"{metric_parts}, ...{u_indices}, ...{v_indices} -> ..."
        else:
            # Partial contraction
            delta_indices = output_indices + result_indices
            sig = f"{metric_parts}, ...{u_indices}, ...{v_indices}, {delta_indices} -> ...{output_indices}"

        _GEOMETRIC_SIGNATURE_CACHE[key] = sig

    return _GEOMETRIC_SIGNATURE_CACHE[key]


def geometric_normalization(j: int, k: int, c: int) -> float:
    """
    Normalization factor for geometric product with c contractions.

    The factor 1/c! accounts for the c! ways to pair contracted indices.
    For the wedge part (remaining indices), the factor r!/(j-c)!(k-c)!
    is the multinomial coefficient for antisymmetrization.

    Args:
        j: grade of first blade
        k: grade of second blade
        c: number of contractions

    Returns the normalization factor.
    """
    r = j + k - 2 * c

    if c == 0:
        # Pure wedge product: multinomial coefficient
        return factorial(r) / (factorial(j) * factorial(k))

    if r == 0:
        # Full contraction to scalar: just the contraction factor
        return 1.0 / factorial(c)

    # Partial contraction: both contraction and wedge factors
    # 1/c! for contractions, r!/((j-c)!(k-c)!) for remaining wedge
    return factorial(r) / (factorial(c) * factorial(j - c) * factorial(k - c))
