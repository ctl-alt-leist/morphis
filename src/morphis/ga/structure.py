"""
Geometric Algebra - Algebraic Structure

Structure constants and einsum signatures for geometric algebra operations.
Includes permutation functions, antisymmetric symbols, Levi-Civita tensors,
generalized Kronecker deltas, and einsum signature builders.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
"""

from itertools import permutations
from math import factorial
from typing import Dict, Tuple

from numpy import transpose, zeros
from numpy.typing import NDArray


INDICES = "abcdefghmnpqrstuvwxyz"


# =============================================================================
# Permutation Functions
# =============================================================================


def permutation_sign(perm: Tuple[int, ...]) -> int:
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

        T^{[m_1 ... m_k]} = (1 / k!) Σ_σ sgn(σ) T^{m_σ(1) ... m_σ(k)}

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


_ANTISYMMETRIC_SYMBOL_CACHE: Dict[Tuple[int, int], NDArray] = {}


def antisymmetric_symbol(k: int, d: int) -> NDArray:
    """
    Compute the k-index antisymmetric symbol ε^{m_1 ... m_k} in d dimensions.
    This is the structure constant of the exterior algebra:

        ε^{m_1 ... m_k} = +1  if (m_1, ..., m_k) is even permutation of distinct indices
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
    Get the Levi-Civita symbol ε^{m_1 ... m_d} for d dimensions. This is the
    fully antisymmetric symbol with d indices in d-dimensional space:

        ε^{m_1 ... m_d} = +1  for even permutations of (0, 1, ..., d - 1)
                        = -1  for odd permutations
                        =  0  if any indices repeat

    Shape is (d,) * d. This is a special case of antisymmetric_symbol(d, d).

    Returns the Levi-Civita tensor.
    """
    return antisymmetric_symbol(d, d)


_GENERALIZED_DELTA_CACHE: Dict[Tuple[int, int], NDArray] = {}


def generalized_delta(k: int, d: int) -> NDArray:
    """
    Compute the generalized Kronecker delta δ^{m_1 ... m_k}_{n_1 ... n_k} in d
    dimensions. This is the antisymmetric projection tensor:

        δ^{m_1 ... m_k}_{n_1 ... n_k} = (1 / k!) Σ_σ sgn(σ) δ^{m_1}_{n_σ(1)} ... δ^{m_k}_{n_σ(k)}

    Shape is (d,) * (2 k), where the first k indices are upper (result) and the
    last k are lower (input). Contracting with a k-tensor antisymmetrizes it:

        T^{[m_1 ... m_k]} = T^{n_1 ... n_k} δ^{m_1 ... m_k}_{n_1 ... n_k}

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


_WEDGE_SIGNATURE_CACHE: Dict[Tuple[int, ...], str] = {}


def wedge_signature(grades: Tuple[int, ...]) -> str:
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


def wedge_normalization(grades: Tuple[int, ...]) -> float:
    """
    Compute the normalization factor for wedge product.

    This is the multinomial coefficient: n! / (g₁! × g₂! × ... × gₖ!)
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


_INTERIOR_SIGNATURE_CACHE: Dict[Tuple[int, int], str] = {}


def interior_signature(j: int, k: int) -> str:
    """
    Einsum signature for interior product (left contraction) of grade j into
    grade k. Contracts j indices using the metric, result is grade (k - j).
    For j = 1, k = 2 returns "am, ...a, ...mn -> ...n".

    Returns the signature string.
    """
    key = (j, k)

    if key not in _INTERIOR_SIGNATURE_CACHE:
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

        _INTERIOR_SIGNATURE_CACHE[key] = sig

    return _INTERIOR_SIGNATURE_CACHE[key]


_COMPLEMENT_SIGNATURE_CACHE: Dict[Tuple[int, int], str] = {}


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


_NORM_SQUARED_SIGNATURE_CACHE: Dict[int, str] = {}


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
