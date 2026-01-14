"""Unit tests for algebra structure module."""

from numpy import array, einsum, outer
from numpy.random import randn
from numpy.testing import assert_array_almost_equal, assert_array_equal

from morphis.operations.structure import (
    INDICES,
    antisymmetric_symbol,
    antisymmetrize,
    complement_signature,
    generalized_delta,
    interior_signature,
    levi_civita,
    norm_squared_signature,
    permutation_sign,
    wedge_normalization,
    wedge_signature,
)


# =============================================================================
# Permutation Sign
# =============================================================================


class TestPermutationSign:
    def test_identity(self):
        assert permutation_sign((0, 1, 2)) == 1

    def test_single_swap(self):
        assert permutation_sign((1, 0, 2)) == -1

    def test_cyclic(self):
        assert permutation_sign((1, 2, 0)) == 1
        assert permutation_sign((2, 0, 1)) == 1

    def test_anticyclic(self):
        assert permutation_sign((0, 2, 1)) == -1
        assert permutation_sign((2, 1, 0)) == -1

    def test_4d(self):
        assert permutation_sign((0, 1, 2, 3)) == 1
        assert permutation_sign((1, 0, 2, 3)) == -1
        assert permutation_sign((3, 2, 1, 0)) == 1


# =============================================================================
# Antisymmetrize
# =============================================================================


class TestAntisymmetrize:
    def test_2d(self):
        u = array([1.0, 0.0, 0.0])
        v = array([0.0, 1.0, 0.0])
        result = antisymmetrize(outer(u, v), k=2)
        assert_array_almost_equal(result[0, 1], -result[1, 0])
        assert_array_almost_equal(result[0, 0], 0)

    def test_3d(self):
        tensor = randn(3, 3, 3)
        anti = antisymmetrize(tensor, k=3)
        assert_array_almost_equal(anti[0, 1, 2], -anti[1, 0, 2])
        assert_array_almost_equal(anti[0, 1, 2], -anti[0, 2, 1])

    def test_diagonal_zero(self):
        tensor = randn(3, 3, 3)
        anti = antisymmetrize(tensor, k=3)
        assert_array_almost_equal(anti[0, 0, 1], 0)
        assert_array_almost_equal(anti[1, 1, 2], 0)

    def test_with_cdim(self):
        tensor = randn(5, 3, 3)
        anti = antisymmetrize(tensor, k=2, cdim=1)
        assert anti.shape == (5, 3, 3)
        for k in range(5):
            assert_array_almost_equal(anti[k, 0, 1], -anti[k, 1, 0])

    def test_scales_with_reapplication(self):
        # Antisymmetrizing an antisymmetric tensor multiplies by k!
        tensor = randn(3, 3)
        anti_1 = antisymmetrize(tensor, k=2)
        anti_2 = antisymmetrize(anti_1, k=2)
        assert_array_almost_equal(anti_1 * 2, anti_2)

    def test_k_equals_1(self):
        tensor = randn(4, 4)
        anti = antisymmetrize(tensor, k=1)
        assert_array_almost_equal(anti, tensor)


# =============================================================================
# Antisymmetric Symbol
# =============================================================================


class TestAntisymmetricSymbol:
    def test_2_of_3(self):
        # k=2 indices in d=3 dimensions
        eps = antisymmetric_symbol(2, 3)
        assert eps.shape == (3, 3)
        # For (0,1), the permutation extends to (0,1,2) which is even
        assert eps[0, 1] == 1
        assert eps[1, 0] == -1
        assert eps[0, 0] == 0

    def test_3_of_4(self):
        # k=3 indices in d=4 dimensions
        eps = antisymmetric_symbol(3, 4)
        assert eps.shape == (4, 4, 4)
        assert eps[0, 1, 2] == 1
        assert eps[0, 2, 1] == -1

    def test_equals_levi_civita_when_k_equals_d(self):
        eps_anti = antisymmetric_symbol(3, 3)
        eps_levi = levi_civita(3)
        assert_array_equal(eps_anti, eps_levi)


# =============================================================================
# Levi-Civita Tensor
# =============================================================================


class TestLeviCivita:
    def test_2d(self):
        eps = levi_civita(2)
        assert eps.shape == (2, 2)
        assert eps[0, 1] == 1
        assert eps[1, 0] == -1
        assert eps[0, 0] == 0
        assert eps[1, 1] == 0

    def test_3d(self):
        eps = levi_civita(3)
        assert eps.shape == (3, 3, 3)
        assert eps[0, 1, 2] == 1
        assert eps[0, 2, 1] == -1
        assert eps[1, 0, 2] == -1
        assert eps[2, 1, 0] == -1
        assert eps[1, 2, 0] == 1
        assert eps[0, 0, 1] == 0

    def test_4d(self):
        eps = levi_civita(4)
        assert eps.shape == (4, 4, 4, 4)
        assert eps[0, 1, 2, 3] == 1
        assert eps[0, 1, 3, 2] == -1
        assert eps[3, 2, 1, 0] == 1

    def test_caching(self):
        eps_1 = levi_civita(3)
        eps_2 = levi_civita(3)
        assert eps_1 is eps_2


# =============================================================================
# Generalized Delta
# =============================================================================


class TestGeneralizedDelta:
    def test_shape(self):
        delta = generalized_delta(2, 3)
        assert delta.shape == (3, 3, 3, 3)

    def test_antisymmetric_in_upper_indices(self):
        delta = generalized_delta(2, 3)
        assert_array_almost_equal(delta[0, 1, :, :], -delta[1, 0, :, :])

    def test_antisymmetric_in_lower_indices(self):
        delta = generalized_delta(2, 3)
        assert_array_almost_equal(delta[:, :, 0, 1], -delta[:, :, 1, 0])

    def test_identity_contraction(self):
        # Contracting generalized delta with itself should give k!
        delta = generalized_delta(2, 3)
        # δ^{ab}_{ab} = 2! * C(3,2) = 2 * 3 = 6
        trace = einsum("abab->", delta)
        assert_array_almost_equal(trace, 3.0)  # C(3,2) = 3


# =============================================================================
# Einsum Signatures
# =============================================================================


class TestWedgeSignature:
    def test_two_vectors(self):
        sig = wedge_signature((1, 1))
        assert "...a" in sig
        assert "...b" in sig
        assert "cdab" in sig  # delta indices
        assert "-> ...cd" in sig

    def test_vector_bivector(self):
        sig = wedge_signature((1, 2))
        assert "...a" in sig
        assert "...bc" in sig
        assert "defabc" in sig
        assert "-> ...def" in sig

    def test_three_vectors(self):
        sig = wedge_signature((1, 1, 1))
        assert "...a" in sig
        assert "...b" in sig
        assert "...c" in sig
        assert "defabc" in sig
        assert "-> ...def" in sig

    def test_scalar_vector(self):
        sig = wedge_signature((0, 1))
        assert "..." in sig
        assert "a" in sig

    def test_all_scalars(self):
        assert wedge_signature((0, 0)) == "..., ... -> ..."

    def test_caching(self):
        sig_1 = wedge_signature((1, 1))
        sig_2 = wedge_signature((1, 1))
        assert sig_1 is sig_2

    def test_signature_works_with_einsum(self):
        sig = wedge_signature((1, 1))
        u = array([1.0, 0.0, 0.0])
        v = array([0.0, 1.0, 0.0])
        delta = generalized_delta(2, 3)
        result = einsum(sig, u, v, delta)
        assert result.shape == (3, 3)
        # Result should be antisymmetric
        assert_array_almost_equal(result[0, 1], -result[1, 0])


class TestWedgeNormalization:
    def test_two_vectors(self):
        # n=2, grades=(1,1): 2! / (1! * 1!) = 2
        assert wedge_normalization((1, 1)) == 2.0

    def test_three_vectors(self):
        # n=3, grades=(1,1,1): 3! / (1! * 1! * 1!) = 6
        assert wedge_normalization((1, 1, 1)) == 6.0

    def test_vector_bivector(self):
        # n=3, grades=(1,2): 3! / (1! * 2!) = 3
        assert wedge_normalization((1, 2)) == 3.0

    def test_bivector_bivector(self):
        # n=4, grades=(2,2): 4! / (2! * 2!) = 6
        assert wedge_normalization((2, 2)) == 6.0

    def test_all_scalars(self):
        assert wedge_normalization((0, 0)) == 1.0

    def test_scalar_vector(self):
        # n=1, grades=(0,1): 1! / 1! = 1
        assert wedge_normalization((0, 1)) == 1.0


class TestInteriorSignature:
    def test_vector_bivector(self):
        assert interior_signature(1, 2) == "ab, ...a, ...bc -> ...c"

    def test_vector_trivector(self):
        assert interior_signature(1, 3) == "ab, ...a, ...bcd -> ...cd"

    def test_bivector_trivector(self):
        assert interior_signature(2, 3) == "ac, bd, ...ab, ...cde -> ...e"

    def test_scalar_vector(self):
        sig = interior_signature(0, 1)
        assert "a" in sig

    def test_scalar_scalar(self):
        assert interior_signature(0, 0) == "..., ... -> ..."


class TestComplementSignature:
    def test_vector_4d(self):
        assert complement_signature(1, 4) == "...a, abcd -> ...bcd"

    def test_bivector_4d(self):
        assert complement_signature(2, 4) == "...ab, abcd -> ...cd"

    def test_trivector_4d(self):
        assert complement_signature(3, 4) == "...abc, abcd -> ...d"

    def test_scalar_3d(self):
        sig = complement_signature(0, 3)
        assert "abc" in sig


class TestNormSquaredSignature:
    def test_vector(self):
        assert norm_squared_signature(1) == "ab, ...a, ...b -> ..."

    def test_bivector(self):
        assert norm_squared_signature(2) == "ac, bd, ...ab, ...cd -> ..."

    def test_trivector(self):
        sig = norm_squared_signature(3)
        assert "ad" in sig
        assert "be" in sig
        assert "cf" in sig

    def test_scalar(self):
        assert norm_squared_signature(0) == "..., ... -> ..."


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_indices_string(self):
        # Verify INDICES contains expected characters
        assert "a" in INDICES
        assert "b" in INDICES
        assert "m" in INDICES
        assert "n" in INDICES
        # Should not contain i or j
        assert "i" not in INDICES
        assert "j" not in INDICES

    def test_einsum_signature_valid(self):
        # Verify signatures work with einsum
        sig = wedge_signature((1, 1))
        u = array([1.0, 0.0, 0.0, 0.0])
        v = array([0.0, 1.0, 0.0, 0.0])
        delta = generalized_delta(2, 4)
        result = einsum(sig, u, v, delta)
        assert result.shape == (4, 4)

    def test_levi_civita_contraction(self):
        # ε^{abc} ε_{abc} = d! = 6 for d=3
        eps = levi_civita(3)
        contraction = einsum("abc,abc->", eps, eps)
        assert_array_almost_equal(contraction, 6.0)
