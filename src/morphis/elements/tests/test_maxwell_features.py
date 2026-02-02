"""
Tests for Maxwell features: Vector.stack, Vector.real/imag, and lot indexing.
"""

import numpy as np
import pytest

from morphis.elements import LotIndexed, Vector, euclidean_metric


@pytest.fixture
def g():
    """3D Euclidean metric."""
    return euclidean_metric(3)


# =============================================================================
# Vector.stack tests
# =============================================================================


class TestVectorStack:
    """Tests for Vector.stack class method."""

    def test_stack_basic(self, g):
        """Stack multiple vectors into a lot dimension."""
        vectors = [Vector(np.array([1.0, 0, 0]), grade=1, metric=g) for _ in range(10)]
        stacked = Vector.stack(vectors, axis=0)

        assert stacked.lot == (10,)
        assert stacked.grade == 1
        assert stacked.data.shape == (10, 3)

    def test_stack_preserves_data(self, g):
        """Stacked vectors have correct data."""
        v1 = Vector(np.array([1.0, 0, 0]), grade=1, metric=g)
        v2 = Vector(np.array([0.0, 1, 0]), grade=1, metric=g)
        v3 = Vector(np.array([0.0, 0, 1]), grade=1, metric=g)

        stacked = Vector.stack([v1, v2, v3], axis=0)

        np.testing.assert_array_equal(stacked.at[0].data, v1.data)
        np.testing.assert_array_equal(stacked.at[1].data, v2.data)
        np.testing.assert_array_equal(stacked.at[2].data, v3.data)

    def test_stack_empty_raises(self, g):
        """Stacking empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="Cannot stack empty sequence"):
            Vector.stack([], axis=0)

    def test_stack_grade_mismatch_raises(self, g):
        """Stacking vectors with different grades raises ValueError."""
        v1 = Vector(np.array([1.0, 0, 0]), grade=1, metric=g)
        v2 = Vector(np.array([[0.0, 1, 0], [0, 0, 0], [0, 0, 0]]), grade=2, metric=g)

        with pytest.raises(ValueError, match="Grade mismatch"):
            Vector.stack([v1, v2], axis=0)

    def test_stack_with_existing_lot(self, g):
        """Stack vectors that already have lot dimensions."""
        vectors = [Vector(np.random.randn(5, 3), grade=1, metric=g) for _ in range(3)]
        stacked = Vector.stack(vectors, axis=0)

        assert stacked.lot == (3, 5)
        assert stacked.data.shape == (3, 5, 3)

    def test_stack_axis_parameter(self, g):
        """Stack along different axes."""
        vectors = [Vector(np.random.randn(5, 3), grade=1, metric=g) for _ in range(3)]

        stacked_0 = Vector.stack(vectors, axis=0)
        assert stacked_0.lot == (3, 5)

        stacked_1 = Vector.stack(vectors, axis=1)
        assert stacked_1.lot == (5, 3)


# =============================================================================
# Vector.real / Vector.imag tests
# =============================================================================


class TestVectorRealImag:
    """Tests for Vector.real and Vector.imag properties."""

    def test_real_from_complex(self, g):
        """Extract real part from complex vector."""
        v = Vector(np.array([1 + 2j, 3 - 4j, 5j]), grade=1, metric=g)

        real_v = v.real

        np.testing.assert_array_equal(real_v.data, [1, 3, 0])
        assert real_v.data.dtype == np.float64

    def test_imag_from_complex(self, g):
        """Extract imaginary part from complex vector."""
        v = Vector(np.array([1 + 2j, 3 - 4j, 5j]), grade=1, metric=g)

        imag_v = v.imag

        np.testing.assert_array_equal(imag_v.data, [2, -4, 5])
        assert imag_v.data.dtype == np.float64

    def test_real_from_real(self, g):
        """Real part of real vector is itself."""
        v = Vector(np.array([1.0, 2, 3]), grade=1, metric=g)

        real_v = v.real

        np.testing.assert_array_equal(real_v.data, v.data)
        assert real_v is not v  # Should be a copy

    def test_imag_from_real(self, g):
        """Imaginary part of real vector is zero."""
        v = Vector(np.array([1.0, 2, 3]), grade=1, metric=g)

        imag_v = v.imag

        np.testing.assert_array_equal(imag_v.data, [0, 0, 0])

    def test_phasor_time_evolution(self, g):
        """Test time evolution pattern for phasors."""
        omega = 2 * np.pi * 60  # 60 Hz
        t = 0.001  # 1 ms

        b_phasor = Vector(np.array([1 + 0j, 0, 0]), grade=1, metric=g)
        b_instant = (b_phasor * np.exp(1j * omega * t)).real

        assert b_instant.data.dtype == np.float64
        # At t=0.001s, phase = 2*pi*60*0.001 = 0.377 rad
        expected = np.cos(omega * t)
        np.testing.assert_almost_equal(b_instant.data[0], expected)

    def test_preserves_grade_and_metric(self, g):
        """Real and imag preserve grade and metric."""
        v = Vector(np.array([1 + 2j, 3 - 4j, 5j]), grade=1, metric=g)

        assert v.real.grade == v.grade
        assert v.real.metric == v.metric
        assert v.imag.grade == v.grade
        assert v.imag.metric == v.metric

    def test_preserves_lot(self, g):
        """Real and imag preserve lot dimensions."""
        v = Vector(np.random.randn(5, 3, 3) + 1j * np.random.randn(5, 3, 3), grade=1, metric=g)

        assert v.real.lot == v.lot
        assert v.imag.lot == v.lot


# =============================================================================
# Lot Indexing tests
# =============================================================================


class TestLotIndexing:
    """Tests for lot indexing with LotIndexed wrapper."""

    def test_lot_indexing_returns_lot_indexed(self, g):
        """Bracket notation with lot-length string returns LotIndexed."""
        v = Vector(np.random.randn(5, 3), grade=1, metric=g)  # lot=(5,)

        indexed = v["m"]

        assert isinstance(indexed, LotIndexed)
        assert indexed.indices == "m"
        assert indexed.vector is v

    def test_lot_indexing_validation(self, g):
        """Index count must match lot or total dimensions."""
        v = Vector(np.random.randn(5, 3), grade=1, metric=g)  # lot=(5,), total ndim=2

        # "m" -> LotIndexed (1 char = 1 lot dim)
        assert isinstance(v["m"], LotIndexed)

        # "mn" -> IndexedTensor (2 chars = 2 total dims)
        from morphis.algebra.contraction import IndexedTensor

        assert isinstance(v["mn"], IndexedTensor)

        # "mnk" -> ValueError (3 chars, but only 2 total dims and 1 lot dim)
        with pytest.raises(ValueError):
            v["mnk"]

    def test_reordering(self, g):
        """Reorder lot dimensions via bracket notation."""
        v = Vector(np.random.randn(3, 5, 3), grade=1, metric=g)  # lot=(3, 5)
        vi = v["mn"]

        reordered = vi["nm"]

        assert reordered.indices == "nm"
        assert reordered.vector.lot == (5, 3)
        # Check data was actually transposed
        np.testing.assert_array_equal(reordered.vector.data[0, 0], v.data[0, 0])

    def test_subtraction_outer_product(self, g):
        """Subtraction with non-shared indices creates outer product."""
        x = Vector(np.random.randn(5, 3), grade=1, metric=g)  # lot=(5,)
        y = Vector(np.random.randn(3, 100, 3), grade=1, metric=g)  # lot=(3, 100)

        # y["nk"] - x["m"] should give lot (N, K, M) by default
        r = y["nk"] - x["m"]

        # Default order: left_only + right_only + shared
        # left_only = "nk", right_only = "m", shared = none
        assert r.indices == "nkm"
        assert r.vector.lot == (3, 100, 5)
        assert r.vector.grade == 1

    def test_subtraction_reorder(self, g):
        """Reorder result of subtraction."""
        x = Vector(np.random.randn(5, 3), grade=1, metric=g)  # lot=(5,)
        y = Vector(np.random.randn(3, 100, 3), grade=1, metric=g)  # lot=(3, 100)

        r = (y["nk"] - x["m"])["mnk"]

        assert r.indices == "mnk"
        assert r.vector.lot == (5, 3, 100)

    def test_wedge_broadcasting(self, g):
        """Wedge product with lot broadcasting."""
        dl = Vector(np.random.randn(3, 99, 3), grade=1, metric=g)  # lot=(3, 99)
        r = Vector(np.random.randn(5, 3, 99, 3), grade=1, metric=g)  # lot=(5, 3, 99)

        kernel = dl["nk"] ^ r["mnk"]

        # nk shared (element-wise), m is outer
        # left_only=none, right_only=m, shared=nk
        assert kernel.indices == "mnk"
        assert kernel.vector.lot == (5, 3, 99)
        assert kernel.vector.grade == 2  # wedge of two grade-1 vectors

    def test_division_broadcasting(self, g):
        """Division with lot broadcasting."""
        kernel = Vector(np.random.randn(5, 3, 99, 3, 3), grade=2, metric=g)
        norm = Vector(np.abs(np.random.randn(5, 3, 99)) + 0.1, grade=0, metric=g)

        result = kernel["mnk"] / norm["mnk"]

        # All shared -> pure element-wise
        assert result.indices == "mnk"
        assert result.vector.lot == (5, 3, 99)

    def test_contraction(self, g):
        """Multiplication contracts shared indices."""
        G = Vector(np.random.randn(5, 3, 3, 3), grade=2, metric=g)  # lot=(5, 3)
        q = Vector(np.random.randn(3, 3, 3), grade=2, metric=g)  # lot=(3,)

        b = G["mn"] * q["n"]

        # n contracts -> result lot (M,)
        assert b.indices == "m"
        assert b.vector.lot == (5,)
        assert b.vector.grade == 2

    def test_hadamard(self, g):
        """Hadamard (element-wise multiply) with &."""
        A = Vector(np.random.randn(5, 3, 3), grade=1, metric=g)  # lot=(5, 3)
        B = Vector(np.random.randn(5, 3, 3), grade=1, metric=g)  # lot=(5, 3)

        result = A["mn"] & B["mn"]

        assert result.indices == "mn"
        assert result.vector.lot == (5, 3)
        np.testing.assert_array_almost_equal(result.vector.data, A.data * B.data)

    def test_norm_preserves_indices(self, g):
        """Norm operation preserves lot indices."""
        v = Vector(np.random.randn(5, 3, 3), grade=1, metric=g)  # lot=(5, 3)
        vi = v["mn"]

        n = vi.norm()

        assert n.indices == "mn"
        assert n.vector.lot == (5, 3)
        assert n.vector.grade == 0

    def test_sum_removes_index(self, g):
        """Sum over axis removes that index."""
        v = Vector(np.random.randn(5, 3, 99, 3), grade=1, metric=g)  # lot=(5, 3, 99)
        vi = v["mnk"]

        summed = vi.sum(axis=2)

        assert summed.indices == "mn"
        assert summed.vector.lot == (5, 3)

    def test_pow_preserves_indices(self, g):
        """Power operation preserves indices."""
        n = Vector(np.abs(np.random.randn(5, 3, 99)) + 0.1, grade=0, metric=g)
        ni = n["mnk"]

        cubed = ni**3

        assert cubed.indices == "mnk"
        assert cubed.vector.lot == (5, 3, 99)


class TestLotIndexingBiotSavart:
    """Integration test mimicking Biot-Savart computation pattern."""

    def test_biot_savart_pattern(self, g):
        """Test the Biot-Savart broadcasting pattern from the spec."""
        M = 5  # sensors
        N = 3  # wires
        K = 100  # points per wire

        # Sensor positions: lot (M,)
        x = Vector(np.random.randn(M, 3), grade=1, metric=g)

        # Wire points: lot (N, K)
        y = Vector(np.random.randn(N, K, 3), grade=1, metric=g)

        # Line elements: lot (N, K-1) via slicing
        dl = y.at[:, 1:] - y.at[:, :-1]
        assert dl.lot == (N, K - 1)

        # Separation vectors with broadcasting
        r = (y.at[:, :-1]["nk"] - x["m"])["mnk"]
        assert r.vector.lot == (M, N, K - 1)
        assert r.vector.grade == 1

        # Wedge product
        kernel = dl["nk"] ^ r["mnk"]
        assert kernel.vector.lot == (M, N, K - 1)
        assert kernel.vector.grade == 2

        # Normalize (simplified - not the full 1/r^3)
        norm_vals = r.norm()
        normalized = kernel / (norm_vals**3)
        assert normalized.vector.lot == (M, N, K - 1)

        # Sum over K dimension
        G = normalized.sum(axis=2)
        assert G.indices == "mn"
        assert G.vector.lot == (M, N)
