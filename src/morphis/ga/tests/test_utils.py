"""Unit tests for ga_utils.py"""

import pytest
from numpy import array

from morphis.ga.model import vector_blade
from morphis.ga.utils import (
    broadcast_collection_shape,
    get_common_cdim,
    get_common_dim,
    same_dim,
    validate_same_dim,
)


# =============================================================================
# Blade Dimension Helpers
# =============================================================================


class TestGetCommonDim:
    def test_single_blade(self):
        u = vector_blade(array([1.0, 0.0, 0.0]))
        assert get_common_dim(u) == 3

    def test_same_dim(self):
        u = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        v = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        assert get_common_dim(u, v) == 4

    def test_different_dim_raises(self):
        u = vector_blade(array([1.0, 0.0, 0.0]))
        v = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            get_common_dim(u, v)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one blade"):
            get_common_dim()


class TestGetCommonCdim:
    def test_single_blade(self):
        u = vector_blade(array([1.0, 0.0, 0.0]))
        assert get_common_cdim(u) == 0

    def test_max_cdim(self):
        u = vector_blade(array([1.0, 0.0, 0.0]))
        v = vector_blade(array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), cdim=1)
        assert get_common_cdim(u, v) == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one blade"):
            get_common_cdim()


class TestValidateSameDim:
    def test_single_blade_ok(self):
        u = vector_blade(array([1.0, 0.0, 0.0]))
        validate_same_dim(u)  # Should not raise

    def test_same_dim_ok(self):
        u = vector_blade(array([1.0, 0.0, 0.0]))
        v = vector_blade(array([0.0, 1.0, 0.0]))
        validate_same_dim(u, v)  # Should not raise

    def test_different_dim_raises(self):
        u = vector_blade(array([1.0, 0.0, 0.0]))
        v = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            validate_same_dim(u, v)


class TestSameDimDecorator:
    def test_valid_call(self):
        @same_dim
        def add_blades(u, v):
            return u.data + v.data

        u = vector_blade(array([1.0, 0.0, 0.0]))
        v = vector_blade(array([0.0, 1.0, 0.0]))
        result = add_blades(u, v)
        assert result.shape == (3,)

    def test_invalid_call_raises(self):
        @same_dim
        def add_blades(u, v):
            return u.data + v.data

        u = vector_blade(array([1.0, 0.0, 0.0]))
        v = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            add_blades(u, v)


# =============================================================================
# Collection Shape Helpers
# =============================================================================


class TestBroadcastCollectionShape:
    def test_single_blade(self):
        u = vector_blade(array([1.0, 0.0, 0.0]))
        assert broadcast_collection_shape(u) == ()

    def test_same_shape(self):
        u = vector_blade(array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), cdim=1)
        v = vector_blade(array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]), cdim=1)
        assert broadcast_collection_shape(u, v) == (2,)

    def test_broadcast_with_1(self):
        u = vector_blade(array([1.0, 0.0, 0.0]))  # cdim=0
        v = vector_blade(array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), cdim=1)  # cdim=1
        assert broadcast_collection_shape(u, v) == (2,)

    def test_incompatible_raises(self):
        u = vector_blade(array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), cdim=1)  # shape (2,)
        v = vector_blade(array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]), cdim=1)  # shape (3,)
        with pytest.raises(ValueError, match="Cannot broadcast"):
            broadcast_collection_shape(u, v)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one blade"):
            broadcast_collection_shape()
