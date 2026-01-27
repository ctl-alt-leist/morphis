"""Tests for the Tensor base class."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from morphis.elements.metric import euclidean_metric
from morphis.elements.tensor import Tensor


class TestTensorConstruction:
    """Tests for Tensor construction and validation."""

    def test_scalar_tensor(self):
        """A (0,0)-tensor is a scalar."""
        g = euclidean_metric(3)
        t = Tensor(data=np.array(5.0), contravariant=0, covariant=0, metric=g)
        assert t.contravariant == 0
        assert t.covariant == 0
        assert t.rank == (0, 0)
        assert t.total_rank == 0
        assert t.geometric_shape == ()
        assert t.collection == ()

    def test_vector_tensor(self):
        """A (1,0)-tensor is a vector."""
        g = euclidean_metric(3)
        t = Tensor(data=[1, 2, 3], contravariant=1, covariant=0, metric=g)
        assert t.contravariant == 1
        assert t.covariant == 0
        assert t.rank == (1, 0)
        assert t.total_rank == 1
        assert t.geometric_shape == (3,)
        assert t.contravariant_shape == (3,)
        assert t.covariant_shape == ()
        assert t.collection == ()

    def test_covector_tensor(self):
        """A (0,1)-tensor is a covector (1-form)."""
        g = euclidean_metric(3)
        t = Tensor(data=[1, 2, 3], contravariant=0, covariant=1, metric=g)
        assert t.contravariant == 0
        assert t.covariant == 1
        assert t.rank == (0, 1)
        assert t.covariant_shape == (3,)

    def test_rank_2_tensor(self):
        """A (2,0)-tensor has two contravariant indices."""
        g = euclidean_metric(3)
        data = np.zeros((3, 3))
        t = Tensor(data=data, contravariant=2, covariant=0, metric=g)
        assert t.rank == (2, 0)
        assert t.geometric_shape == (3, 3)
        assert t.contravariant_shape == (3, 3)

    def test_mixed_tensor(self):
        """A (1,1)-tensor has one up and one down index."""
        g = euclidean_metric(3)
        data = np.eye(3)
        t = Tensor(data=data, contravariant=1, covariant=1, metric=g)
        assert t.rank == (1, 1)
        assert t.total_rank == 2
        assert t.geometric_shape == (3, 3)
        assert t.contravariant_shape == (3,)
        assert t.covariant_shape == (3,)

    def test_collection_inference(self):
        """Collection is inferred from data shape."""
        g = euclidean_metric(3)
        # 5 vectors in 3D
        data = np.random.randn(5, 3)
        t = Tensor(data=data, contravariant=1, covariant=0, metric=g)
        assert t.collection == (5,)
        assert t.geometric_shape == (3,)

    def test_collection_2d(self):
        """2D collection dimensions work."""
        g = euclidean_metric(3)
        # 4x5 batch of vectors
        data = np.random.randn(4, 5, 3)
        t = Tensor(data=data, contravariant=1, covariant=0, metric=g)
        assert t.collection == (4, 5)

    def test_metric_inference(self):
        """Metric is inferred from data shape if not provided."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        t = Tensor(data=data, contravariant=1, covariant=0)
        assert t.metric.dim == 4

    def test_invalid_negative_contravariant(self):
        """Negative contravariant raises error."""
        g = euclidean_metric(3)
        with pytest.raises(ValueError, match="contravariant must be non-negative"):
            Tensor(data=[1, 2, 3], contravariant=-1, covariant=0, metric=g)

    def test_invalid_negative_covariant(self):
        """Negative covariant raises error."""
        g = euclidean_metric(3)
        with pytest.raises(ValueError, match="covariant must be non-negative"):
            Tensor(data=[1, 2, 3], contravariant=0, covariant=-1, metric=g)

    def test_shape_mismatch(self):
        """Shape inconsistent with rank raises error."""
        g = euclidean_metric(3)
        # 1D data but claiming rank (2,0)
        with pytest.raises(ValueError):
            Tensor(data=[1, 2, 3], contravariant=2, covariant=0, metric=g)


class TestTensorProperties:
    """Tests for Tensor properties."""

    def test_shape(self):
        """shape property returns full array shape."""
        g = euclidean_metric(3)
        data = np.zeros((2, 3, 3))
        t = Tensor(data=data, contravariant=2, covariant=0, metric=g)
        assert t.shape == (2, 3, 3)

    def test_ndim(self):
        """ndim property returns total dimensions."""
        g = euclidean_metric(3)
        data = np.zeros((2, 3, 3))
        t = Tensor(data=data, contravariant=2, covariant=0, metric=g)
        assert t.ndim == 3

    def test_dim(self):
        """dim property returns metric dimension."""
        g = euclidean_metric(4)
        t = Tensor(data=np.zeros(4), contravariant=1, covariant=0, metric=g)
        assert t.dim == 4


class TestTensorMethods:
    """Tests for Tensor methods."""

    def test_copy(self):
        """copy creates independent tensor."""
        g = euclidean_metric(3)
        t1 = Tensor(data=[1, 2, 3], contravariant=1, covariant=0, metric=g)
        t2 = t1.copy()

        # Modify original
        t1.data[0] = 99

        # Copy is unaffected
        assert t2.data[0] == 1

    def test_with_metric(self):
        """with_metric returns tensor with new metric."""
        m1 = euclidean_metric(3)
        m2 = euclidean_metric(3)  # Different instance
        t1 = Tensor(data=[1, 2, 3], contravariant=1, covariant=0, metric=m1)
        t2 = t1.with_metric(m2)

        assert t2.metric is m2
        assert_array_equal(t2.data, t1.data)

    def test_getitem(self):
        """Indexing works on tensor data."""
        g = euclidean_metric(3)
        t = Tensor(data=[1, 2, 3], contravariant=1, covariant=0, metric=g)
        assert t[0] == 1.0
        assert t[1] == 2.0

    def test_setitem(self):
        """Setting values works on tensor data."""
        g = euclidean_metric(3)
        t = Tensor(data=[1, 2, 3], contravariant=1, covariant=0, metric=g)
        t[0] = 99
        assert t[0] == 99.0

    def test_array_conversion(self):
        """np.asarray works on tensor."""
        g = euclidean_metric(3)
        t = Tensor(data=[1, 2, 3], contravariant=1, covariant=0, metric=g)
        arr = np.asarray(t)
        assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_repr(self):
        """repr returns informative string."""
        g = euclidean_metric(3)
        t = Tensor(data=[1, 2, 3], contravariant=1, covariant=0, metric=g)
        r = repr(t)
        assert "Tensor" in r
        assert "(1,0)" in r
        assert "dim=3" in r
