"""Tests for Operator class."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from morphis.algebra import VectorSpec
from morphis.elements import Vector, euclidean_metric
from morphis.operations import Operator


class TestOperatorConstruction:
    """Tests for Operator construction and validation."""

    def test_basic_construction(self):
        """Test basic operator construction."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        assert op.shape == (d, d, M, N)
        assert op.input_shape == (N,)
        assert op.output_shape == (M, d, d)

    def test_vector_to_vector_construction(self):
        """Test vector->vector operator (like rotation matrix)."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, M, N, d)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=1, collection=1, dim=d),
            output_spec=VectorSpec(grade=1, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        assert op.shape == (d, M, N, d)
        assert op.input_shape == (N, d)
        assert op.output_shape == (M, d)

    def test_wrong_ndim_raises(self):
        """Test that wrong number of dimensions raises."""
        G_data = np.random.randn(3, 3, 10)  # Missing input collection dim

        with pytest.raises(ValueError, match="Data has 3 dimensions"):
            Operator(
                data=G_data,
                input_spec=VectorSpec(grade=0, collection=1, dim=3),
                output_spec=VectorSpec(grade=2, collection=1, dim=3),
                metric=euclidean_metric(3),
            )

    def test_mismatched_dims_raises(self):
        """Test that mismatched input/output dims raise."""
        G_data = np.random.randn(3, 3, 10, 5)

        with pytest.raises(ValueError, match="Input dim 4 doesn't match output dim 3"):
            Operator(
                data=G_data,
                input_spec=VectorSpec(grade=0, collection=1, dim=4),  # Wrong dim
                output_spec=VectorSpec(grade=2, collection=1, dim=3),
                metric=euclidean_metric(3),
            )

    def test_complex_data_preserved(self):
        """Test that complex data is preserved."""
        G_data = np.random.randn(3, 3, 10, 5) + 1j * np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        assert op.data.dtype == np.complex128


class TestOperatorApply:
    """Tests for Operator.apply() method."""

    def test_forward_scalar_to_bivector(self):
        """Test forward application: scalar -> bivector."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)
        G_data = (G_data - G_data.transpose(1, 0, 2, 3)) / 2  # Antisymmetrize

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        I = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))
        B = op.apply(I)

        assert B.grade == 2
        assert B.shape == (M, d, d)
        assert B.metric == euclidean_metric(d)

    def test_forward_preserves_antisymmetry(self):
        """Test that bivector output is antisymmetric."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)
        G_data = (G_data - G_data.transpose(1, 0, 2, 3)) / 2

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        I = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))
        B = op.apply(I)

        # Check antisymmetry: B[m, a, b] = -B[m, b, a]
        assert_allclose(B.data, -B.data.transpose(0, 2, 1), atol=1e-14)

    def test_forward_vector_to_vector(self):
        """Test forward application: vector -> vector (rotation-like)."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, M, N, d)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=1, collection=1, dim=d),
            output_spec=VectorSpec(grade=1, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        v = Vector(np.random.randn(N, d), grade=1, metric=euclidean_metric(d))
        w = op.apply(v)

        assert w.grade == 1
        assert w.shape == (M, d)

    def test_mul_operator(self):
        """Test L * x syntax."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        I = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))
        B1 = op.apply(I)
        B2 = op * I

        assert_allclose(B1.data, B2.data)

    def test_call_operator(self):
        """Test L(x) syntax."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        I = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))
        B1 = op.apply(I)
        B2 = op(I)

        assert_allclose(B1.data, B2.data)

    def test_wrong_grade_raises(self):
        """Test that wrong input grade raises."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        v = Vector(np.random.randn(5, 3), grade=1, metric=euclidean_metric(3))  # Wrong grade

        with pytest.raises(ValueError, match="Input grade 1 doesn't match"):
            op.apply(v)

    def test_wrong_shape_raises(self):
        """Test that wrong input shape raises."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        I = Vector(np.random.randn(7), grade=0, metric=euclidean_metric(3))  # Wrong size

        with pytest.raises(ValueError, match="Input shape"):
            op.apply(I)

    def test_complex_operator_application(self):
        """Test applying complex-valued operator."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N) + 1j * np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        # Complex input (phasor)
        I = Vector(
            np.random.randn(N) + 1j * np.random.randn(N),
            grade=0,
            metric=euclidean_metric(d),
        )
        B = op.apply(I)

        assert B.data.dtype == np.complex128


class TestOperatorAdjoint:
    """Tests for Operator.adjoint() method."""

    def test_adjoint_swaps_specs(self):
        """Test that adjoint swaps input/output specs."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        adj = op.adjoint()

        assert adj.input_spec == op.output_spec
        assert adj.output_spec == op.input_spec

    def test_adjoint_involution(self):
        """Test that adjoint of adjoint equals original."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        adj_adj = op.adjoint().adjoint()

        assert_allclose(adj_adj.data, op.data)

    def test_adjoint_complex_conjugates(self):
        """Test that adjoint conjugates complex data."""
        G_data = np.random.randn(3, 3, 10, 5) + 1j * np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        adj = op.adjoint()

        # Adjoint of adjoint should recover original
        adj_adj = adj.adjoint()
        assert_allclose(adj_adj.data, op.data)

    def test_H_property(self):
        """Test .H property is alias for adjoint."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        assert_allclose(op.H.data, op.adjoint().data)

    def test_adjoint_inner_product_property(self):
        """Test that <Lx, y> = <x, L^H y> (in flattened sense)."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        x = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))
        y = Vector(np.random.randn(M, d, d), grade=2, metric=euclidean_metric(d))

        Lx = op.apply(x)
        Lhy = op.adjoint().apply(y)

        # <Lx, y> = sum(Lx * y)
        inner1 = np.sum(Lx.data.conj() * y.data)
        # <x, L^H y> = sum(x * L^H y)
        inner2 = np.sum(x.data.conj() * Lhy.data)

        assert_allclose(inner1, inner2, rtol=1e-10)


class TestOperatorProperties:
    """Tests for Operator properties."""

    def test_dim_property(self):
        """Test dim property."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        assert op.dim == 3

    def test_input_collection(self):
        """Test input_collection property."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        assert op.input_collection == (5,)

    def test_output_collection(self):
        """Test output_collection property."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        assert op.output_collection == (10,)

    def test_repr(self):
        """Test __repr__ method."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        repr_str = repr(op)
        assert "Operator" in repr_str
        assert "shape=" in repr_str
