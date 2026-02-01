"""Tests for einsum pattern generation."""

import pytest

from morphis.algebra import (
    INPUT_COLLECTION,
    INPUT_GEOMETRIC,
    OUTPUT_COLLECTION,
    OUTPUT_GEOMETRIC,
    VectorSpec,
    adjoint_signature,
    forward_signature,
    operator_shape,
)


class TestForwardSignature:
    """Tests for forward_signature function."""

    def test_scalar_to_bivector(self):
        """Test pattern for scalar currents to bivector fields."""
        input_spec = VectorSpec(grade=0, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=2, lot=(1,), dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Layout: (*out_lot, *in_lot, *out_geo, *in_geo)
        # Operator: KnWX (out_lot=K, in_lot=n, out_geo=WX, in_geo=none)
        # Input: n (in_lot=n, in_geo=none)
        # Output: KWX (out_lot=K, out_geo=WX)
        assert sig == "KnWX,n->KWX"

    def test_vector_to_bivector(self):
        """Test pattern for vector to bivector."""
        input_spec = VectorSpec(grade=1, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=2, lot=(1,), dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: KnWXa (out_lot=K, in_lot=n, out_geo=WX, in_geo=a)
        # Input: na
        # Output: KWX
        assert sig == "KnWXa,na->KWX"

    def test_scalar_to_scalar(self):
        """Test pattern for scalar to scalar."""
        input_spec = VectorSpec(grade=0, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=0, lot=(1,), dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: Kn (out_lot=K, in_lot=n)
        # Input: n
        # Output: K
        assert sig == "Kn,n->K"

    def test_vector_to_vector(self):
        """Test pattern for vector to vector (rotation matrix)."""
        input_spec = VectorSpec(grade=1, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=1, lot=(1,), dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: KnWa (out_lot=K, in_lot=n, out_geo=W, in_geo=a)
        # Input: na
        # Output: KW
        assert sig == "KnWa,na->KW"

    def test_bivector_to_scalar(self):
        """Test pattern for bivector to scalar."""
        input_spec = VectorSpec(grade=2, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=0, lot=(1,), dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: Knab (out_lot=K, in_lot=n, out_geo=none, in_geo=ab)
        # Input: nab
        # Output: K
        assert sig == "Knab,nab->K"

    def test_no_lot(self):
        """Test pattern without lot dimensions."""
        input_spec = VectorSpec(grade=1, lot=(), dim=3)
        output_spec = VectorSpec(grade=2, lot=(), dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: WXa (no lot, out_geo=WX, in_geo=a)
        # Input: a
        # Output: WX
        assert sig == "WXa,a->WX"

    def test_multiple_lot(self):
        """Test pattern with multiple lot dimensions."""
        input_spec = VectorSpec(grade=0, lot=(1, 1), dim=3)
        output_spec = VectorSpec(grade=1, lot=(1, 1), dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: KLnoW (out_lot=KL, in_lot=no, out_geo=W, in_geo=none)
        # Input: no
        # Output: KLW
        assert sig == "KLnoW,no->KLW"

    def test_caching(self):
        """Test that signatures are cached."""
        input_spec = VectorSpec(grade=0, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=2, lot=(1,), dim=3)

        sig1 = forward_signature(input_spec, output_spec)
        sig2 = forward_signature(input_spec, output_spec)

        assert sig1 is sig2  # Same object due to caching


class TestAdjointSignature:
    """Tests for adjoint_signature function."""

    def test_scalar_to_bivector_adjoint(self):
        """Test adjoint pattern for scalar->bivector (becomes bivector->scalar)."""
        input_spec = VectorSpec(grade=0, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=2, lot=(1,), dim=3)

        sig = adjoint_signature(input_spec, output_spec)

        # Operator: KnWX (same as forward, lot-first)
        # Adjoint input (original output): KWX
        # Adjoint output (original input): n
        assert sig == "KnWX,KWX->n"

    def test_vector_to_vector_adjoint(self):
        """Test adjoint pattern for vector->vector."""
        input_spec = VectorSpec(grade=1, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=1, lot=(1,), dim=3)

        sig = adjoint_signature(input_spec, output_spec)

        # Operator: KnWa (lot-first)
        # Adjoint input: KW
        # Adjoint output: na
        assert sig == "KnWa,KW->na"


class TestOperatorShape:
    """Tests for operator_shape function."""

    def test_scalar_to_bivector_shape(self):
        """Test operator shape for scalar->bivector case."""
        input_spec = VectorSpec(grade=0, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=2, lot=(1,), dim=3)

        shape = operator_shape(
            input_spec,
            output_spec,
            input_lot=(5,),
            output_lot=(10,),
        )

        # Shape: (*out_lot, *in_lot, *out_geo, *in_geo)
        # = (10, 5, 3, 3)
        assert shape == (10, 5, 3, 3)

    def test_vector_to_vector_shape(self):
        """Test operator shape for vector->vector case."""
        input_spec = VectorSpec(grade=1, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=1, lot=(1,), dim=3)

        shape = operator_shape(
            input_spec,
            output_spec,
            input_lot=(5,),
            output_lot=(10,),
        )

        # Shape: (10, 5, 3, 3)
        assert shape == (10, 5, 3, 3)

    def test_wrong_input_lot_raises(self):
        """Test that wrong input lot shape raises."""
        input_spec = VectorSpec(grade=0, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=2, lot=(1,), dim=3)

        with pytest.raises(ValueError, match="input_lot has 2 dims"):
            operator_shape(
                input_spec,
                output_spec,
                input_lot=(5, 3),  # Wrong: should be 1 dim
                output_lot=(10,),
            )

    def test_wrong_output_lot_raises(self):
        """Test that wrong output lot shape raises."""
        input_spec = VectorSpec(grade=0, lot=(1,), dim=3)
        output_spec = VectorSpec(grade=2, lot=(1,), dim=3)

        with pytest.raises(ValueError, match="output_lot has 0 dims"):
            operator_shape(
                input_spec,
                output_spec,
                input_lot=(5,),
                output_lot=(),  # Wrong: should be 1 dim
            )


class TestIndexPools:
    """Tests for index pool constants."""

    def test_pools_are_disjoint(self):
        """Test that all index pools are disjoint."""
        all_indices = set(OUTPUT_GEOMETRIC) | set(OUTPUT_COLLECTION) | set(INPUT_COLLECTION) | set(INPUT_GEOMETRIC)

        total_length = len(OUTPUT_GEOMETRIC) + len(OUTPUT_COLLECTION) + len(INPUT_COLLECTION) + len(INPUT_GEOMETRIC)

        assert len(all_indices) == total_length, "Index pools have overlapping characters"

    def test_pool_sizes(self):
        """Test that index pools have expected sizes."""
        assert len(OUTPUT_GEOMETRIC) >= 4, "Need at least grade-4 support"
        assert len(OUTPUT_COLLECTION) >= 4, "Need at least 4 lot dims"
        assert len(INPUT_COLLECTION) >= 4, "Need at least 4 lot dims"
        assert len(INPUT_GEOMETRIC) >= 4, "Need at least grade-4 support"
