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
        input_spec = VectorSpec(grade=0, collection=1, dim=3)
        output_spec = VectorSpec(grade=2, collection=1, dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: WXKn (out_geo=WX, out_coll=K, in_coll=n, in_geo=none)
        # Input: n (in_coll=n, in_geo=none)
        # Output: KWX (out_coll=K, out_geo=WX)
        assert sig == "WXKn,n->KWX"

    def test_vector_to_bivector(self):
        """Test pattern for vector to bivector."""
        input_spec = VectorSpec(grade=1, collection=1, dim=3)
        output_spec = VectorSpec(grade=2, collection=1, dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: WXKna
        # Input: na
        # Output: KWX
        assert sig == "WXKna,na->KWX"

    def test_scalar_to_scalar(self):
        """Test pattern for scalar to scalar."""
        input_spec = VectorSpec(grade=0, collection=1, dim=3)
        output_spec = VectorSpec(grade=0, collection=1, dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: Kn
        # Input: n
        # Output: K
        assert sig == "Kn,n->K"

    def test_vector_to_vector(self):
        """Test pattern for vector to vector (rotation matrix)."""
        input_spec = VectorSpec(grade=1, collection=1, dim=3)
        output_spec = VectorSpec(grade=1, collection=1, dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: WKna
        # Input: na
        # Output: KW
        assert sig == "WKna,na->KW"

    def test_bivector_to_scalar(self):
        """Test pattern for bivector to scalar."""
        input_spec = VectorSpec(grade=2, collection=1, dim=3)
        output_spec = VectorSpec(grade=0, collection=1, dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: Knab
        # Input: nab
        # Output: K
        assert sig == "Knab,nab->K"

    def test_no_collection(self):
        """Test pattern without collection dimensions."""
        input_spec = VectorSpec(grade=1, collection=0, dim=3)
        output_spec = VectorSpec(grade=2, collection=0, dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: WXa
        # Input: a
        # Output: WX
        assert sig == "WXa,a->WX"

    def test_multiple_collection(self):
        """Test pattern with multiple collection dimensions."""
        input_spec = VectorSpec(grade=0, collection=2, dim=3)
        output_spec = VectorSpec(grade=1, collection=2, dim=3)

        sig = forward_signature(input_spec, output_spec)

        # Operator: WKLno
        # Input: no
        # Output: KLW
        assert sig == "WKLno,no->KLW"

    def test_caching(self):
        """Test that signatures are cached."""
        input_spec = VectorSpec(grade=0, collection=1, dim=3)
        output_spec = VectorSpec(grade=2, collection=1, dim=3)

        sig1 = forward_signature(input_spec, output_spec)
        sig2 = forward_signature(input_spec, output_spec)

        assert sig1 is sig2  # Same object due to caching


class TestAdjointSignature:
    """Tests for adjoint_signature function."""

    def test_scalar_to_bivector_adjoint(self):
        """Test adjoint pattern for scalar->bivector (becomes bivector->scalar)."""
        input_spec = VectorSpec(grade=0, collection=1, dim=3)
        output_spec = VectorSpec(grade=2, collection=1, dim=3)

        sig = adjoint_signature(input_spec, output_spec)

        # Operator: WXKn (same as forward)
        # Adjoint input (original output): KWX
        # Adjoint output (original input): n
        assert sig == "WXKn,KWX->n"

    def test_vector_to_vector_adjoint(self):
        """Test adjoint pattern for vector->vector."""
        input_spec = VectorSpec(grade=1, collection=1, dim=3)
        output_spec = VectorSpec(grade=1, collection=1, dim=3)

        sig = adjoint_signature(input_spec, output_spec)

        # Operator: WKna
        # Adjoint input: KW
        # Adjoint output: na
        assert sig == "WKna,KW->na"


class TestOperatorShape:
    """Tests for operator_shape function."""

    def test_scalar_to_bivector_shape(self):
        """Test operator shape for scalar->bivector case."""
        input_spec = VectorSpec(grade=0, collection=1, dim=3)
        output_spec = VectorSpec(grade=2, collection=1, dim=3)

        shape = operator_shape(
            input_spec,
            output_spec,
            input_collection=(5,),
            output_collection=(10,),
        )

        # Shape: (*out_geo, *out_coll, *in_coll, *in_geo)
        # = (3, 3, 10, 5)
        assert shape == (3, 3, 10, 5)

    def test_vector_to_vector_shape(self):
        """Test operator shape for vector->vector case."""
        input_spec = VectorSpec(grade=1, collection=1, dim=3)
        output_spec = VectorSpec(grade=1, collection=1, dim=3)

        shape = operator_shape(
            input_spec,
            output_spec,
            input_collection=(5,),
            output_collection=(10,),
        )

        # Shape: (3, 10, 5, 3)
        assert shape == (3, 10, 5, 3)

    def test_wrong_input_collection_raises(self):
        """Test that wrong input collection shape raises."""
        input_spec = VectorSpec(grade=0, collection=1, dim=3)
        output_spec = VectorSpec(grade=2, collection=1, dim=3)

        with pytest.raises(ValueError, match="input_collection has 2 dims"):
            operator_shape(
                input_spec,
                output_spec,
                input_collection=(5, 3),  # Wrong: should be 1 dim
                output_collection=(10,),
            )

    def test_wrong_output_collection_raises(self):
        """Test that wrong output collection shape raises."""
        input_spec = VectorSpec(grade=0, collection=1, dim=3)
        output_spec = VectorSpec(grade=2, collection=1, dim=3)

        with pytest.raises(ValueError, match="output_collection has 0 dims"):
            operator_shape(
                input_spec,
                output_spec,
                input_collection=(5,),
                output_collection=(),  # Wrong: should be 1 dim
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
        assert len(OUTPUT_COLLECTION) >= 4, "Need at least 4 collection dims"
        assert len(INPUT_COLLECTION) >= 4, "Need at least 4 collection dims"
        assert len(INPUT_GEOMETRIC) >= 4, "Need at least grade-4 support"
