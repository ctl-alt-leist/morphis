"""Tests for BladeSpec dataclass."""

import pytest

from morphis.algebra import BladeSpec, blade_spec


class TestBladeSpec:
    """Tests for BladeSpec dataclass."""

    def test_basic_creation(self):
        """Test basic BladeSpec creation."""
        spec = BladeSpec(grade=2, collection=1, dim=3)

        assert spec.grade == 2
        assert spec.collection == 1
        assert spec.dim == 3

    def test_geometric_shape_scalar(self):
        """Test geometric_shape for scalars (grade=0)."""
        spec = BladeSpec(grade=0, collection=1, dim=3)

        assert spec.geometric_shape == ()

    def test_geometric_shape_vector(self):
        """Test geometric_shape for vectors (grade=1)."""
        spec = BladeSpec(grade=1, collection=1, dim=3)

        assert spec.geometric_shape == (3,)

    def test_geometric_shape_bivector(self):
        """Test geometric_shape for bivectors (grade=2)."""
        spec = BladeSpec(grade=2, collection=1, dim=3)

        assert spec.geometric_shape == (3, 3)

    def test_geometric_shape_trivector(self):
        """Test geometric_shape for trivectors (grade=3)."""
        spec = BladeSpec(grade=3, collection=1, dim=4)

        assert spec.geometric_shape == (4, 4, 4)

    def test_total_axes_scalar(self):
        """Test total_axes for scalar with collection."""
        spec = BladeSpec(grade=0, collection=1, dim=3)

        assert spec.total_axes == 1

    def test_total_axes_bivector(self):
        """Test total_axes for bivector with collection."""
        spec = BladeSpec(grade=2, collection=1, dim=3)

        assert spec.total_axes == 3

    def test_total_axes_no_collection(self):
        """Test total_axes without collection dimensions."""
        spec = BladeSpec(grade=2, collection=0, dim=3)

        assert spec.total_axes == 2

    def test_total_axes_multiple_collection(self):
        """Test total_axes with multiple collection dimensions."""
        spec = BladeSpec(grade=1, collection=2, dim=3)

        assert spec.total_axes == 3

    def test_blade_shape(self):
        """Test blade_shape computation."""
        spec = BladeSpec(grade=2, collection=1, dim=3)

        shape = spec.blade_shape((10,))
        assert shape == (10, 3, 3)

    def test_blade_shape_multiple_collection(self):
        """Test blade_shape with multiple collection dimensions."""
        spec = BladeSpec(grade=1, collection=2, dim=4)

        shape = spec.blade_shape((5, 10))
        assert shape == (5, 10, 4)

    def test_blade_shape_wrong_collection(self):
        """Test blade_shape raises on wrong collection shape."""
        spec = BladeSpec(grade=2, collection=1, dim=3)

        with pytest.raises(ValueError, match="collection_shape has 2 dims"):
            spec.blade_shape((5, 10))

    def test_frozen(self):
        """Test that BladeSpec is immutable (frozen)."""
        spec = BladeSpec(grade=2, collection=1, dim=3)

        with pytest.raises(AttributeError):
            spec.grade = 1

    def test_negative_grade_raises(self):
        """Test that negative grade raises ValueError."""
        with pytest.raises(ValueError, match="grade must be non-negative"):
            BladeSpec(grade=-1, collection=1, dim=3)

    def test_negative_collection_raises(self):
        """Test that negative collection raises ValueError."""
        with pytest.raises(ValueError, match="collection must be non-negative"):
            BladeSpec(grade=1, collection=-1, dim=3)

    def test_zero_dim_raises(self):
        """Test that zero dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            BladeSpec(grade=1, collection=1, dim=0)

    def test_grade_exceeds_dim_raises(self):
        """Test that grade > dim raises ValueError."""
        with pytest.raises(ValueError, match="grade 4 cannot exceed dim 3"):
            BladeSpec(grade=4, collection=1, dim=3)


class TestBladeSpecHelper:
    """Tests for blade_spec helper function."""

    def test_default_collection(self):
        """Test that blade_spec defaults to collection=1."""
        spec = blade_spec(grade=2, dim=3)

        assert spec.collection == 1

    def test_explicit_collection(self):
        """Test blade_spec with explicit collection."""
        spec = blade_spec(grade=2, dim=3, collection=0)

        assert spec.collection == 0

    def test_returns_bladespec(self):
        """Test that blade_spec returns BladeSpec instance."""
        spec = blade_spec(grade=1, dim=3)

        assert isinstance(spec, BladeSpec)
