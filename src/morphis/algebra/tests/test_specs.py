"""Tests for VectorSpec dataclass."""

import pytest

from morphis.algebra import VectorSpec, vector_spec


class TestVectorSpec:
    """Tests for VectorSpec dataclass."""

    def test_basic_creation(self):
        """Test basic VectorSpec creation."""
        spec = VectorSpec(grade=2, lot=(10,), dim=3)

        assert spec.grade == 2
        assert spec.lot == (10,)
        assert spec.dim == 3

    def test_geo_scalar(self):
        """Test geo for scalars (grade=0)."""
        spec = VectorSpec(grade=0, lot=(10,), dim=3)

        assert spec.geo == ()

    def test_geo_vector(self):
        """Test geo for vectors (grade=1)."""
        spec = VectorSpec(grade=1, lot=(10,), dim=3)

        assert spec.geo == (3,)

    def test_geo_bivector(self):
        """Test geo for bivectors (grade=2)."""
        spec = VectorSpec(grade=2, lot=(10,), dim=3)

        assert spec.geo == (3, 3)

    def test_geo_trivector(self):
        """Test geo for trivectors (grade=3)."""
        spec = VectorSpec(grade=3, lot=(10,), dim=4)

        assert spec.geo == (4, 4, 4)

    def test_shape(self):
        """Test shape = lot + geo."""
        spec = VectorSpec(grade=2, lot=(10, 5), dim=3)

        assert spec.shape == (10, 5, 3, 3)
        assert spec.shape == spec.lot + spec.geo

    def test_total_axes_scalar(self):
        """Test total_axes for scalar with lot."""
        spec = VectorSpec(grade=0, lot=(10,), dim=3)

        assert spec.total_axes == 1

    def test_total_axes_bivector(self):
        """Test total_axes for bivector with lot."""
        spec = VectorSpec(grade=2, lot=(10,), dim=3)

        assert spec.total_axes == 3

    def test_total_axes_no_lot(self):
        """Test total_axes without lot dimensions."""
        spec = VectorSpec(grade=2, lot=(), dim=3)

        assert spec.total_axes == 2

    def test_total_axes_multiple_lot(self):
        """Test total_axes with multiple lot dimensions."""
        spec = VectorSpec(grade=1, lot=(5, 10), dim=3)

        assert spec.total_axes == 3

    def test_frozen(self):
        """Test that VectorSpec is immutable (frozen)."""
        from pydantic import ValidationError

        spec = VectorSpec(grade=2, lot=(10,), dim=3)

        with pytest.raises(ValidationError):
            spec.grade = 1

    def test_negative_grade_raises(self):
        """Test that negative grade raises ValueError."""
        with pytest.raises(ValueError, match="grade must be non-negative"):
            VectorSpec(grade=-1, lot=(10,), dim=3)

    def test_negative_lot_size_raises(self):
        """Test that negative lot dimension raises ValueError."""
        with pytest.raises(ValueError, match="lot dimension.*must be non-negative"):
            VectorSpec(grade=1, lot=(-1,), dim=3)

    def test_zero_dim_raises(self):
        """Test that zero dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            VectorSpec(grade=1, lot=(10,), dim=0)

    def test_grade_exceeds_dim_raises(self):
        """Test that grade > dim raises ValueError."""
        with pytest.raises(ValueError, match="grade 4 cannot exceed dim 3"):
            VectorSpec(grade=4, lot=(10,), dim=3)

    # Backwards compatibility tests
    def test_collection_property(self):
        """Test collection property returns len(lot)."""
        spec = VectorSpec(grade=2, lot=(10, 5), dim=3)

        assert spec.collection == 2  # len(lot)

    def test_geometric_shape_property(self):
        """Test geometric_shape is alias for geo."""
        spec = VectorSpec(grade=2, lot=(10,), dim=3)

        assert spec.geometric_shape == spec.geo
        assert spec.geometric_shape == (3, 3)


class TestVectorSpecHelper:
    """Tests for vector_spec helper function."""

    def test_default_lot(self):
        """Test that vector_spec defaults to lot=()."""
        spec = vector_spec(grade=2, dim=3)

        assert spec.lot == ()

    def test_explicit_lot(self):
        """Test vector_spec with explicit lot."""
        spec = vector_spec(grade=2, dim=3, lot=(10, 5))

        assert spec.lot == (10, 5)

    def test_returns_vectorspec(self):
        """Test that vector_spec returns VectorSpec instance."""
        spec = vector_spec(grade=1, dim=3)

        assert isinstance(spec, VectorSpec)


class TestVectorSpecLegacy:
    """Tests for backwards compatibility with legacy collection=int API."""

    def test_legacy_collection_int(self):
        """Test that collection=int still works with deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = VectorSpec(grade=2, collection=1, dim=3)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

        assert spec.collection == 1  # len(lot)
        assert len(spec.lot) == 1

    def test_legacy_vector_spec_collection(self):
        """Test that vector_spec(collection=int) works with deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = vector_spec(grade=2, dim=3, collection=2)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        assert spec.collection == 2  # len(lot)
