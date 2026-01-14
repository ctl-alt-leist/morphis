"""Unit tests for context management system."""

from numpy import array

from morphis.ga.context import (
    CGA,
    PGA,
    STA,
    GeometricContext,
    Signature,
    Structure,
    degenerate,
    euclidean,
    lorentzian,
)
from morphis.ga.model import vector_blade
from morphis.ga.operations import wedge


# =============================================================================
# Signature Enum
# =============================================================================


class TestSignature:
    def test_from_tuple_euclidean(self):
        assert Signature.from_tuple((1, 1, 1)) == Signature.EUCLIDEAN

    def test_from_tuple_lorentzian(self):
        assert Signature.from_tuple((1, -1, -1, -1)) == Signature.LORENTZIAN
        assert Signature.from_tuple((-1, 1, 1, 1)) == Signature.LORENTZIAN

    def test_from_tuple_degenerate(self):
        assert Signature.from_tuple((0, 1, 1, 1)) == Signature.DEGENERATE
        assert Signature.from_tuple((1, 0, 1)) == Signature.DEGENERATE


# =============================================================================
# Structure Enum
# =============================================================================


class TestStructure:
    def test_structure_values(self):
        assert Structure.FLAT is not None
        assert Structure.PROJECTIVE is not None
        assert Structure.CONFORMAL is not None
        assert Structure.ROUND is not None


# =============================================================================
# GeometricContext
# =============================================================================


class TestGeometricContext:
    def test_creation(self):
        ctx = GeometricContext(Signature.EUCLIDEAN, Structure.FLAT)
        assert ctx.signature == Signature.EUCLIDEAN
        assert ctx.structure == Structure.FLAT

    def test_repr(self):
        ctx = GeometricContext(Signature.DEGENERATE, Structure.PROJECTIVE)
        assert repr(ctx) == "degenerate.projective"

    def test_is_compatible_same(self):
        ctx1 = euclidean.flat
        ctx2 = euclidean.flat
        assert ctx1.is_compatible(ctx2)

    def test_is_compatible_different_signature(self):
        ctx1 = euclidean.flat
        ctx2 = lorentzian.flat
        assert not ctx1.is_compatible(ctx2)

    def test_is_compatible_different_structure(self):
        ctx1 = euclidean.flat
        ctx2 = euclidean.projective
        assert not ctx1.is_compatible(ctx2)

    def test_merge_matching(self):
        ctx1 = euclidean.flat
        ctx2 = euclidean.flat
        assert GeometricContext.merge(ctx1, ctx2) == ctx1

    def test_merge_different_returns_none(self):
        ctx1 = euclidean.flat
        ctx2 = degenerate.projective
        assert GeometricContext.merge(ctx1, ctx2) is None

    def test_merge_with_none(self):
        ctx1 = euclidean.flat
        assert GeometricContext.merge(ctx1, None) == ctx1
        assert GeometricContext.merge(None, ctx1) == ctx1

    def test_merge_all_none(self):
        assert GeometricContext.merge(None, None) is None

    def test_merge_multiple_matching(self):
        ctx = euclidean.flat
        assert GeometricContext.merge(ctx, ctx, ctx) == ctx

    def test_merge_multiple_with_mismatch(self):
        ctx1 = euclidean.flat
        ctx2 = euclidean.flat
        ctx3 = lorentzian.flat
        assert GeometricContext.merge(ctx1, ctx2, ctx3) is None


# =============================================================================
# Context Namespaces
# =============================================================================


class TestContextNamespaces:
    def test_euclidean_flat(self):
        ctx = euclidean.flat
        assert ctx.signature == Signature.EUCLIDEAN
        assert ctx.structure == Structure.FLAT

    def test_euclidean_conformal(self):
        ctx = euclidean.conformal
        assert ctx.signature == Signature.EUCLIDEAN
        assert ctx.structure == Structure.CONFORMAL

    def test_degenerate_projective(self):
        ctx = degenerate.projective
        assert ctx.signature == Signature.DEGENERATE
        assert ctx.structure == Structure.PROJECTIVE

    def test_lorentzian_flat(self):
        ctx = lorentzian.flat
        assert ctx.signature == Signature.LORENTZIAN
        assert ctx.structure == Structure.FLAT

    def test_aliases(self):
        assert PGA == degenerate.projective
        assert CGA == euclidean.conformal
        assert STA == lorentzian.flat


# =============================================================================
# Blade Context
# =============================================================================


class TestBladeContext:
    def test_blade_default_no_context(self):
        v = vector_blade(array([1.0, 0.0, 0.0]))
        assert v.context is None

    def test_blade_with_context(self):
        v = vector_blade(array([1.0, 0.0, 0.0]))
        v_ctx = v.with_context(euclidean.flat)
        assert v_ctx.context == euclidean.flat
        assert v.context is None  # Original unchanged

    def test_add_preserves_matching_context(self):
        v1 = vector_blade(array([1.0, 0.0, 0.0])).with_context(euclidean.flat)
        v2 = vector_blade(array([0.0, 1.0, 0.0])).with_context(euclidean.flat)
        result = v1 + v2
        assert result.context == euclidean.flat

    def test_add_clears_mismatched_context(self):
        v1 = vector_blade(array([1.0, 0.0, 0.0])).with_context(euclidean.flat)
        v2 = vector_blade(array([0.0, 1.0, 0.0])).with_context(lorentzian.flat)
        result = v1 + v2
        assert result.context is None

    def test_add_with_none_context(self):
        v1 = vector_blade(array([1.0, 0.0, 0.0])).with_context(euclidean.flat)
        v2 = vector_blade(array([0.0, 1.0, 0.0]))
        result = v1 + v2
        assert result.context == euclidean.flat

    def test_sub_preserves_context(self):
        v1 = vector_blade(array([1.0, 0.0, 0.0])).with_context(euclidean.flat)
        v2 = vector_blade(array([0.0, 1.0, 0.0])).with_context(euclidean.flat)
        result = v1 - v2
        assert result.context == euclidean.flat

    def test_mul_preserves_context(self):
        v = vector_blade(array([1.0, 0.0, 0.0])).with_context(euclidean.flat)
        result = v * 2.0
        assert result.context == euclidean.flat

    def test_rmul_preserves_context(self):
        v = vector_blade(array([1.0, 0.0, 0.0])).with_context(euclidean.flat)
        result = 2.0 * v
        assert result.context == euclidean.flat

    def test_div_preserves_context(self):
        v = vector_blade(array([2.0, 0.0, 0.0])).with_context(euclidean.flat)
        result = v / 2.0
        assert result.context == euclidean.flat

    def test_neg_preserves_context(self):
        v = vector_blade(array([1.0, 0.0, 0.0])).with_context(euclidean.flat)
        result = -v
        assert result.context == euclidean.flat

    def test_repr_with_context(self):
        v = vector_blade(array([1.0, 0.0, 0.0])).with_context(euclidean.flat)
        repr_str = repr(v)
        assert "euclidean.flat" in repr_str

    def test_repr_without_context(self):
        v = vector_blade(array([1.0, 0.0, 0.0]))
        repr_str = repr(v)
        assert "context" not in repr_str


# =============================================================================
# Operations Context
# =============================================================================


class TestOperationsContext:
    def test_wedge_preserves_matching_context(self):
        v1 = vector_blade(array([1.0, 0.0, 0.0])).with_context(euclidean.flat)
        v2 = vector_blade(array([0.0, 1.0, 0.0])).with_context(euclidean.flat)
        result = wedge(v1, v2)
        assert result.context == euclidean.flat

    def test_wedge_clears_mismatched_context(self):
        v1 = vector_blade(array([1.0, 0.0, 0.0])).with_context(euclidean.flat)
        v2 = vector_blade(array([0.0, 1.0, 0.0])).with_context(lorentzian.flat)
        result = wedge(v1, v2)
        assert result.context is None

    def test_wedge_single_blade_preserves_context(self):
        v = vector_blade(array([1.0, 0.0, 0.0])).with_context(euclidean.flat)
        result = wedge(v)
        assert result.context == euclidean.flat

    def test_wedge_multiple_matching_context(self):
        ctx = euclidean.flat
        v1 = vector_blade(array([1.0, 0.0, 0.0])).with_context(ctx)
        v2 = vector_blade(array([0.0, 1.0, 0.0])).with_context(ctx)
        v3 = vector_blade(array([0.0, 0.0, 1.0])).with_context(ctx)
        result = wedge(v1, v2, v3)
        assert result.context == ctx


# =============================================================================
# Projective Context
# =============================================================================


class TestProjectiveContext:
    def test_point_sets_pga_context(self):
        from morphis.geometry.projective import point

        p = point(array([1.0, 2.0, 3.0]))
        assert p.context == degenerate.projective

    def test_direction_sets_pga_context(self):
        from morphis.geometry.projective import direction

        d = direction(array([1.0, 0.0, 0.0]))
        assert d.context == degenerate.projective

    def test_line_preserves_pga_context(self):
        from morphis.geometry.projective import line, point

        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        ln = line(p, q)
        assert ln.context == degenerate.projective

    def test_plane_preserves_pga_context(self):
        from morphis.geometry.projective import plane, point

        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        r = point(array([0.0, 1.0, 0.0]))
        pl = plane(p, q, r)
        assert pl.context == degenerate.projective


# =============================================================================
# Context Hashability
# =============================================================================


class TestContextHashability:
    def test_context_hashable(self):
        ctx = euclidean.flat
        d = {ctx: "value"}
        assert d[ctx] == "value"

    def test_context_in_set(self):
        ctx1 = euclidean.flat
        ctx2 = euclidean.flat
        s = {ctx1, ctx2}
        assert len(s) == 1

    def test_different_contexts_different_hash(self):
        ctx1 = euclidean.flat
        ctx2 = degenerate.projective
        assert hash(ctx1) != hash(ctx2)
