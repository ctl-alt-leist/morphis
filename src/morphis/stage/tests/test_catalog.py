"""
Unit tests for the catalog and written expressions.
"""

import pytest
from numpy import allclose, pi
from pydantic import ValidationError

from morphis.elements import Dot, basis_vector, metric
from morphis.stage import Stage
from morphis.stage.catalog import evaluate, sections


class TestCatalog:
    def test_sections(self):
        listed = sections()
        assert listed["Elements"] == ["Vector", "Bivector", "Frame", "Expression"]
        assert listed["Groups"] == ["Group"]

    def test_create_names_and_records_its_source(self):
        stage = Stage(metric(space=4))
        first, second = stage.create("Vector"), stage.create("Vector")
        assert (first.name, second.name) == ("vector", "vector 2")
        assert stage.journal[-1].source == "Vector"
        assert "motion" in first.dials

    def test_create_a_group(self):
        stage = Stage(metric(space=3))
        group = stage.create("Group")
        assert group.kind == "Group"

    def test_vector_is_spatial(self):
        stage = Stage(metric(space=3, projective=1))
        vector = stage.create("Vector")
        assert allclose(vector.object.data, basis_vector(1, stage.metric).data)

    def test_frame_is_filled(self):
        stage = Stage(metric(space=4))
        assert stage.create("Frame").options == {"filled": True}


class TestExpressions:
    def test_basis_and_blades(self):
        g = metric(space=4)
        e1, e3 = basis_vector(1, g), basis_vector(3, g)
        assert allclose(evaluate("e1 ^ e3", g).data, (e1 ^ e3).data)
        assert allclose(evaluate("e13", g).data, (e1 ^ e3).data)
        assert allclose(evaluate("e31", g).data, -(e1 ^ e3).data)
        assert allclose(evaluate("cos(pi / 3) * e1", g).data, 0.5 * e1.data)

    def test_kahler_partner(self):
        g = metric(modes=2)
        assert allclose(evaluate("e1dot", g).data, basis_vector(Dot(1), g).data)

    def test_refused(self):
        g = metric(space=3)
        with pytest.raises(ValueError, match="cannot read"):
            evaluate("e1 +", g)
        with pytest.raises(ValueError, match="does not give"):
            evaluate("2.0", g)
        with pytest.raises(ValueError, match="cannot read"):
            evaluate("__import__('os')", g)

    def test_edited_on_the_stage(self):
        stage = Stage(metric(space=4))
        written = stage.create("Expression")
        assert written.kind == "Vector"
        stage.set(written, "shape.text", "e12 + e34")
        assert written.object.grade == 2
        count = len(stage.journal)
        with pytest.raises(ValidationError):
            stage.set(written, "shape.text", "e1 ^")
        assert len(stage.journal) == count

    def test_expression_turns(self):
        stage = Stage(metric(space=3))
        written = stage.create("Expression")
        stage.set(written, "motion.rate.1,2", pi / 2)
        stage.play()
        stage.advance(1.0)
        assert allclose(written.object.data, [0, 1, 0])


class TestOperations:
    def test_listed_where_they_apply(self):
        assert "Boost" not in sections(metric(space=3))["Operations"]
        assert "Boost" in sections(metric(time=1, space=3))["Operations"]
        with pytest.raises(ValueError):
            Stage(metric(space=3)).create("Boost")

    def test_start_at_rest(self):
        stage = Stage(metric(space=3))
        turn = stage.create("Rotation", "turn")
        assert turn.kind == "Rotation"
        assert not any(turn.dials["motion"].rate.values())
        assert not turn.on
        assert not any(Stage(metric(space=3)).create("Translation").dials["motion"].velocity.values())


class TestSurfaces:
    def test_listed_outside_pga(self):
        assert sections(metric(space=4))["Surfaces"] == ["Sphere", "Torus", "Plane"]
        assert "Surfaces" not in sections(metric(space=3, projective=1))

    def test_clifford_torus_from_four_dimensions(self):
        torus = Stage(metric(space=4)).create("Torus")
        squares = (torus.object.vertices.data**2).sum(axis=-1)
        assert allclose(squares, 1.0**2 + 0.4**2)

    def test_ring_torus_in_three(self):
        torus = Stage(metric(space=3)).create("Torus")
        assert torus.object.vertices.data[:, 2].max() == pytest.approx(0.4, abs=1e-2)

    def test_sphere_in_spacetime_is_spatial(self):
        sphere = Stage(metric(time=1, space=3)).create("Sphere")
        assert allclose(sphere.object.vertices.data[:, 0], 0.0)
        assert allclose((sphere.object.vertices.data[:, 1:] ** 2).sum(axis=-1), 1.0)

    def test_resolution_rebuilds(self):
        stage = Stage(metric(space=4))
        torus = stage.create("Torus")
        stage.set(torus, "shape.resolution", 12)
        assert torus.object.n_vertices == 144
        assert torus.get("look.opacity") == 0.6

    def test_rest_is_kept_apart_from_the_drawn(self):
        stage = Stage(metric(space=4))
        torus = stage.create("Torus")
        stage.set(torus, "motion.rate.1,3", pi / 2)
        stage.play()
        stage.advance(1.0)
        assert torus.rest.vertices.data is not torus.object.vertices.data
        assert not allclose(torus.rest.vertices.data, torus.object.vertices.data)
