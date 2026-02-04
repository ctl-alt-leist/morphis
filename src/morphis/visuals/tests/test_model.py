"""Unit tests for VisualModel class and apply_similarity method."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from morphis.elements import Vector, euclidean_metric
from morphis.elements.multivector import MultiVector
from morphis.operations import wedge
from morphis.transforms import align_vectors, point_alignment, rotor
from morphis.visuals.model import VisualModel


# =============================================================================
# Test Fixtures
# =============================================================================


def make_vector(coords):
    """Create a grade-1 vector with Euclidean metric."""
    g = euclidean_metric(len(coords))
    return Vector(np.array(coords, dtype=float), grade=1, metric=g)


def make_triangle_model():
    """Create a simple triangle model."""
    g = euclidean_metric(3)
    vertices = Vector(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        grade=1,
        metric=g,
    )
    faces = np.array([3, 0, 1, 2])
    return VisualModel(vertices=vertices, faces=faces)


# =============================================================================
# VisualModel Construction Tests
# =============================================================================


class TestVisualModelConstruction:
    def test_basic_creation(self):
        """Test creating a simple triangle model."""
        model = make_triangle_model()

        assert model.n_vertices == 3
        assert model.n_faces == 1
        assert model.dim == 3
        assert model.lot == (3,)

    def test_wrong_grade_raises(self):
        """Vertices must be grade-1."""
        g = euclidean_metric(3)
        vertices = Vector(np.zeros((3, 3, 3)), grade=2, metric=g)
        faces = np.array([3, 0, 1, 2])

        with pytest.raises(ValueError, match="grade-1"):
            VisualModel(vertices=vertices, faces=faces)

    def test_wrong_dim_raises(self):
        """Vertices must be 3D."""
        g = euclidean_metric(4)
        vertices = Vector(np.zeros((3, 4)), grade=1, metric=g)
        faces = np.array([3, 0, 1, 2])

        with pytest.raises(ValueError, match="3D"):
            VisualModel(vertices=vertices, faces=faces)

    def test_invalid_face_index_raises(self):
        """Face indices must be within vertex count."""
        g = euclidean_metric(3)
        vertices = Vector(np.zeros((3, 3)), grade=1, metric=g)
        faces = np.array([3, 0, 1, 5])  # Index 5 out of range

        with pytest.raises(ValueError, match="vertex"):
            VisualModel(vertices=vertices, faces=faces)

    def test_empty_faces(self):
        """Model can have vertices with no faces (point cloud)."""
        g = euclidean_metric(3)
        vertices = Vector(np.random.randn(10, 3), grade=1, metric=g)
        faces = np.array([], dtype=int)

        model = VisualModel(vertices=vertices, faces=faces)
        assert model.n_vertices == 10
        assert model.n_faces == 0

    def test_metric_inherited_from_vertices(self):
        """Model metric comes from vertices."""
        model = make_triangle_model()
        assert model.metric == model.vertices.metric


# =============================================================================
# VisualModel Mesh Tests
# =============================================================================


class TestVisualModelMesh:
    def test_mesh_creation(self):
        """Test PyVista mesh is created on access."""
        model = make_triangle_model()

        # Mesh should be None initially
        assert model._mesh is None

        # Access mesh
        mesh = model.mesh
        assert mesh is not None
        assert mesh.n_points == 3
        assert mesh.n_cells == 1

    def test_mesh_caching(self):
        """Mesh is cached after first access."""
        model = make_triangle_model()

        mesh1 = model.mesh
        mesh2 = model.mesh
        assert mesh1 is mesh2

    def test_sync_mesh_updates_points(self):
        """Test sync_mesh updates PyVista mesh."""
        model = make_triangle_model()

        # Access mesh to create it
        _ = model.mesh

        # Modify vertices directly
        model.vertices.data[1] = [2, 0, 0]

        # Sync mesh
        model.sync_mesh()

        # Mesh should be updated
        assert_allclose(model.mesh.points[1], [2, 0, 0])


# =============================================================================
# apply_similarity Tests (on Vector)
# =============================================================================


class TestVectorApplySimilarity:
    def test_with_rotor_only(self):
        """apply_similarity with just a rotor (no translation)."""
        g = euclidean_metric(3)
        e1 = Vector([1, 0, 0], grade=1, metric=g)
        e2 = Vector([0, 1, 0], grade=1, metric=g)
        B = wedge(e1, e2)

        # Rotate e1 by 90 degrees in xy-plane
        R = rotor(B, np.pi / 2)
        result = e1.apply_similarity(R)

        assert_allclose(result.data, [0, 1, 0], atol=1e-10)

    def test_with_rotor_and_translation(self):
        """apply_similarity with rotor and translation."""
        g = euclidean_metric(3)
        e1 = Vector([1, 0, 0], grade=1, metric=g)
        e2 = Vector([0, 1, 0], grade=1, metric=g)
        B = wedge(e1, e2)

        # Rotate e1 by 90 degrees, then translate by [0, 0, 5]
        R = rotor(B, np.pi / 2)
        t = Vector([0, 0, 5], grade=1, metric=g)
        result = e1.apply_similarity(R, t)

        # [1,0,0] rotated 90 deg -> [0,1,0], then +[0,0,5] -> [0,1,5]
        assert_allclose(result.data, [0, 1, 5], atol=1e-10)

    def test_with_similarity_versor(self):
        """apply_similarity with similarity versor from align_vectors."""
        u = make_vector([1, 0, 0])
        v = make_vector([0, 2, 0])  # Rotated 90 deg and scaled by 2

        S = align_vectors(u, v)
        result = u.apply_similarity(S)

        assert_allclose(result.data, v.data, atol=1e-10)

    def test_with_similarity_versor_and_translation(self):
        """apply_similarity with similarity versor and translation."""
        u = make_vector([1, 0, 0])
        v = make_vector([0, 2, 0])
        t = make_vector([1, 1, 1])

        S = align_vectors(u, v)
        result = u.apply_similarity(S, t)

        # u transformed to v, then + t
        expected = v.data + t.data
        assert_allclose(result.data, expected, atol=1e-10)

    def test_batch_vectors(self):
        """apply_similarity works on batched vectors."""
        g = euclidean_metric(3)
        # 5 vectors
        vertices = Vector(np.random.randn(5, 3), grade=1, metric=g)

        u = make_vector([1, 0, 0])
        v = make_vector([0, 1, 0])
        S = align_vectors(u, v)

        result = vertices.apply_similarity(S)
        assert result.lot == (5,)


# =============================================================================
# apply_similarity Tests (on MultiVector)
# =============================================================================


class TestMultiVectorApplySimilarity:
    def test_multivector_apply_similarity(self):
        """apply_similarity on MultiVector transforms all grades."""
        g = euclidean_metric(3)
        e1 = Vector([1, 0, 0], grade=1, metric=g)
        e2 = Vector([0, 1, 0], grade=1, metric=g)
        B = wedge(e1, e2)

        # Create a multivector with grade-1 and grade-2 components
        M = MultiVector(e1, B)

        # Rotate 90 degrees
        R = rotor(B, np.pi / 2)
        result = M.apply_similarity(R)

        # e1 should become e2
        grade1 = result.grade_select(1)
        assert grade1 is not None
        assert_allclose(grade1.data, [0, 1, 0], atol=1e-10)


# =============================================================================
# VisualModel.apply_similarity Tests
# =============================================================================


class TestVisualModelApplySimilarity:
    def test_apply_similarity_returns_new_model(self):
        """apply_similarity returns a new VisualModel."""
        model = make_triangle_model()
        original_vertices = model.vertices.data.copy()

        u = make_vector([1, 0, 0])
        v = make_vector([0, 1, 0])
        S = align_vectors(u, v)

        new_model = model.apply_similarity(S)

        # Original unchanged
        assert_array_equal(model.vertices.data, original_vertices)

        # New model is different object
        assert new_model is not model

    def test_apply_similarity_with_translation(self):
        """apply_similarity with similarity versor and translation."""
        model = make_triangle_model()

        # Identity rotation (scale 1)
        u = make_vector([1, 0, 0])
        S = align_vectors(u, u)
        t = make_vector([10, 0, 0])

        new_model = model.apply_similarity(S, t)

        # All vertices should be shifted by [10, 0, 0]
        expected = model.vertices.data + t.data
        assert_allclose(new_model.vertices.data, expected, atol=1e-10)

    def test_apply_similarity_with_point_alignment(self):
        """Use point_alignment to register model coordinates."""
        model = make_triangle_model()

        # Define source and target anchor points
        u1 = make_vector([0, 0, 0])
        u2 = make_vector([1, 0, 0])
        v1 = make_vector([5, 5, 0])
        v2 = make_vector([5, 6, 0])  # Rotated 90 deg from u1-u2 axis

        S, t = point_alignment(u1, u2, v1, v2)
        new_model = model.apply_similarity(S, t)

        # First vertex (at origin) should map to v1
        assert_allclose(new_model.vertices.data[0], v1.data, atol=1e-10)

        # Second vertex (at [1,0,0]) should map to v2
        assert_allclose(new_model.vertices.data[1], v2.data, atol=1e-10)

    def test_faces_preserved(self):
        """Faces are preserved through transformations."""
        model = make_triangle_model()
        original_faces = model.faces.copy()

        u = make_vector([1, 0, 0])
        v = make_vector([0, 2, 0])
        S = align_vectors(u, v)

        new_model = model.apply_similarity(S)

        assert_array_equal(new_model.faces, original_faces)


# =============================================================================
# VisualModel Factory Tests
# =============================================================================


class TestVisualModelFactory:
    def test_from_mesh(self):
        """Test creating from PyVista mesh."""
        from pyvista import Cube

        mesh = Cube()
        model = VisualModel.from_mesh(mesh)

        assert model.n_vertices == mesh.n_points
        assert model.n_faces > 0

    def test_copy(self):
        """Test deep copy."""
        original = make_triangle_model()
        copy = original.copy()

        # Modify original
        original.vertices.data[0] = [100, 100, 100]

        # Copy should be unchanged
        assert not np.allclose(copy.vertices.data[0], [100, 100, 100])

    def test_with_metric(self):
        """Test with_metric returns new model."""
        model = make_triangle_model()
        new_metric = euclidean_metric(3)

        new_model = model.with_metric(new_metric)

        assert new_model is not model
        assert new_model.metric == new_metric


# =============================================================================
# VisualModel Animation Integration Tests
# =============================================================================


class TestVisualModelAnimation:
    def test_animation_watch_model(self):
        """Animation.watch() accepts VisualModel."""
        from morphis.visuals.loop import Animation

        model = make_triangle_model()
        anim = Animation(frame_rate=60)

        obj_id = anim.watch(model, color=(0.5, 0.5, 0.5))

        assert obj_id == id(model)
        assert obj_id in anim._tracks
        assert anim._tracks[obj_id].is_model is True
        assert anim._tracks[obj_id].grade == -2

    def test_animation_track_model_properties(self):
        """AnimationTrack correctly stores model properties."""
        from morphis.visuals.loop import Animation

        model = make_triangle_model()
        anim = Animation(frame_rate=60)

        anim.watch(model, color=(1.0, 0.0, 0.0), opacity=0.8)

        track = anim._tracks[id(model)]
        assert track.is_model is True
        assert track.color == (1.0, 0.0, 0.0)
        assert track.opacity == 0.8
        assert track.target is model

    def test_animation_capture_syncs_model_mesh(self):
        """Animation.capture() syncs VisualModel mesh."""
        from morphis.visuals.loop import Animation

        model = make_triangle_model()
        anim = Animation(frame_rate=60)

        anim.watch(model)
        anim.start(live=False)

        # Modify vertices directly
        model.vertices.data[0] = [5.0, 5.0, 5.0]

        # Capture should sync mesh
        anim.capture(0.0)

        # Check snapshot contains updated vertices
        snapshot = anim._snapshots[0]
        obj_id = id(model)
        _origin, vectors, _opacity, _axes = snapshot.states[obj_id]

        # vectors should contain the updated vertex data
        assert_allclose(vectors[0], [5.0, 5.0, 5.0])

    def test_animation_model_geometry_extraction(self):
        """Animation correctly extracts geometry from VisualModel."""
        from morphis.visuals.loop import Animation

        model = make_triangle_model()
        anim = Animation(frame_rate=60)

        anim.watch(model)
        track = anim._tracks[id(model)]

        origin, vectors, _ = anim._tracked_to_geometry(track)

        # Origin should be zeros
        assert_allclose(origin, [0.0, 0.0, 0.0])

        # Vectors should be the vertices
        assert vectors.shape == (3, 3)
        assert_allclose(vectors, model.vertices.data)

    def test_animation_unwatch_model(self):
        """Animation.unwatch() removes VisualModel from tracking."""
        from morphis.visuals.loop import Animation

        model = make_triangle_model()
        anim = Animation(frame_rate=60)

        anim.watch(model)
        assert id(model) in anim._tracks

        anim.unwatch(model)
        assert id(model) not in anim._tracks
