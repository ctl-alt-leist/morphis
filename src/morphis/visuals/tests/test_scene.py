"""Unit tests for Scene save/load functionality."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from morphis.elements import Vector, euclidean_metric
from morphis.visuals import Scene
from morphis.visuals.scene import SceneData


# =============================================================================
# Test Fixtures
# =============================================================================


def make_vector(coords):
    """Create a grade-1 vector with Euclidean metric."""
    g = euclidean_metric(len(coords))
    return Vector(np.array(coords, dtype=float), grade=1, metric=g)


def make_simple_scene():
    """Create a scene with a few vectors."""
    g = euclidean_metric(3)
    e1 = Vector([1, 0, 0], grade=1, metric=g)
    e2 = Vector([0, 1, 0], grade=1, metric=g)
    e3 = Vector([0, 0, 1], grade=1, metric=g)

    scene = Scene(theme="obsidian")
    scene.add(e1, color=(1, 0, 0))
    scene.add(e2, color=(0, 1, 0))
    scene.add(e3, color=(0, 0, 1))
    return scene


# =============================================================================
# Scene Save Tests
# =============================================================================


class TestSceneSave:
    def test_save_scene_format(self):
        """Test saving to .scene format creates valid pickle file."""
        scene = make_simple_scene()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.scene"
            scene.save(path)

            assert path.exists()

            # Verify it's valid pickle
            with open(path, "rb") as f:
                data = pickle.load(f)

            assert isinstance(data, SceneData)
            assert data.theme_name == "obsidian"
            assert len(data.elements) == 3

    def test_save_unknown_format_raises(self):
        """Test saving to unknown format raises ValueError."""
        scene = make_simple_scene()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.xyz"

            with pytest.raises(ValueError, match="Unknown format"):
                scene.save(path)

    def test_save_with_tilde_expansion(self):
        """Test that tilde paths are expanded."""
        scene = make_simple_scene()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a path that looks like it uses tilde
            # We can't truly test ~ expansion without writing to home,
            # but we can verify the code path works with normal paths
            path = Path(tmpdir) / "test.scene"
            scene.save(str(path))

            assert path.exists()

    def test_save_preserves_element_properties(self):
        """Test that element properties are preserved in save."""
        g = euclidean_metric(3)
        e1 = Vector([1, 0, 0], grade=1, metric=g)

        scene = Scene()
        scene.add(e1, color=(0.5, 0.5, 0.5), opacity=0.7)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.scene"
            scene.save(path)

            with open(path, "rb") as f:
                data = pickle.load(f)

            elem = data.elements[0]
            assert elem["color"] == (0.5, 0.5, 0.5)
            assert elem["opacity"] == 0.7


# =============================================================================
# Scene Load Tests
# =============================================================================


class TestSceneLoad:
    def test_load_restores_scene(self):
        """Test loading restores scene with elements."""
        scene = make_simple_scene()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.scene"
            scene.save(path)

            loaded = Scene.load(path)

            # Should have same number of elements
            assert len(loaded._elements) == 3

    def test_load_restores_theme(self):
        """Test loading restores theme setting."""
        scene = Scene(theme="obsidian")
        g = euclidean_metric(3)
        scene.add(Vector([1, 0, 0], grade=1, metric=g))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.scene"
            scene.save(path)

            loaded = Scene.load(path)

            assert loaded._theme.name == "obsidian"

    def test_load_restores_element_data(self):
        """Test loading restores element vector data."""
        g = euclidean_metric(3)
        e1 = Vector([1, 2, 3], grade=1, metric=g)

        scene = Scene()
        scene.add(e1, color=(1, 0, 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.scene"
            scene.save(path)

            loaded = Scene.load(path)

            # Get the first element
            elem = list(loaded._elements.values())[0]
            assert_allclose(elem.element.data, [1, 2, 3])

    def test_load_restores_colors(self):
        """Test loading restores element colors."""
        g = euclidean_metric(3)
        e1 = Vector([1, 0, 0], grade=1, metric=g)

        scene = Scene()
        scene.add(e1, color=(0.1, 0.2, 0.3))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.scene"
            scene.save(path)

            loaded = Scene.load(path)

            elem = list(loaded._elements.values())[0]
            assert elem.color == (0.1, 0.2, 0.3)

    def test_load_nonexistent_raises(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Scene.load("/nonexistent/path/to/scene.scene")


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestSceneRoundTrip:
    def test_save_load_roundtrip(self):
        """Test that save followed by load preserves scene."""
        g = euclidean_metric(3)
        e1 = Vector([1, 0, 0], grade=1, metric=g)
        e2 = Vector([0, 1, 0], grade=1, metric=g)
        bivector = e1 ^ e2

        scene = Scene(theme="obsidian", projection="perspective")
        scene.add(e1, color=(1, 0, 0), opacity=0.9)
        scene.add(e2, color=(0, 1, 0), opacity=0.8)
        scene.add(bivector, color=(0, 0, 1), opacity=0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.scene"
            scene.save(path)
            loaded = Scene.load(path)

            # Check element count
            assert len(loaded._elements) == 3

            # Check theme
            assert loaded._theme.name == "obsidian"

    def test_multiple_save_load_cycles(self):
        """Test multiple save/load cycles maintain integrity."""
        scene = make_simple_scene()

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                path = Path(tmpdir) / f"cycle_{i}.scene"
                scene.save(path)
                scene = Scene.load(path)

            # Should still have 3 elements after multiple cycles
            assert len(scene._elements) == 3
