"""
Surface Element - 3D Mesh Geometry

Surface represents a 3D mesh with vertices as morphis Vectors, enabling
GA transformations on mesh geometry. Topology (faces) is stored separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

from numpy import asarray
from numpy.typing import NDArray
from pydantic import ConfigDict, field_validator, model_validator

from morphis.elements.base import Element
from morphis.elements.metric import Metric, euclidean_metric
from morphis.elements.vector import Vector


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector


# =============================================================================
# Intermediate Operation Class
# =============================================================================


class _SurfaceOp:
    """
    Intermediate state for Surface operations that yield non-Vector results.

    When a MultiVector multiplies a Surface (e.g., M * s), the result is
    a MultiVector per vertex. This class tracks the Surface context so that
    subsequent operations (e.g., * ~M) can complete and return a Surface.

    This enables the sandwich product pattern: M * s * ~M → Surface
    """

    __slots__ = ("surface", "current")

    def __init__(self, surface: "Surface", current):
        """
        Args:
            surface: Original Surface (for faces topology)
            current: Current state (MultiVector from M * vertices)
        """
        self.surface = surface
        self.current = current

    def __mul__(self, other) -> "Surface | _SurfaceOp":
        """
        Right multiplication: (M * surface) * other.

        If result is grade-1, returns Surface. Otherwise keeps intermediate.
        """
        from numpy import allclose

        from morphis.config import TOLERANCE
        from morphis.elements.multivector import MultiVector

        result = self.current * other

        # Check if result is grade-1 (can become Surface)
        if isinstance(result, Vector) and result.grade == 1:
            return Surface(vertices=result, faces=self.surface.faces)

        if isinstance(result, MultiVector):
            # Check if grade-1 exists and all other grades are negligible
            grade1 = result[1]
            if grade1 is not None:
                other_grades_zero = all(allclose(result[k].data, 0, atol=TOLERANCE) for k in result.grades if k != 1)
                if other_grades_zero:
                    return Surface(vertices=grade1, faces=self.surface.faces)

        # Still mixed grade, keep as intermediate
        return _SurfaceOp(self.surface, result)

    def __repr__(self) -> str:
        return f"_SurfaceOp(current={type(self.current).__name__})"


class Surface(Element):
    """
    A 3D mesh with GA-compatible vertices.

    Stores mesh geometry where vertices are grade-1 Vectors in 3D Euclidean
    space. Faces define triangle topology. Supports loading from various
    file formats (OBJ, STL, PLY, GLB/GLTF).

    Inherits from Element, providing:
        - metric: The geometric context (from vertices)
        - lot: Vertex count as (N,)
        - dim: 3 (Euclidean 3D)
        - apply_similarity(S, t): Transform vertices

    Attributes:
        vertices: Grade-1 Vector with lot=(N,) representing N vertex positions
        faces: NDArray of face connectivity [n_verts, v0, v1, v2, n_verts, ...]

    The face format uses:
        [3, v0, v1, v2, 3, v3, v4, v5, ...]
    where 3 indicates triangle, followed by vertex indices.

    Examples:
        # Create a simple triangle
        vertices = Vector([[0, 0, 0], [1, 0, 0], [0, 1, 0]], grade=1, metric=g)
        faces = np.array([3, 0, 1, 2])
        surface = Surface(vertices=vertices, faces=faces)

        # Transform vertices
        S = align_vectors(u, v)
        surface_transformed = surface.apply_similarity(S, t)

        # Load from file
        surface = Surface.from_file("model.obj")
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    vertices: Vector
    faces: NDArray

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("faces", mode="before")
    @classmethod
    def _convert_faces(cls, v):
        """Ensure faces is an integer array."""
        return asarray(v, dtype=int)

    @model_validator(mode="after")
    def _validate_geometry(self):
        """Validate vertices and faces, sync metric and lot."""
        # Vertices must be grade-1 in 3D
        if self.vertices.grade != 1:
            raise ValueError(f"Vertices must be grade-1 Vectors, got grade={self.vertices.grade}")

        if self.vertices.dim != 3:
            raise ValueError(f"Vertices must be 3D, got dim={self.vertices.dim}")

        if len(self.vertices.lot) != 1:
            raise ValueError(f"Vertices must have 1D lot (vertex count), got lot={self.vertices.lot}")

        # Sync metric and lot from vertices
        object.__setattr__(self, "metric", self.vertices.metric)
        object.__setattr__(self, "lot", self.vertices.lot)

        # Validate face indices are within bounds
        n_vertices = self.vertices.lot[0]
        if self.faces.size > 0:
            idx = 0
            while idx < len(self.faces):
                count = self.faces[idx]
                vertex_indices = self.faces[idx + 1 : idx + 1 + count]
                if len(vertex_indices) > 0 and vertex_indices.max() >= n_vertices:
                    raise ValueError(
                        f"Face references vertex {vertex_indices.max()}, but only {n_vertices} vertices exist"
                    )
                idx += 1 + count

        return self

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the surface."""
        return self.vertices.lot[0]

    @property
    def n_faces(self) -> int:
        """Number of faces in the surface."""
        if self.faces.size == 0:
            return 0

        count = 0
        idx = 0
        while idx < len(self.faces):
            face_size = self.faces[idx]
            count += 1
            idx += 1 + face_size
        return count

    # =========================================================================
    # Transformation Methods
    # =========================================================================

    def apply_similarity(
        self,
        S: "MultiVector",
        t: Vector | None = None,
    ) -> Self:
        """
        Apply a similarity transformation to vertices.

        Computes: transform(vertices, S) + t

        The similarity versor S (from align_vectors or a rotor) is applied via
        sandwich product. The optional translation t is added after.

        Args:
            S: Similarity versor or rotor (MultiVector with grades {0, 2})
            t: Optional translation vector (grade-1). Added after sandwich product.

        Returns:
            New Surface with transformed vertices (faces unchanged).

        Example:
            S, t = point_alignment(u1, u2, v1, v2)
            surface_world = surface_local.apply_similarity(S, t)
        """
        new_vertices = self.vertices.apply_similarity(S, t)
        return Surface(vertices=new_vertices, faces=self.faces)

    # =========================================================================
    # Arithmetic Operators
    # =========================================================================

    def __add__(self, other: Vector) -> "Surface":
        """
        Translation: s + v.

        Translates all vertices by adding a grade-1 vector.
        """
        return Surface(vertices=self.vertices + other, faces=self.faces)

    def __sub__(self, other: Vector) -> "Surface":
        """
        Translation: s - v.

        Translates all vertices by subtracting a grade-1 vector.
        """
        return Surface(vertices=self.vertices - other, faces=self.faces)

    def __neg__(self) -> "Surface":
        """
        Negation: -s.

        Reflects all vertices through the origin.
        """
        return Surface(vertices=-self.vertices, faces=self.faces)

    def __mul__(self, other) -> "Surface | _SurfaceOp":
        """
        Right multiplication: s * other.

        - Scalar: returns scaled Surface
        - MultiVector: returns intermediate for sandwich product
        """
        from morphis.elements.multivector import MultiVector

        result = self.vertices * other

        if isinstance(result, Vector):
            return Surface(vertices=result, faces=self.faces)
        if isinstance(result, MultiVector):
            return _SurfaceOp(self, result)

        raise TypeError(f"Cannot multiply Surface by {type(other)}")

    def __rmul__(self, other) -> "Surface | _SurfaceOp":
        """
        Left multiplication: other * s.

        - Scalar: returns scaled Surface
        - MultiVector: returns intermediate for sandwich product (M * s)
        """
        from morphis.elements.multivector import MultiVector

        result = other * self.vertices

        if isinstance(result, Vector):
            return Surface(vertices=result, faces=self.faces)
        if isinstance(result, MultiVector):
            return _SurfaceOp(self, result)

        raise TypeError(f"Cannot multiply {type(other)} by Surface")

    def __truediv__(self, other) -> "Surface":
        """
        Scalar division: s / a.

        Scales all vertices by 1/a.
        """
        return Surface(vertices=self.vertices / other, faces=self.faces)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        metric: Metric | None = None,
    ) -> Surface:
        """
        Load a 3D mesh from file.

        Supports formats:
        - OBJ (.obj)
        - STL (.stl)
        - PLY (.ply)
        - VTK (.vtk)
        - GLB (.glb) - requires trimesh, pygltflib, DracoPy
        - GLTF (.gltf) - requires trimesh, pygltflib, DracoPy

        Args:
            path: Path to the model file
            metric: Optional metric (defaults to 3D Euclidean)

        Returns:
            Surface with vertices as Vectors

        Examples:
            surface = Surface.from_file("bunny.obj")
            surface = Surface.from_file("moon.glb")
        """
        if metric is None:
            metric = euclidean_metric(3)

        path_str = str(path).lower()

        # Handle GLB/GLTF via specialized loaders
        if path_str.endswith((".glb", ".gltf")):
            return cls._from_gltf(path, metric)

        # All other formats via trimesh (lightweight, no pyvista needed)
        return cls._from_mesh_file(path, metric)

    @classmethod
    def _from_mesh_file(
        cls,
        path: str | Path,
        metric: Metric,
    ) -> Surface:
        """Load OBJ/STL/PLY via trimesh."""
        import trimesh

        mesh = trimesh.load(str(path))

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
            if mesh is None or len(mesh.vertices) == 0:
                raise ValueError("No geometry found in file")

        vertices = Vector(
            asarray(mesh.vertices, dtype=float),
            grade=1,
            metric=metric,
        )

        tri_faces = asarray(mesh.faces, dtype=int)
        pv_faces = asarray(
            [[3, *face] for face in tri_faces],
            dtype=int,
        ).flatten()

        return cls(vertices=vertices, faces=pv_faces)

    @classmethod
    def _from_gltf(
        cls,
        path: str | Path,
        metric: Metric,
    ) -> Surface:
        """Load GLB/GLTF file, handling Draco compression if present."""
        from pygltflib import GLTF2

        gltf = GLTF2.load(str(path))

        # Check if Draco compression is used
        uses_draco = gltf.extensionsUsed is not None and "KHR_draco_mesh_compression" in gltf.extensionsUsed

        if uses_draco:
            return cls._from_gltf_draco(gltf, metric)
        else:
            return cls._from_gltf_standard(gltf, metric)

    @classmethod
    def _from_gltf_draco(
        cls,
        gltf,
        metric: Metric,
    ) -> Surface:
        """Load Draco-compressed GLB/GLTF file."""
        import DracoPy

        binary_blob = gltf.binary_blob()

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                if prim.extensions is None:
                    continue

                draco_ext = prim.extensions.get("KHR_draco_mesh_compression")
                if draco_ext is None:
                    continue

                # Get compressed data from buffer view
                bv = gltf.bufferViews[draco_ext["bufferView"]]
                draco_data = binary_blob[bv.byteOffset : bv.byteOffset + bv.byteLength]

                # Decode with DracoPy
                decoded = DracoPy.decode(draco_data)

                # Collect vertices and faces
                all_vertices.append(asarray(decoded.points, dtype=float))

                faces = asarray(decoded.faces, dtype=int) + vertex_offset
                all_faces.append(faces)

                vertex_offset += len(decoded.points)

        if not all_vertices:
            raise ValueError("No Draco-compressed geometry found")

        # Combine all primitives
        combined_vertices = asarray([v for verts in all_vertices for v in verts], dtype=float)
        combined_faces = asarray([f for faces in all_faces for f in faces], dtype=int)

        vertices = Vector(combined_vertices, grade=1, metric=metric)

        # Convert faces to indexed format
        pv_faces = asarray(
            [[3, *face] for face in combined_faces],
            dtype=int,
        ).flatten()

        return cls(vertices=vertices, faces=pv_faces)

    @classmethod
    def _from_gltf_standard(
        cls,
        gltf,
        metric: Metric,
    ) -> Surface:
        """Load standard (non-Draco) GLB/GLTF file via trimesh."""
        import trimesh

        # For non-Draco files, trimesh works fine
        scene = trimesh.load(gltf)

        if isinstance(scene, trimesh.Scene):
            mesh = scene.to_geometry()
            if mesh is None or len(mesh.vertices) == 0:
                raise ValueError("No geometry found")
        else:
            mesh = scene

        vertices = Vector(
            asarray(mesh.vertices, dtype=float),
            grade=1,
            metric=metric,
        )

        tri_faces = asarray(mesh.faces, dtype=int)
        pv_faces = asarray(
            [[3, *face] for face in tri_faces],
            dtype=int,
        ).flatten()

        return cls(vertices=vertices, faces=pv_faces)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def copy(self) -> Surface:
        """Create a deep copy of this surface."""
        return Surface(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
        )

    def with_metric(self, metric: Metric) -> Surface:
        """Return a new Surface with the specified metric context."""
        return Surface(
            vertices=self.vertices.with_metric(metric),
            faces=self.faces,
        )

    def __repr__(self) -> str:
        return f"Surface(n_vertices={self.n_vertices}, n_faces={self.n_faces})"
