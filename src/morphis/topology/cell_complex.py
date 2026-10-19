"""
Topology - Cell Complexes

A cell complex is the combinatorial shape of a sampled manifold: which sample
points bound which cells. It knows nothing of dimension, metric, or position,
so one complex serves a torus in ℝ³, in ℝ⁴, and in ℝ^{2,2}.

Top cells are rows of vertex indices. A cube cell of dimension r lists its 2^r
corners in binary order, corner (c_1, ..., c_r) at position Σ c_a 2^{a-1}, so a
square is (00, 10, 01, 11); a simplex cell of dimension r lists its r + 1
corners. The vertices of a grid complex are numbered in the C order of the
coordinate grid, so a lot-shaped Vector of positions flattens onto them.
"""

from __future__ import annotations

from typing import Literal

from numpy import arange, asarray, indices, ndarray, prod, ravel_multi_index, stack, unique, zeros
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.spatial import cKDTree


CellShape = Literal["cube", "simplex"]


def _cube_faces(r: int) -> list[list[int]]:
    """Corner positions of the 2r facets of an r-cube, each in binary order."""
    faces = []
    for axis in range(r):
        for side in (0, 1):
            corners = [n for n in range(2**r) if (n >> axis) & 1 == side]
            faces.append(corners)

    return faces


def _simplex_faces(r: int) -> list[list[int]]:
    """Corner positions of the r + 1 facets of an r-simplex."""
    faces = [[n for n in range(r + 1) if n != omitted] for omitted in range(r + 1)]

    return faces


class CellComplex(BaseModel):
    """
    A complex of r-cells over numbered vertices.

    Attributes:
        cells: Integer array of shape (n_cells, n_corners), one row per top cell
        dim: Dimension r of the top cells
        shape: "cube" (2^r corners) or "simplex" (r + 1 corners)
        n_vertices: Number of vertices the indices refer to

    Lower-dimensional cells are derived: two cells with the same set of
    distinct vertices are the same cell, and a cell whose corners collapse to
    fewer than r + 1 distinct vertices no longer spans r dimensions and is
    dropped. A quotient that pinches a row of a grid to a point therefore
    leaves a valid complex, as at the poles of a sphere.

    Examples:
        >>> torus = CellComplex.grid((8, 6), periodic=(True, True))
        >>> torus.euler_characteristic()
        0
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    cells: NDArray
    dim: int
    shape: CellShape = "cube"
    n_vertices: int

    @field_validator("cells", mode="before")
    @classmethod
    def _convert_cells(cls, v):
        cells = asarray(v, dtype=int)
        if cells.ndim != 2:
            raise ValueError(f"cells must be a 2D integer array, got shape {cells.shape}")

        return cells

    # =========================================================================
    # Construction
    # =========================================================================

    @classmethod
    def grid(cls, shape: tuple[int, ...], periodic: tuple[bool, ...] | bool = False) -> CellComplex:
        """
        The cube complex of a coordinate grid.

        Axis a has shape[a] sample points. A periodic axis closes into a circle,
        its last sample joined to its first, so it has shape[a] cells along it;
        an open axis has shape[a] - 1.

        Args:
            shape: Number of samples along each coordinate
            periodic: Which axes close into circles, one flag per axis or one
                flag for all

        Returns:
            Cube complex of dimension len(shape), vertices in C order
        """
        shape = tuple(int(n) for n in shape)
        periodic = (periodic,) * len(shape) if isinstance(periodic, bool) else tuple(periodic)
        if len(periodic) != len(shape):
            raise ValueError(f"need one periodic flag per axis, got {len(periodic)} for {len(shape)} axes")

        r = len(shape)
        n_cells_along = [n if closed else n - 1 for n, closed in zip(shape, periodic, strict=True)]
        origins = indices(n_cells_along).reshape(r, -1).T

        corners = []
        for n in range(2**r):
            offset = [(n >> axis) & 1 for axis in range(r)]
            index = [(origins[:, a] + offset[a]) % shape[a] for a in range(r)]
            corners.append(ravel_multi_index(index, shape))

        cells = stack(corners, axis=-1).reshape(-1, 2**r)
        n_vertices = int(prod(shape))

        result = cls(cells=cells, dim=r, shape="cube", n_vertices=n_vertices)

        return result

    # =========================================================================
    # Derived Cells
    # =========================================================================

    def _facet_corners(self) -> list[list[int]]:
        result = _cube_faces(self.dim) if self.shape == "cube" else _simplex_faces(self.dim)

        return result

    def faces(self) -> CellComplex:
        """
        The complex of (r - 1)-cells, each appearing once.

        Raises:
            ValueError: If the complex is 0-dimensional
        """
        faces, _ = self._faces_with_counts()

        return faces

    def _faces_with_counts(self) -> tuple[CellComplex, NDArray]:
        if self.dim == 0:
            raise ValueError("a 0-dimensional complex has no faces")

        rows = []
        keys: dict[tuple[int, ...], int] = {}
        counts: list[int] = []
        for cell in self.cells:
            for corners in self._facet_corners():
                face = cell[corners]
                key = tuple(sorted(set(face.tolist())))
                if len(key) < self.dim:
                    continue

                if key in keys:
                    counts[keys[key]] += 1

                else:
                    keys[key] = len(rows)
                    rows.append(face)
                    counts.append(1)

        n_corners = len(self._facet_corners()[0]) if self.dim > 0 else 1
        cells = stack(rows) if rows else zeros((0, n_corners), dtype=int)
        faces = CellComplex(cells=cells, dim=self.dim - 1, shape=self.shape, n_vertices=self.n_vertices)

        return faces, asarray(counts, dtype=int)

    def skeleton(self, r: int) -> CellComplex:
        """The complex of r-cells, for 0 ≤ r ≤ dim."""
        if not 0 <= r <= self.dim:
            raise ValueError(f"skeleton dimension {r} outside 0 to {self.dim}")

        result = self
        for _ in range(self.dim - r):
            result = result.faces()

        return result

    def vertices(self) -> NDArray:
        """Indices of the vertices that some cell uses, sorted."""
        return unique(self.cells)

    def boundary(self) -> CellComplex:
        """The (r - 1)-cells that bound exactly one top cell."""
        faces, counts = self._faces_with_counts()

        result = CellComplex(
            cells=faces.cells[counts == 1],
            dim=faces.dim,
            shape=self.shape,
            n_vertices=self.n_vertices,
        )

        return result

    def euler_characteristic(self) -> int:
        """χ = Σ_r (-1)^r (number of r-cells), counting the vertices cells use."""
        result = 0
        level = self
        for r in range(self.dim, 0, -1):
            result += (-1) ** r * len(level.cells)
            level = level.faces()

        result += len(level.vertices())

        return result

    def components(self) -> int:
        """Number of connected components, joining vertices that share a cell."""
        parent = arange(self.n_vertices)

        def root(n: int) -> int:
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]

            return n

        for cell in self.cells:
            first = root(int(cell[0]))
            for n in cell[1:]:
                parent[root(int(n))] = first

        result = len({root(int(n)) for n in self.vertices()})

        return result

    # =========================================================================
    # Gluing
    # =========================================================================

    def quotient(self, identifications: NDArray | list[tuple[int, int]]) -> CellComplex:
        """
        Glue vertices together.

        Args:
            identifications: Either pairs (a, b) of vertices to identify, or an
                array of length n_vertices sending each vertex to its class

        Returns:
            Complex on the same vertex numbering, each class represented by
            its smallest member; unused vertices remain counted in n_vertices
        """
        representative = _classes(identifications, self.n_vertices)

        result = CellComplex(
            cells=representative[self.cells],
            dim=self.dim,
            shape=self.shape,
            n_vertices=self.n_vertices,
        )

        return result

    def quotient_coincident(self, positions, tolerance: float = 1e-9) -> CellComplex:
        """
        Glue vertices whose positions coincide within a tolerance.

        This is how separately sampled patches become one manifold. The
        positions are used only to find the identifications.

        Args:
            positions: Array or Vector whose data has shape (n_vertices, dim)
                after flattening any lot axes
            tolerance: Distance below which two vertices are the same point
        """
        data = positions.data if hasattr(positions, "data") and not isinstance(positions, ndarray) else positions
        points = asarray(data).reshape(self.n_vertices, -1)
        pairs = cKDTree(points).query_pairs(tolerance, output_type="ndarray")

        result = self.quotient(pairs)

        return result

    # =========================================================================
    # Rendering
    # =========================================================================

    def triangles(self) -> NDArray:
        """
        Triangles covering the 2-cells, for renderers that draw only triangles.

        Each square (00, 10, 01, 11) splits along its 00–11 diagonal; triangles
        with a repeated vertex are dropped.
        """
        if self.dim != 2:
            raise ValueError(f"triangles need a 2-dimensional complex, got dimension {self.dim}")

        if self.shape == "cube":
            a, b, c, d = self.cells.T
            candidates = stack([stack([a, b, d], axis=-1), stack([a, d, c], axis=-1)], axis=1).reshape(-1, 3)

        else:
            candidates = self.cells

        distinct = (
            (candidates[:, 0] != candidates[:, 1])
            & (candidates[:, 1] != candidates[:, 2])
            & (candidates[:, 0] != candidates[:, 2])
        )

        result = candidates[distinct]

        return result


# =============================================================================
# Helpers
# =============================================================================


def _classes(identifications, n_vertices: int) -> NDArray:
    """Map each vertex to the smallest member of its identification class."""
    identifications = asarray(identifications, dtype=int)

    if identifications.ndim == 1:
        if identifications.shape[0] != n_vertices:
            raise ValueError(f"a class map needs one entry per vertex, got {identifications.shape[0]}")

        pairs = stack([arange(n_vertices), identifications], axis=-1)

    else:
        pairs = identifications.reshape(-1, 2)

    parent = arange(n_vertices)

    def root(n: int) -> int:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]

        return n

    for a, b in pairs:
        root_a, root_b = root(int(a)), root(int(b))
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    result = asarray([root(n) for n in range(n_vertices)], dtype=int)

    return result
