"""
Manifolds - Meshes

A mesh is a sampled manifold: vertex positions as a grade-1 Vector of any
dimension and metric, and a CellComplex saying which vertices bound which
cells. It is what passes from geometry to drawing, and it carries no
derivatives. Each top cell also remembers the piece it came from, so a mesh
welded from many patches can still be colored patch by patch.
"""

from __future__ import annotations

from numpy import arange, asarray, concatenate, full, zeros
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from morphis.elements.metric import Metric
from morphis.elements.vector import Vector
from morphis.topology import CellComplex


class Mesh(BaseModel):
    """
    Vertices and the cells they bound.

    Attributes:
        vertices: Grade-1 Vector with lot (n_vertices,)
        cells: Cell complex over the vertex numbering
        pieces: Integer per top cell naming the piece it came from
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    vertices: Vector
    cells: CellComplex
    pieces: NDArray | None = None

    @model_validator(mode="after")
    def _validate(self):
        if self.vertices.grade != 1 or len(self.vertices.lot) != 1:
            raise ValueError("mesh vertices must be a grade-1 Vector with one lot axis")

        if self.vertices.lot[0] != self.cells.n_vertices:
            raise ValueError(f"{self.vertices.lot[0]} vertices for a complex over {self.cells.n_vertices}")

        if self.pieces is None:
            object.__setattr__(self, "pieces", zeros(len(self.cells.cells), dtype=int))

        return self

    @property
    def metric(self) -> Metric:
        return self.vertices.metric

    @property
    def n_vertices(self) -> int:
        return self.cells.n_vertices

    @classmethod
    def from_samples(cls, points: Vector, cells: CellComplex) -> Mesh:
        """A mesh from lot-shaped samples, flattened in the C order the complex expects."""
        data = points.data.reshape(-1, points.dim)

        return cls(vertices=Vector(data, grade=1, metric=points.metric), cells=cells)

    @classmethod
    def union(cls, *meshes: Mesh) -> Mesh:
        """Disjoint union, renumbering vertices; piece k of the result is the k-th mesh."""
        offsets = concatenate([[0], [mesh.n_vertices for mesh in meshes]]).cumsum()
        data = concatenate([mesh.vertices.data for mesh in meshes])
        cells = concatenate([mesh.cells.cells + offset for mesh, offset in zip(meshes, offsets, strict=False)])
        pieces = concatenate([full(len(mesh.cells.cells), k) for k, mesh in enumerate(meshes)])

        first = meshes[0].cells
        complex_ = CellComplex(cells=cells, dim=first.dim, shape=first.shape, n_vertices=int(offsets[-1]))

        return cls(vertices=Vector(data, grade=1, metric=meshes[0].metric), cells=complex_, pieces=pieces)

    def welded(self, tolerance: float = 1e-9) -> Mesh:
        """Glue vertices that coincide, as where patches of an atlas meet."""
        return Mesh(
            vertices=self.vertices,
            cells=self.cells.quotient_coincident(self.vertices, tolerance),
            pieces=self.pieces,
        )

    def piece(self, k: int) -> NDArray:
        """Indices of the top cells that came from piece k."""
        return arange(len(self.cells.cells))[asarray(self.pieces) == k]
