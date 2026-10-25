"""
Manifolds - Atlases

An atlas is a set of patches covering one manifold. Sampling each patch and
gluing the samples that coincide turns many sheets into one surface, whose
topology is then that of the manifold: its Euler characteristic and boundary
are read from the welded complex.
"""

from __future__ import annotations

from typing import Sequence

from morphis.elements.metric import Metric
from morphis.manifold.mesh import Mesh
from morphis.manifold.patch import Patch


class Atlas:
    """
    Patches of one manifold.

    Args:
        patches: The patches, all in one space and of one intrinsic dimension
    """

    __slots__ = ("patches",)

    def __init__(self, patches: Sequence[Patch]):
        patches = tuple(patches)
        if len({patch.metric for patch in patches}) != 1 or len({patch.span for patch in patches}) != 1:
            raise ValueError("an atlas needs patches of one dimension in one space")

        self.patches = patches

    @property
    def metric(self) -> Metric:
        return self.patches[0].metric

    @property
    def span(self) -> int:
        return self.patches[0].span

    def __len__(self) -> int:
        return len(self.patches)

    def __repr__(self) -> str:
        return f"Atlas({len(self)} patches, span={self.span}, {self.metric!r})"

    def mesh(self, shape: int | tuple[int, ...], tolerance: float = 1e-9) -> Mesh:
        """
        Sample every patch on a grid of the given shape and weld the samples.

        Piece k of the mesh is patch k.
        """
        pieces = [Mesh.from_samples(patch(patch.grid(shape)), patch.cells(shape)) for patch in self.patches]

        return Mesh.union(*pieces).welded(tolerance)
