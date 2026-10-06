"""
Depiction

A conceptual figure draws objects that live in their true space, often of
higher dimension than three, inside a 3D drawing space. The depiction is the
deliberate, possibly unfaithful, linear map between the two: for each basis
direction of the true space it names the drawing direction that stands in for
it. Two true directions may share one drawing direction, as when two conjugate
planes are drawn with parallel representative axes; that is a choice of the
figure, stated once here, while the objects keep their exact relationships.

Indices are the user-facing geometric indices of the true space's metric, so
e_1 is always addressed as 1.
"""

from __future__ import annotations

from numpy import asarray, einsum, zeros
from numpy.typing import NDArray

from morphis.elements.metric import Metric
from morphis.elements.vector import Vector


class Depiction:
    """
    Linear map from a true space to the 3D drawing space.

    Args:
        metric: Metric of the true space
        images: Map from geometric index to the drawing vector that depicts it.
            Directions left out are depicted as zero. When the true space is
            3D and no images are given, the depiction is the identity.

    Example:
        D = Depiction(euclidean_metric(4), {1: (1, 0, 0), 2: (0, 0, 1), 3: (1, 0, 0), 4: (0, 1, 0)})
        D(state)  # drawing coordinates of the 4D vector ψ
    """

    def __init__(self, metric: Metric, images: dict[int, tuple[float, float, float]] | None = None):
        is_identity = images is None and metric.dim == 3
        defaults = {metric.to_user(n): row for n, row in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1)))}
        chosen = defaults if is_identity else (images or {})

        matrix = zeros((3, metric.dim))
        for index, image in chosen.items():
            matrix[:, metric.to_internal(index)] = asarray(image, dtype=float)

        self.metric = metric
        self.matrix = matrix

    def __call__(self, v: Vector | NDArray) -> NDArray:
        """Drawing coordinates of a grade-1 vector (any lot) or a raw component array."""
        components = v.data if isinstance(v, Vector) else asarray(v, dtype=float)
        drawn = einsum("ca, ...a -> ...c", self.matrix, components)

        return drawn
