"""
Projective Maps to the Picture

The metric decides the geometry; the projection decides the picture. A
projection from a d-dimensional space to an m-dimensional picture is a
projective map ℝℙ^d → ℝℙ^m, an (m + 1) × (d + 1) matrix acting on homogeneous
coordinates (x, 1):

    (y, w) = M (x, 1),        Π(x) = y / w

Its kernel is the center of projection. An orthographic map has an ideal center
and a constant weight row; a perspective map, such as stereographic projection,
has a finite one. Maps compose with @, so a stereographic map from S³ followed
by an orthographic one reaches the 3D drawing space from ℝ⁴, and longer chains
reach it from any d.

The metric may help build a projection, as for an orthonormal screen, but is
never consulted when one is applied.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from numpy import asarray, concatenate, einsum, eye, ones, sqrt, zeros
from numpy.linalg import svd
from numpy.typing import NDArray

from morphis.elements.frame import Frame
from morphis.elements.metric import Label, Metric, euclidean_metric
from morphis.elements.vector import Vector


class Projection:
    """
    A projective map from a space of any dimension to an m-dimensional picture.

    Args:
        matrix: (m + 1) × (d + 1) matrix on homogeneous coordinates, the last
            row giving the weight
        source: The space being drawn

    Attributes:
        target_dim: Dimension m of the picture
        target: Euclidean metric of the picture, the source of a following map
    """

    __slots__ = ("matrix", "source")

    def __init__(self, matrix: NDArray, source: Metric):
        matrix = asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != source.dim + 1:
            raise ValueError(
                f"a projection from dimension {source.dim} needs {source.dim + 1} columns, got {matrix.shape}"
            )

        self.matrix = matrix
        self.source = source

    @property
    def target_dim(self) -> int:
        return self.matrix.shape[0] - 1

    @property
    def target(self) -> Metric:
        return euclidean_metric(self.target_dim)

    def __repr__(self) -> str:
        return f"Projection({self.source!r} → ℝ^{self.target_dim})"

    # =========================================================================
    # Construction
    # =========================================================================

    @classmethod
    def linear(cls, metric: Metric, images: Mapping[Label, Sequence[float]]) -> Projection:
        """
        Send each basis direction to a chosen picture vector.

        This is the deliberately unfaithful depiction: directions left out are
        drawn as zero, and two directions may share one image.

        Args:
            metric: The space being drawn
            images: Map from label (1, Dot(1), ...) to its picture vector
        """
        m = len(next(iter(images.values())))
        matrix = zeros((m + 1, metric.dim + 1))
        for label, image in images.items():
            matrix[:m, metric.to_internal(label)] = asarray(image, dtype=float)

        matrix[m, metric.dim] = 1.0

        return cls(matrix, metric)

    @classmethod
    def orthographic(cls, screen: Frame | Sequence[Vector]) -> Projection:
        """
        Orthographic projection onto a screen: picture coordinate c is e_c · x.

        Args:
            screen: The screen directions, an orthonormal Frame or a sequence of
                grade-1 Vectors
        """
        frame = screen if isinstance(screen, Frame) else Frame(*screen)
        metric = frame.metric
        m = frame.span
        matrix = zeros((m + 1, metric.dim + 1))
        matrix[:m, : metric.dim] = einsum("cm, mn -> cn", frame.data, metric.data)
        matrix[m, metric.dim] = 1.0

        return cls(matrix, metric)

    @classmethod
    def stereographic(cls, pole: Vector) -> Projection:
        """
        Stereographic projection of the sphere |x| = |pole| from the pole.

        Points are sent from the pole onto the hyperplane through the origin
        perpendicular to it: y = ρ x_⊥ / (ρ - n · x), with n the unit pole and ρ
        its length. The picture coordinates are the basis directions other
        than the pole's, in order, when the pole is a basis direction, and an
        orthonormal completion of n otherwise.
        """
        metric = pole.metric
        d = metric.dim
        radius = float(sqrt(abs(einsum("m, mn, n ->", pole.data, metric.data, pole.data))))
        direction = pole.data / radius

        is_axis = (abs(direction) > 1e-12).sum() == 1
        if is_axis:
            rest = [n for n in range(d) if abs(direction[n]) <= 1e-12]
            complement = eye(d)[rest]

        else:
            _, _, rows = svd(direction[None, :])
            complement = rows[1:]

        matrix = zeros((d, d + 1))
        matrix[: d - 1, :d] = einsum("cm, mn -> cn", complement, metric.data)
        matrix[d - 1, :d] = -einsum("m, mn -> n", direction, metric.data) / radius
        matrix[d - 1, d] = 1.0

        return cls(matrix, metric)

    def __matmul__(self, other: Projection) -> Projection:
        """Composition (self ∘ other): first other, then self."""
        if not isinstance(other, Projection):
            return NotImplemented

        if other.target_dim != self.source.dim:
            raise ValueError(f"cannot follow a map to ℝ^{other.target_dim} by one from dimension {self.source.dim}")

        return Projection(self.matrix @ other.matrix, other.source)

    # =========================================================================
    # Application
    # =========================================================================

    def _homogeneous(self, points: Vector | NDArray) -> NDArray:
        data = points.data if isinstance(points, Vector) else asarray(points, dtype=float)
        lifted = concatenate([data, ones(data.shape[:-1] + (1,))], axis=-1)

        return einsum("am, ...m -> ...a", self.matrix, lifted)

    def __call__(self, points: Vector | NDArray) -> NDArray:
        """Picture coordinates of points of any lot, dividing by the weight once, at the end."""
        image = self._homogeneous(points)

        return image[..., :-1] / image[..., -1:]

    def pushforward(self, points: Vector | NDArray, vectors: Frame | Vector | NDArray) -> NDArray:
        """
        The differential of the map at points, applied to tangent vectors.

        For Π(x) = y / w the differential is dΠ(v) = (A v - Π(x) (c · v)) / w,
        where A is the linear block and c the weight row.

        Args:
            points: Base points, lot-shaped
            vectors: A Frame at those points, or vectors broadcasting against them

        Returns:
            Picture vectors; a Frame of span k gives shape (*lot, k, m)
        """
        image = self._homogeneous(points)
        weight = image[..., -1]
        picture = image[..., :-1] / weight[..., None]

        data = vectors.data if isinstance(vectors, (Frame, Vector)) else asarray(vectors, dtype=float)
        is_frame = isinstance(vectors, Frame)
        if is_frame:
            picture, weight = picture[..., None, :], weight[..., None]

        linear = einsum("am, ...m -> ...a", self.matrix[:-1, :-1], data)
        rate = einsum("m, ...m -> ...", self.matrix[-1, :-1], data)

        result = (linear - picture * rate[..., None]) / weight[..., None]

        return result

    def pinching(self, points: Vector | NDArray, frame: Frame) -> NDArray:
        """
        How nearly the map collapses a tangent space: s_min / s_max of dΠ on the frame.

        It is 1 where the tangent space is drawn without distortion of shape and
        0 at a pinch point, where the differential drops rank and the drawn
        surface folds to a cusp.
        """
        singular = svd(self.pushforward(points, frame), compute_uv=False)

        return singular[..., -1] / singular[..., 0]
