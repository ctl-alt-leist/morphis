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

A View is the setting a renderer is given: which kind of projection, onto
which three directions, and its few parameters. It names labels rather than
storage slots, so it holds no metric; each element resolves it against its own
metric into a Projection to the 3D drawing space.

    Orthographic(axes=(1, 2, 3))                 the axes, read off directly
    Perspective(axes=(1, 2, 3), distance=4.0)    an eye at that distance along the dropped directions
    Stereographic(axes=(1, 2, 3), pole=4)        from the pole of the sphere, then onto the axes
"""

from __future__ import annotations

from typing import Mapping, Sequence

from numpy import asarray, concatenate, einsum, eye, ones, sqrt, zeros
from numpy.linalg import svd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from morphis.elements.frame import Frame
from morphis.elements.metric import Label, Metric, euclidean_metric
from morphis.elements.vector import Vector, basis_vector


# The first three spatial directions; x is 1 in every signature
DEFAULT_AXES: tuple[Label, Label, Label] = (1, 2, 3)


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

    @property
    def is_affine(self) -> bool:
        """True when the weight is constant, so displacements have images of their own."""
        return bool((abs(self.matrix[-1, :-1]) < 1e-15).all())

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

    def blade(self, b: Vector, at: Vector | NDArray | None = None) -> NDArray:
        """
        A k-vector attached at a point, carried into the picture by ∧ᵏ dΠ.

        A bivector at x becomes B'^{pq} = dΠ^p_a dΠ^q_b B^{ab}, the outermorphism
        of the differential, so its plane is the image plane and its magnitude
        the drawn area. Under an orthographic map onto basis directions this is
        the sub-array of components at those directions.

        Args:
            b: A grade-k Vector of any lot
            at: The point it is attached at, the origin by default

        Returns:
            Picture components, shape (*lot, m, ..., m) with k picture axes
        """
        d = self.source.dim
        point = zeros(d) if at is None else (at.data if isinstance(at, Vector) else asarray(at, dtype=float))

        # Row a is dΠ(e_a)
        images = self.pushforward(point[..., None, :], eye(d))

        source, picture = "abcd"[: b.grade], "pqrs"[: b.grade]
        factors = [f"...{a}{p}" for a, p in zip(source, picture, strict=True)]
        subscripts = ", ".join([*factors, f"...{source}"]) + f" -> ...{picture}"
        result = einsum(subscripts, *([images] * b.grade), b.data)

        return result


# =============================================================================
# Views
# =============================================================================


class View(BaseModel):
    """
    How a renderer draws a space of any dimension in three: the kind of
    projection, the three directions that become the picture's x, y, z, and the
    projection's parameters.

    Spaces of dimension 3 or less are drawn in their own coordinates by every
    view. Above that, resolve() builds the map for the element's metric, and an
    axis that is not a direction of it raises IndexError.

    Args:
        axes: The labels of the directions drawn as x, y, z
    """

    model_config = ConfigDict(frozen=True)

    axes: tuple[Label, Label, Label] = DEFAULT_AXES

    @field_validator("axes", mode="before")
    @classmethod
    def _three_labels(cls, axes):
        axes = tuple(axes)
        if len(axes) != 3:
            raise ValueError(f"a view draws exactly 3 directions, got {axes!r}")

        return axes

    def resolve(self, metric: Metric) -> Projection:
        """The projection of this view from the given space to 3D."""
        result = _own_coordinates(metric) if metric.dim <= 3 else self._project(metric)

        return result

    def _project(self, metric: Metric) -> Projection:
        raise NotImplementedError

    def _slots(self, metric: Metric) -> list[int]:
        slots = [metric.to_internal(label) for label in self.axes]
        if len(set(slots)) != 3:
            raise ValueError(f"the axes of a view must be distinct, got {self.axes!r}")

        return slots


class Orthographic(View):
    """
    Parallel projection onto the axes: x, y, z are the components along them,
    and every other direction is dropped.
    """

    def _project(self, metric: Metric) -> Projection:
        matrix = zeros((4, metric.dim + 1))
        for row, slot in enumerate(self._slots(metric)):
            matrix[row, slot] = 1.0

        matrix[3, metric.dim] = 1.0

        return Projection(matrix, metric)


class Perspective(View):
    """
    Central projection with the eye at the given distance along the dropped
    directions.

    A point x is drawn at D x_axes / (D - Σ x_dropped), with D the distance:
    the weight row is 1 - Σ x_dropped / D, so the center of projection is the
    flat where the axes components vanish and Σ x_dropped = D. From ℝ⁴ the eye
    sits at D e_4, and the far cell of a tesseract is drawn inside the near one.

    Args:
        distance: Distance of the eye from the origin
    """

    distance: float = 4.0

    def _project(self, metric: Metric) -> Projection:
        slots = self._slots(metric)
        matrix = zeros((4, metric.dim + 1))
        for row, slot in enumerate(slots):
            matrix[row, slot] = 1.0

        dropped = [slot for slot in range(metric.dim) if slot not in slots]
        matrix[3, dropped] = -1.0 / self.distance
        matrix[3, metric.dim] = 1.0

        return Projection(matrix, metric)


class Stereographic(View):
    """
    Stereographic projection of the sphere of the given radius from the pole
    onto the hyperplane through the origin perpendicular to it, followed by
    the orthographic projection onto the axes.

    A great circle through the pole is drawn as a straight line, and every
    other circle on the sphere as a circle.

    Args:
        pole: Label of the pole direction; the last direction by default
        radius: Radius of the sphere
    """

    pole: Label | None = None
    radius: float = 1.0

    def _project(self, metric: Metric) -> Projection:
        pole = metric.dim - 1 if self.pole is None else metric.to_internal(self.pole)
        slots = self._slots(metric)
        if pole in slots:
            raise ValueError(f"the pole {metric.to_user(pole)} cannot also be an axis of the view")

        # The stereographic picture keeps the other directions in storage order
        sphere = Projection.stereographic(self.radius * basis_vector(metric.to_user(pole), metric))
        rest = [slot for slot in range(metric.dim) if slot != pole]
        matrix = zeros((4, metric.dim))
        for row, slot in enumerate(slots):
            matrix[row, rest.index(slot)] = 1.0

        matrix[3, metric.dim - 1] = 1.0

        return Projection(matrix, sphere.target) @ sphere


def _own_coordinates(metric: Metric) -> Projection:
    """A space of dimension 3 or less, drawn in its own coordinates and padded with zeros."""
    matrix = zeros((4, metric.dim + 1))
    matrix[: metric.dim, : metric.dim] = eye(metric.dim)
    matrix[3, metric.dim] = 1.0

    return Projection(matrix, metric)


def resolve(view: View | Projection, metric: Metric) -> Projection:
    """The projection to 3D that a view, or a given projection, draws the space with."""
    if isinstance(view, Projection):
        if view.source.dim != metric.dim or view.target_dim != 3:
            raise ValueError(f"{view!r} cannot draw an element of {metric!r} in 3D")

        result = view

    else:
        result = view.resolve(metric)

    return result
