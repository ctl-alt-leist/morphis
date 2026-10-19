"""
Geometric Algebra - Metric and Space

The Metric is the complete geometric context for GA objects: the metric tensor
g_{ab} together with the block layout of the space it acts on. A space is
declared by its blocks,

    metric(space=3)                          # Euclidean ℝ³
    metric(space=3, projective=1)            # PGA
    metric(time=1, space=3)                  # spacetime
    metric(time=1, space=3, projective=1)    # spacetime PGA
    metric(modes=2)                          # ℂ² as ℝ⁴

and the layout fixes the storage order and the label of every direction. Every
GA object requires a Metric; operations between objects with different metrics
raise ValueError.
"""

from __future__ import annotations

from functools import cached_property, lru_cache
from typing import Literal, NamedTuple

from numpy import array_equal, asarray, diag, sum as np_sum
from numpy.linalg import eigvalsh
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# =============================================================================
# Labels
# =============================================================================


class Dot(BaseModel):
    """
    The partner label ṁ of mode m in a space with complex dimensions.

    A space declared with modes pairs each representative direction e_m with
    its partner e_ṁ = J(e_m). The representative is labeled by the integer m
    and the partner by Dot(m), so basis_vector(Dot(1), g) is e_1̇.
    """

    model_config = ConfigDict(frozen=True)

    mode: int

    def __init__(self, mode: int, /):
        super().__init__(mode=mode)

    def __str__(self) -> str:
        return f"{self.mode}̇"

    def __repr__(self) -> str:
        return f"Dot({self.mode})"


Label = int | Dot


# =============================================================================
# Blocks
# =============================================================================


BlockKind = Literal["time", "space", "projective", "modes"]


class Block(NamedTuple):
    """
    A run of directions of one kind, in storage order.

    A modes block of size n holds 2n directions, each mode adjacent to its
    partner: 1, 1̇, 2, 2̇, ...
    """

    kind: BlockKind
    size: int

    @property
    def dim(self) -> int:
        """Number of directions the block contributes."""
        return 2 * self.size if self.kind == "modes" else self.size


Convention = Literal["mostly minus", "mostly plus"]

_CONVENTIONS: tuple[str, ...] = ("mostly minus", "mostly plus")


def _block_labels(blocks: tuple[Block, ...]) -> tuple[Label, ...]:
    """
    Label every direction of a block layout.

    The first spatial direction is always 1. Time is 0. A projective direction
    stored before the spatial block is 0; projective directions stored after it
    continue the spatial count. Modes are labeled m and Dot(m).
    """
    n_spatial = sum(block.size for block in blocks if block.kind in ("space", "modes"))
    seen_spatial = False
    labels: list[Label] = []

    for block in blocks:
        if block.kind == "time":
            labels.extend(range(1 - block.size, 1))

        elif block.kind == "space":
            labels.extend(range(1, block.size + 1))
            seen_spatial = True

        elif block.kind == "modes":
            for m in range(1, block.size + 1):
                labels.extend((m, Dot(m)))
            seen_spatial = True

        elif seen_spatial:
            labels.extend(range(n_spatial + 1, n_spatial + block.size + 1))

        else:
            labels.extend(range(1 - block.size, 1))

    return tuple(labels)


# =============================================================================
# Metric
# =============================================================================


class Metric(BaseModel):
    """
    Complete geometric context: the metric tensor and the block layout.

    Attributes:
        data: The metric tensor g_{ab} in storage order
        blocks: The block layout, in storage order

    The layout determines the label of each direction (see `labels`), while the
    tensor determines the geometry. Two metrics are equal when both agree, so
    ℝ^{2,2} and ℝ^{1,3} are different spaces.

    Examples:
        >>> g = metric(time=1, space=3)
        >>> g.labels
        (0, 1, 2, 3)
        >>> g.signature
        (1, 3, 0)
        >>> metric(modes=2).labels
        (1, Dot(1), 2, Dot(2))
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    data: NDArray
    blocks: tuple[Block, ...]

    @field_validator("data", mode="before")
    @classmethod
    def _convert_to_array(cls, v):
        return asarray(v, dtype=float)

    @field_validator("blocks", mode="before")
    @classmethod
    def _convert_blocks(cls, v):
        return tuple(Block(*block) for block in v)

    @model_validator(mode="after")
    def _check_layout(self):
        if self.data.ndim != 2 or self.data.shape[0] != self.data.shape[1]:
            raise ValueError(f"metric tensor must be square, got shape {self.data.shape}")

        if not array_equal(self.data, self.data.T):
            raise ValueError("metric tensor must be symmetric")

        n_directions = sum(block.dim for block in self.blocks)
        if n_directions != self.dim:
            raise ValueError(f"blocks describe {n_directions} directions but the tensor is {self.dim}-dimensional")

        return self

    # =========================================================================
    # Shape of the Space
    # =========================================================================

    @property
    def dim(self) -> int:
        """Dimension of the underlying vector space."""
        return self.data.shape[0]

    def _count(self, kind: BlockKind) -> int:
        return sum(block.size for block in self.blocks if block.kind == kind)

    @property
    def n_time(self) -> int:
        """Number of time directions."""
        return self._count("time")

    @property
    def n_space(self) -> int:
        """Number of real spatial directions (modes excluded)."""
        return self._count("space")

    @property
    def n_projective(self) -> int:
        """Number of projective (null) directions."""
        return self._count("projective")

    @property
    def n_modes(self) -> int:
        """Number of complex modes; each contributes two real directions."""
        return self._count("modes")

    @property
    def is_projective(self) -> bool:
        """True when the space carries a projective direction."""
        return self.n_projective > 0

    @cached_property
    def kinds(self) -> tuple[BlockKind, ...]:
        """The block kind of each direction, in storage order."""
        kinds = tuple(block.kind for block in self.blocks for _ in range(block.dim))

        return kinds

    # =========================================================================
    # Signature
    # =========================================================================

    @property
    def signature(self) -> tuple[int, int, int]:
        """
        The inertia (p, q, r) of the tensor: counts of positive, negative, and
        zero eigenvalues.
        """
        eigenvalues = eigvalsh(self.data)
        scale = max(1.0, float(abs(eigenvalues).max(initial=0.0)))
        tol = 1e-12 * scale
        p = int(np_sum(eigenvalues > tol))
        q = int(np_sum(eigenvalues < -tol))
        r = self.dim - p - q

        return (p, q, r)

    @property
    def convention(self) -> Convention | None:
        """
        The sign convention of a space with time: "mostly minus" when time
        squares to +1, "mostly plus" when it squares to -1. None without time.
        """
        result = None
        if self.n_time > 0:
            time_slot = self.kinds.index("time")
            result = "mostly minus" if self.data[time_slot, time_slot] > 0 else "mostly plus"

        return result

    # =========================================================================
    # Geometric Index Convention
    # =========================================================================
    #
    # Storage is always 0-based. These methods translate between the label of
    # a direction (its geometric index, as written on paper) and its storage
    # slot. They apply to geometric axes only; lot axes are standard 0-based
    # Python indexing. Every geometric access point (.on, basis construction,
    # display) goes through them, so the block layout is the single place that
    # defines the convention.

    @cached_property
    def labels(self) -> tuple[Label, ...]:
        """The label of each direction, in storage order."""
        return _block_labels(self.blocks)

    def to_internal(self, index: Label) -> int:
        """
        Convert a geometric index (a direction's label) to its storage slot.

        Raises IndexError if no direction carries that label.
        """
        labels = self.labels
        key = index if isinstance(index, Dot) else int(index)
        if key not in labels:
            shown = ", ".join(str(label) for label in labels)
            raise IndexError(f"geometric index {key} is not a direction of {self!r}; valid indices are {shown}")

        internal = labels.index(key)

        return internal

    def to_user(self, internal: int) -> Label:
        """
        Convert a storage slot to the label of its direction.

        Raises IndexError if the slot is outside 0 to dim - 1.
        """
        if not (0 <= internal < self.dim):
            raise IndexError(
                f"internal slot {internal} out of range for {self.dim}-dimensional metric; "
                f"valid slots are 0 to {self.dim - 1}"
            )

        user = self.labels[internal]

        return user

    def to_internal_multi(self, indices: tuple[Label, ...]) -> list[int]:
        """Convert a tuple of geometric indices to storage slots."""
        return [self.to_internal(n) for n in indices]

    # =========================================================================
    # Protocols
    # =========================================================================

    def __getitem__(self, index):
        """Index into the metric tensor: metric[a, b] -> g_{ab} (storage slots)."""
        return self.data[index]

    def __array__(self, dtype=None, copy=None):
        """NumPy array protocol for interoperability."""
        return self.data if dtype is None else self.data.astype(dtype)

    def __repr__(self) -> str:
        blocks = ", ".join(f"{block.kind}={block.size}" for block in self.blocks)
        convention = f"; {self.convention}" if self.convention is not None else ""

        return f"Metric({blocks}{convention})"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        """Two metrics are equal when their layouts and tensors agree."""
        if not isinstance(other, Metric):
            return NotImplemented

        result = self is other or (self.blocks == other.blocks and array_equal(self.data, other.data))

        return result

    def __hash__(self) -> int:
        return hash((self.blocks, self.data.tobytes()))

    def is_compatible(self, other: Metric) -> bool:
        """Check if two metrics are compatible for operations."""
        return self == other

    @classmethod
    def merge(cls, *metrics: Metric) -> Metric:
        """
        Merge metrics from multiple operands.

        All metrics must be identical. Raises ValueError if incompatible.
        """
        if not metrics:
            raise ValueError("At least one metric required")

        first = metrics[0]
        for m in metrics[1:]:
            if not first.is_compatible(m):
                raise ValueError(f"Incompatible metrics: {first} and {m}")

        return first


# =============================================================================
# Factories
# =============================================================================


def metric(
    *,
    time: int = 0,
    space: int = 0,
    projective: int = 0,
    modes: int = 0,
    convention: Convention = "mostly minus",
) -> Metric:
    """
    Declare a flat space by its blocks.

    Storage follows the blocks in the order (time, space, projective), except
    that a purely spatial space with a single projective direction is stored
    (projective, space). Directions are labeled by position with the first
    spatial direction always 1: time is 0, a leading projective direction is
    0, and trailing projective directions continue the spatial count. Modes
    take the place of the spatial block and are labeled 1, 1̇, 2, 2̇, ...

    Args:
        time: Number of time directions (at most 1 for now)
        space: Number of real spatial directions
        projective: Number of projective (null) directions
        modes: Number of complex modes, each two real directions
        convention: Sign convention when time is present. "mostly minus" is
            (+, -, -, -); "mostly plus" is (-, +, +, +). Without time, every
            spatial direction squares to +1.

    Returns:
        Cached Metric instance

    Examples:
        >>> metric(space=3).labels
        (1, 2, 3)
        >>> metric(space=3, projective=1).labels
        (0, 1, 2, 3)
        >>> metric(time=1, space=3, projective=1).labels
        (0, 1, 2, 3, 4)
        >>> metric(time=1, space=3, convention="mostly plus").signature
        (3, 1, 0)
    """
    return _metric(time, space, projective, modes, convention)


@lru_cache(maxsize=None)
def _metric(time: int, space: int, projective: int, modes: int, convention: str) -> Metric:
    if convention not in _CONVENTIONS:
        raise ValueError(f"Unknown convention {convention!r}; use one of {_CONVENTIONS}")

    if min(time, space, projective, modes) < 0:
        raise ValueError("block sizes must be non-negative")

    if time > 1:
        raise NotImplementedError("spaces with more than one time direction are not supported yet")

    if space > 0 and modes > 0:
        raise ValueError("a space has either a real spatial block or modes, not both")

    if time + space + projective + modes == 0:
        raise ValueError("a space needs at least one direction")

    spatial = Block("modes", modes) if modes > 0 else Block("space", space)
    if time == 0 and projective == 1:
        order = (Block("projective", projective), spatial)
    else:
        order = (Block("time", time), spatial, Block("projective", projective))

    blocks = tuple(block for block in order if block.size > 0)

    time_sign = 1.0 if convention == "mostly minus" else -1.0
    spatial_sign = -time_sign if time > 0 else 1.0
    signs = {"time": time_sign, "space": spatial_sign, "modes": spatial_sign, "projective": 0.0}
    squares = [signs[block.kind] for block in blocks for _ in range(block.dim)]

    result = Metric(data=diag(squares), blocks=blocks)

    return result


def euclidean_metric(dim: int) -> Metric:
    """Euclidean ℝ^dim, the same as metric(space=dim)."""
    return metric(space=dim)


def pga_metric(dim: int) -> Metric:
    """Projective GA of Euclidean ℝ^dim, the same as metric(space=dim, projective=1)."""
    return metric(space=dim, projective=1)


def lorentzian_metric(dim: int) -> Metric:
    """Spacetime of total dimension dim in (+, -, -, -), the same as metric(time=1, space=dim - 1)."""
    return metric(time=1, space=dim - 1)
