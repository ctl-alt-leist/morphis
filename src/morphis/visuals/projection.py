"""
Projection Utilities for High-Dimensional Vectors

Tools for projecting d-dimensional blades to 3D (or 2D) for visualization.
Supports configurable axis selection and different projection methods.

Projection axes are user-facing geometric indices, the same indices used by
`basis_vector` and `.on[...]`. They are translated to internal storage slots
through the element's `Metric`, so `(1, 2, 3)` is always x, y, z: in Euclidean
space index 0 is forbidden, while in Lorentzian and PGA space index 0 is the
time or ideal direction.
"""

from numbers import Integral
from typing import Literal

from numpy import abs as np_abs, argsort, zeros
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from morphis.elements.metric import Metric, euclidean_metric
from morphis.elements.vector import Vector


# The first three spatial directions. x is index 1 in every signature, so this
# default is valid for Euclidean, Lorentzian, and PGA metrics alike.
DEFAULT_PROJECTION: tuple[int, int, int] = (1, 2, 3)


def validate_projection_axes(axes: tuple[int, ...]) -> tuple[int, int, int]:
    """
    Check that projection axes are exactly three integers and return them as a tuple.

    Range checking needs a metric and happens in `projection_slots`.
    """
    axes = tuple(axes)
    if len(axes) != 3 or not all(isinstance(axis, Integral) for axis in axes):
        raise ValueError(f"Projection requires exactly 3 integer geometric indices, got {axes!r}")

    axes = tuple(int(axis) for axis in axes)

    return axes


def projection_slots(axes: tuple[int, ...], metric: Metric) -> tuple[int, ...]:
    """
    Translate user-facing projection axes to internal storage slots.

    Routes through `Metric.to_internal_multi`, so an index outside the metric's
    valid range (for example 0 in a Euclidean metric) raises IndexError.
    """
    slots = tuple(metric.to_internal_multi(axes))

    return slots


def basis_labels(axes: tuple[int, ...]) -> tuple[str, ...]:
    """
    Basis-axis labels for user-facing geometric indices.

    The label carries the same index the user addressed: axes (2, 3, 4) give
    e_2, e_3, e_4.
    """
    labels = tuple(f"$\\mathbf{{e}}_{{{axis}}}$" for axis in axes)

    return labels


class ProjectionConfig(BaseModel):
    """
    Configuration for projecting high-dimensional blades to 3D.

    Attributes:
        axes: User-facing geometric indices to project onto, e.g. (1, 2, 3) for
            x, y, z or (2, 3, 4) for e_2, e_3, e_4. Translated per blade through
            its metric. None selects axes automatically (principal method).
        method: Projection method
            - 'slice': Take specified axes directly
            - 'principal': Choose axes with largest components
        target_dim: Target dimension (default 3)
    """

    model_config = ConfigDict(frozen=True)

    axes: tuple[int, ...] | None = None
    method: Literal["slice", "principal"] = "slice"
    target_dim: int = 3


def _extract_principal_axes(data: NDArray, grade: int, target_dim: int, metric: Metric) -> tuple[int, ...]:
    """
    Find the internal storage slots with largest component magnitudes.

    For vectors: find axes with largest absolute values.
    For bivectors: find axes that span the largest plane components.
    For other grades: the first target_dim spatial directions.

    Returns internal 0-based slots; callers report them to users via `to_user`.
    """
    if grade == 1:
        # Sum absolute values across collection dims, find largest
        flat_data = data.reshape(-1, data.shape[-1])
        magnitudes = np_abs(flat_data).sum(axis=0)
        top_indices = argsort(magnitudes)[-target_dim:]
        return tuple(sorted(top_indices))

    elif grade == 2:
        # For bivectors, find pairs with largest components
        dim = data.shape[-1]
        flat_data = data.reshape(-1, dim, dim)

        # Sum |B^{mn}| for each m,n
        pair_mags = np_abs(flat_data).sum(axis=0)

        # Find axes that appear in largest pairs
        axis_scores = zeros(dim)
        for m in range(dim):
            for n in range(dim):
                if m != n:
                    axis_scores[m] += pair_mags[m, n]
                    axis_scores[n] += pair_mags[m, n]

        top_indices = argsort(axis_scores)[-target_dim:]
        return tuple(sorted(top_indices))

    else:
        # Default: the first target_dim spatial directions (e_1, e_2, ...)
        spatial = tuple(range(1, target_dim + 1))
        return projection_slots(spatial, metric)


def project_vector(blade: Vector, config: ProjectionConfig) -> Vector:
    """
    Project a vector blade to lower dimension.

    Args:
        blade: Grade-1 blade with dim > target_dim
        config: Projection configuration

    Returns:
        Vector with dim=target_dim, projected vector components
    """
    if blade.grade != 1:
        raise ValueError(f"project_vector requires grade-1, got {blade.grade}")

    target_dim = config.target_dim
    if blade.dim <= target_dim:
        return blade

    if config.method == "principal" or config.axes is None:
        axes = _extract_principal_axes(blade.data, 1, target_dim, blade.metric)
    else:
        axes = projection_slots(config.axes[:target_dim], blade.metric)

    projected_data = blade.data[..., list(axes)]

    return Vector(
        data=projected_data,
        grade=1,
        metric=euclidean_metric(target_dim),
        collection=blade.collection,
    )


def project_bivector(blade: Vector, config: ProjectionConfig) -> Vector:
    """
    Project a bivector blade to lower dimension.

    Extracts the bivector components in the subspace spanned by selected axes.

    Args:
        blade: Grade-2 blade with dim > target_dim
        config: Projection configuration

    Returns:
        Vector with dim=target_dim, projected bivector components
    """
    if blade.grade != 2:
        raise ValueError(f"project_bivector requires grade-2, got {blade.grade}")

    target_dim = config.target_dim
    if blade.dim <= target_dim:
        return blade

    if config.method == "principal" or config.axes is None:
        axes = _extract_principal_axes(blade.data, 2, target_dim, blade.metric)
    else:
        axes = projection_slots(config.axes[:target_dim], blade.metric)

    # Extract submatrix for selected axes
    axes_list = list(axes)
    collection_shape = blade.collection
    projected_shape = collection_shape + (target_dim, target_dim)
    projected_data = zeros(projected_shape, dtype=blade.data.dtype)

    for new_m, old_m in enumerate(axes_list):
        for new_n, old_n in enumerate(axes_list):
            projected_data[..., new_m, new_n] = blade.data[..., old_m, old_n]

    return Vector(
        data=projected_data,
        grade=2,
        metric=euclidean_metric(target_dim),
        collection=blade.collection,
    )


def project_trivector(blade: Vector, config: ProjectionConfig) -> Vector:
    """
    Project a trivector blade to 3D.

    Args:
        blade: Grade-3 blade with dim > 3
        config: Projection configuration

    Returns:
        Vector with dim=3, projected trivector components
    """
    if blade.grade != 3:
        raise ValueError(f"project_trivector requires grade-3, got {blade.grade}")

    target_dim = config.target_dim
    if blade.dim <= target_dim:
        return blade

    if config.method == "principal" or config.axes is None:
        axes = _extract_principal_axes(blade.data, 3, target_dim, blade.metric)
    else:
        axes = projection_slots(config.axes[:target_dim], blade.metric)

    axes_list = list(axes)
    collection_shape = blade.collection
    projected_shape = collection_shape + (target_dim, target_dim, target_dim)
    projected_data = zeros(projected_shape, dtype=blade.data.dtype)

    for new_a, old_a in enumerate(axes_list):
        for new_b, old_b in enumerate(axes_list):
            for new_c, old_c in enumerate(axes_list):
                projected_data[..., new_a, new_b, new_c] = blade.data[..., old_a, old_b, old_c]

    return Vector(
        data=projected_data,
        grade=3,
        metric=euclidean_metric(target_dim),
        collection=blade.collection,
    )


def project_quadvector(blade: Vector, config: ProjectionConfig) -> Vector:
    """
    Project a quadvector (4-blade) to lower dimension.

    Args:
        blade: Grade-4 blade with dim > target_dim
        config: Projection configuration

    Returns:
        Vector with dim=target_dim, projected quadvector components
    """
    if blade.grade != 4:
        raise ValueError(f"project_quadvector requires grade-4, got {blade.grade}")

    target_dim = config.target_dim
    if blade.dim <= target_dim:
        return blade

    if config.method == "principal" or config.axes is None:
        axes = _extract_principal_axes(blade.data, 4, target_dim, blade.metric)
    else:
        axes = projection_slots(config.axes[:target_dim], blade.metric)

    axes_list = list(axes)
    collection_shape = blade.collection
    projected_shape = collection_shape + (target_dim,) * 4
    projected_data = zeros(projected_shape, dtype=blade.data.dtype)

    for new_a, old_a in enumerate(axes_list):
        for new_b, old_b in enumerate(axes_list):
            for new_c, old_c in enumerate(axes_list):
                for new_d, old_d in enumerate(axes_list):
                    projected_data[..., new_a, new_b, new_c, new_d] = blade.data[..., old_a, old_b, old_c, old_d]

    return Vector(
        data=projected_data,
        grade=4,
        metric=euclidean_metric(target_dim),
        collection=blade.collection,
    )


def project_blade(blade: Vector, config: ProjectionConfig | None = None) -> Vector:
    """
    Project a blade to lower dimension for visualization.

    Automatically selects appropriate projection based on grade.

    Args:
        blade: Vector of any grade
        config: Projection configuration (uses defaults if None)

    Returns:
        Projected blade with dim=target_dim (or original if already small enough)
    """
    if config is None:
        config = ProjectionConfig()

    target_dim = config.target_dim

    if blade.dim <= target_dim:
        return blade

    if blade.grade == 0:
        # Scalars don't depend on dimension for visualization
        return Vector(
            data=blade.data.copy(),
            grade=0,
            metric=euclidean_metric(target_dim),
            collection=blade.collection,
        )
    elif blade.grade == 1:
        return project_vector(blade, config)
    elif blade.grade == 2:
        return project_bivector(blade, config)
    elif blade.grade == 3:
        return project_trivector(blade, config)
    elif blade.grade == 4:
        return project_quadvector(blade, config)
    else:
        # For higher grades, project component-wise
        # This is a simplification; full projection is complex
        raise NotImplementedError(
            f"Projection for grade-{blade.grade} blades not yet implemented. "
            f"Consider visualizing the dual (grade {blade.dim - blade.grade})."
        )


def get_projection_axes(blade: Vector, config: ProjectionConfig | None = None) -> tuple[int, ...]:
    """
    Get the user-facing geometric indices that would be used for projection.

    Useful for labeling visualizations with which axes are shown. The result
    uses the same convention as `basis_vector` and `.on[...]`, so it can be
    passed straight to `basis_labels`.
    """
    if config is None:
        config = ProjectionConfig()

    target_dim = config.target_dim
    metric = blade.metric

    if blade.dim <= target_dim:
        slots = tuple(range(blade.dim))
    elif config.axes is not None and config.method == "slice":
        slots = projection_slots(config.axes[:target_dim], metric)
    else:
        slots = _extract_principal_axes(blade.data, blade.grade, target_dim, blade.metric)

    axes = tuple(metric.to_user(int(slot)) for slot in slots)

    return axes
