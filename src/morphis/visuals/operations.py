"""
Visualization of Geometric Algebra Operations

Rendering functions for visualizing meet, join, and other GA operations.
Shows inputs and results together to illustrate geometric relationships.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from morphis.elements.vector import Vector
from morphis.operations.subspaces import meet
from morphis.visuals.canvas import Canvas
from morphis.visuals.drawing.vectors import VectorStyle, visualize_blade
from morphis.visuals.projection import ProjectionConfig
from morphis.visuals.theme import Color


# =============================================================================
# Operation Visualization Style
# =============================================================================


class OperationStyle(BaseModel):
    """Style parameters for operation visualization."""

    model_config = ConfigDict(frozen=True)

    # Input blade styling
    input_opacity: float = 0.4
    input_color_1: Color | None = None
    input_color_2: Color | None = None

    # Result blade styling
    result_opacity: float = 0.9
    result_color: Color | None = None

    # Common parameters
    scale: float = 1.0


# =============================================================================
# Join (Wedge) Visualization
# =============================================================================


def render_join(
    u: Vector,
    v: Vector,
    canvas: Canvas | None = None,
    projection: ProjectionConfig | None = None,
    style: OperationStyle | None = None,
) -> Canvas:
    """
    Visualize the join (wedge product) of two blades.

    Shows:
    - Input blades u and v in muted colors
    - Result u /\\ v highlighted
    - Visual indication of how inputs span the result

    The join represents the smallest subspace containing both inputs.

    Args:
        u, v: Input blades
        canvas: Existing canvas or None
        projection: For d > 3 blades
        style: Operation-specific styling

    Returns:
        Canvas with rendered operation
    """
    if canvas is None:
        canvas = Canvas(show_basis=True)

    style = style or OperationStyle()

    # Compute join
    result = u ^ v

    # Colors for inputs and result
    color_1 = style.input_color_1 or canvas.theme.palette[0]
    color_2 = style.input_color_2 or canvas.theme.palette[1]
    result_color = style.result_color or canvas.theme.accent

    # Render input u
    u_style = VectorStyle(
        color=color_1,
        opacity=style.input_opacity,
        scale=style.scale,
    )
    visualize_blade(u, canvas, projection=projection, style=u_style)

    # Render input v
    v_style = VectorStyle(
        color=color_2,
        opacity=style.input_opacity,
        scale=style.scale,
    )
    visualize_blade(v, canvas, projection=projection, style=v_style)

    # Render result
    result_style = VectorStyle(
        color=result_color,
        opacity=style.result_opacity,
        scale=style.scale,
    )
    visualize_blade(result, canvas, projection=projection, style=result_style)

    return canvas


# =============================================================================
# Meet Visualization
# =============================================================================


def render_meet(
    u: Vector,
    v: Vector,
    canvas: Canvas | None = None,
    projection: ProjectionConfig | None = None,
    style: OperationStyle | None = None,
) -> Canvas:
    """
    Visualize the meet (intersection) of two blades.

    Shows:
    - Input blades u and v in muted colors
    - Result meet(u, v) highlighted
    - Visual indication of how result is contained in both inputs

    The meet represents the intersection of two subspaces.

    Args:
        u, v: Input blades
        canvas: Existing canvas or None
        projection: For d > 3 blades
        style: Operation-specific styling

    Returns:
        Canvas with rendered operation
    """
    if canvas is None:
        canvas = Canvas(show_basis=True)

    style = style or OperationStyle()

    # Compute meet
    result = meet(u, v)

    # Colors
    color_1 = style.input_color_1 or canvas.theme.palette[0]
    color_2 = style.input_color_2 or canvas.theme.palette[1]
    result_color = style.result_color or canvas.theme.accent

    # Render input u
    u_style = VectorStyle(
        color=color_1,
        opacity=style.input_opacity,
        scale=style.scale,
    )
    visualize_blade(u, canvas, projection=projection, style=u_style)

    # Render input v
    v_style = VectorStyle(
        color=color_2,
        opacity=style.input_opacity,
        scale=style.scale,
    )
    visualize_blade(v, canvas, projection=projection, style=v_style)

    # Render result
    result_style = VectorStyle(
        color=result_color,
        opacity=style.result_opacity,
        scale=style.scale,
    )
    visualize_blade(result, canvas, projection=projection, style=result_style)

    return canvas


# =============================================================================
# Combined Meet/Join Visualization
# =============================================================================


def render_meet_join(
    u: Vector,
    v: Vector,
    canvas: Canvas | None = None,
    show: Literal["meet", "join", "both"] = "both",
    projection: ProjectionConfig | None = None,
    style: OperationStyle | None = None,
) -> Canvas:
    """
    Visualize meet and/or join operations between two blades.

    Args:
        u, v: Input blades
        canvas: Existing canvas or None
        show: Which operations to show ('meet', 'join', or 'both')
        projection: For d > 3 blades
        style: Operation-specific styling

    Returns:
        Canvas with rendered operation(s)
    """
    if canvas is None:
        canvas = Canvas(show_basis=True)

    style = style or OperationStyle()

    # Colors
    color_1 = style.input_color_1 or canvas.theme.palette[0]
    color_2 = style.input_color_2 or canvas.theme.palette[1]
    meet_color = canvas.theme.e1  # Red family for meet
    join_color = canvas.theme.e3  # Blue family for join

    # Render inputs
    u_style = VectorStyle(
        color=color_1,
        opacity=style.input_opacity,
        scale=style.scale,
    )
    visualize_blade(u, canvas, projection=projection, style=u_style)

    v_style = VectorStyle(
        color=color_2,
        opacity=style.input_opacity,
        scale=style.scale,
    )
    visualize_blade(v, canvas, projection=projection, style=v_style)

    # Render meet
    if show in ("meet", "both"):
        meet_result = meet(u, v)
        meet_style = VectorStyle(
            color=meet_color,
            opacity=style.result_opacity,
            scale=style.scale * 0.9,
        )
        visualize_blade(meet_result, canvas, projection=projection, style=meet_style)

    # Render join
    if show in ("join", "both"):
        join_result = u ^ v
        join_style = VectorStyle(
            color=join_color,
            opacity=style.result_opacity * 0.7,
            scale=style.scale * 1.1,
        )
        visualize_blade(join_result, canvas, projection=projection, style=join_style)

    return canvas


# =============================================================================
# Dual Visualization
# =============================================================================


def render_with_dual(
    blade: Vector,
    canvas: Canvas | None = None,
    dual_type: Literal["right", "left", "hodge"] = "right",
    projection: ProjectionConfig | None = None,
    style: OperationStyle | None = None,
) -> Canvas:
    """
    Visualize a blade alongside its dual.

    Shows the orthogonality relationship between a blade and its complement.

    Args:
        blade: Input blade
        canvas: Existing canvas or None
        dual_type: Type of dual ('right', 'left', 'hodge')
        projection: For d > 3 blades
        style: Operation-specific styling

    Returns:
        Canvas with blade and dual rendered
    """
    from morphis.operations.duality import hodge_dual, left_complement, right_complement

    if canvas is None:
        canvas = Canvas(show_basis=True)

    style = style or OperationStyle()

    # Compute dual
    if dual_type == "right":
        dual = right_complement(blade)
    elif dual_type == "left":
        dual = left_complement(blade)
    else:
        dual = hodge_dual(blade)

    # Colors
    blade_color = style.input_color_1 or canvas.theme.palette[0]
    dual_color = style.input_color_2 or canvas.theme.palette[3]

    # Render original
    blade_style = VectorStyle(
        color=blade_color,
        opacity=style.result_opacity,
        scale=style.scale,
    )
    visualize_blade(blade, canvas, projection=projection, style=blade_style)

    # Render dual
    dual_style = VectorStyle(
        color=dual_color,
        opacity=style.input_opacity,
        scale=style.scale * 0.8,
    )
    visualize_blade(dual, canvas, projection=projection, style=dual_style)

    return canvas
