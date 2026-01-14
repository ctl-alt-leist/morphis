"""
Drawing submodule for blade visualization.

Contains mesh creation and rendering utilities for geometric algebra objects.
"""

from morphis.visuals.drawing.blades import (
    BladeStyle,
    create_blade_mesh,
    create_frame_mesh,
    draw_blade,
    draw_coordinate_basis,
    render_bivector,
    render_trivector,
    render_vector,
    visualize_blade,
)


__all__ = [
    "BladeStyle",
    "create_blade_mesh",
    "create_frame_mesh",
    "draw_blade",
    "draw_coordinate_basis",
    "render_bivector",
    "render_trivector",
    "render_vector",
    "visualize_blade",
]
