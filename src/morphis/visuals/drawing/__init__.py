"""
Drawing submodule for blade visualization.

Contains mesh creation and rendering utilities for geometric algebra objects.
"""

from morphis.visuals.drawing.vectors import (
    VectorStyle as VectorStyle,
    create_blade_mesh as create_blade_mesh,
    create_frame_mesh as create_frame_mesh,
    draw_blade as draw_blade,
    draw_coordinate_basis as draw_coordinate_basis,
    render_bivector as render_bivector,
    render_trivector as render_trivector,
    render_vector as render_vector,
    visualize_blade as visualize_blade,
)
