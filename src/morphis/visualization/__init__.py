"""
Visualization Module

3D visualization tools built on PyVista with themed color palettes.

Quick start - Canvas primitives:
    from morphis.visualization import Canvas

    canvas = Canvas("obsidian")  # or "paper", "midnight", "chalk"
    canvas.arrow([0, 0, 0], [1, 0.5, 0.3])
    canvas.curve(helix_points)
    canvas.show()

Blade visualization:
    from morphis.visualization import visualize_blade, visualize_blades
    from morphis.ga.model import vector_blade, bivector_blade

    v = vector_blade([1, 2, 3])
    canvas = visualize_blade(v)
    canvas.show()

    # Bivector with different modes
    B = bivector_blade(...)
    canvas = visualize_blade(B, mode='circle')  # or 'parallelogram', 'plane'

PGA visualization:
    from morphis.visualization import visualize_pga_blade
    from morphis.geometry.projective import point, line

    p = point([1, 2, 3])
    canvas = visualize_pga_blade(p)  # Renders at Euclidean location

Operations visualization:
    from morphis.visualization import render_meet_join

    canvas = render_meet_join(u, v, show='both')

Themes:
    - obsidian: Dark with warm undertones
    - paper: Light with warm undertones (publication-ready)
    - midnight: Deep dark with cool blue cast
    - chalk: Light with cool undertones
"""

from morphis.visualization.blades import (
    BladeStyle,
    render_bivector,
    render_scalar,
    render_trivector,
    render_vector,
    visualize_blade,
    visualize_blades,
)
from morphis.visualization.canvas import ArrowStyle, Canvas, CurveStyle
from morphis.visualization.drawing import (
    draw_basis_blade,
    draw_blade,
    draw_coordinate_basis,
    factor_blade,
)
from morphis.visualization.transforms import (
    AnimationSequence,
    AnimationSegment,
    BladeTransform,
    ease_in_out_cubic,
    ease_in_out_sine,
    ease_in_quad,
    ease_linear,
    ease_out_quad,
)
from morphis.visualization.animated import AnimatedCanvas
from morphis.visualization.contexts import (
    PGAStyle,
    render_pga_line,
    render_pga_plane,
    render_pga_point,
    visualize_pga_blade,
    visualize_pga_scene,
)
from morphis.visualization.operations import (
    OperationStyle,
    render_join,
    render_meet,
    render_meet_join,
    render_with_dual,
)
from morphis.visualization.projection import (
    ProjectionConfig,
    get_projection_axes,
    project_blade,
)
from morphis.visualization.theme import (
    CHALK,
    DEFAULT_THEME,
    MIDNIGHT,
    OBSIDIAN,
    PAPER,
    Color,
    Palette,
    Theme,
    get_theme,
    list_themes,
)
