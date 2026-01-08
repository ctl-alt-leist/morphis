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

from morphis.visualization.animated import AnimatedCanvas as AnimatedCanvas
from morphis.visualization.blades import (
    BladeStyle as BladeStyle,
    render_bivector as render_bivector,
    render_scalar as render_scalar,
    render_trivector as render_trivector,
    render_vector as render_vector,
    visualize_blade as visualize_blade,
    visualize_blades as visualize_blades,
)
from morphis.visualization.canvas import ArrowStyle as ArrowStyle, Canvas as Canvas, CurveStyle as CurveStyle
from morphis.visualization.contexts import (
    PGAStyle as PGAStyle,
    render_pga_line as render_pga_line,
    render_pga_plane as render_pga_plane,
    render_pga_point as render_pga_point,
    visualize_pga_blade as visualize_pga_blade,
    visualize_pga_scene as visualize_pga_scene,
)
from morphis.visualization.drawing import (
    draw_basis_blade as draw_basis_blade,
    draw_blade as draw_blade,
    draw_coordinate_basis as draw_coordinate_basis,
    factor_blade as factor_blade,
)
from morphis.visualization.operations import (
    OperationStyle as OperationStyle,
    render_join as render_join,
    render_meet as render_meet,
    render_meet_join as render_meet_join,
    render_with_dual as render_with_dual,
)
from morphis.visualization.projection import (
    ProjectionConfig as ProjectionConfig,
    get_projection_axes as get_projection_axes,
    project_blade as project_blade,
)
from morphis.visualization.theme import (
    CHALK as CHALK,
    DEFAULT_THEME as DEFAULT_THEME,
    MIDNIGHT as MIDNIGHT,
    OBSIDIAN as OBSIDIAN,
    PAPER as PAPER,
    Color as Color,
    Palette as Palette,
    Theme as Theme,
    get_theme as get_theme,
    list_themes as list_themes,
)
from morphis.visualization.transforms import (
    AnimationSegment as AnimationSegment,
    AnimationSequence as AnimationSequence,
    BladeTransform as BladeTransform,
    ease_in_out_cubic as ease_in_out_cubic,
    ease_in_out_sine as ease_in_out_sine,
    ease_in_quad as ease_in_quad,
    ease_linear as ease_linear,
    ease_out_quad as ease_out_quad,
)
