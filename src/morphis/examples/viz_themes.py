"""
Visualization Themes Demo

Showcases all four themes with the same geometric content:
- Basis vectors (e1, e2, e3)
- A set of arrows cycling through the palette
- A parametric curve (helix)
- Sample points

Each theme opens in its own window for side-by-side comparison.
"""

from numpy import cos, linspace, pi, sin, stack

from morphis.visualization import Canvas, list_themes


# =============================================================================
# Demo Geometry
# =============================================================================


def helix(radius: float = 0.3, pitch: float = 0.15, turns: float = 2.0, n: int = 100):
    """Generate helix points centered at origin."""
    t = linspace(0, turns * 2 * pi, n)
    x = radius * cos(t)
    y = radius * sin(t)
    z = pitch * t

    return stack([x, y, z], axis=1)


def spiral_arrows(n: int = 8, radius: float = 0.8, height: float = 1.0):
    """Generate arrow starts and directions in a spiral pattern."""
    t = linspace(0, 2 * pi, n, endpoint=False)
    h = linspace(0, height, n)

    starts = stack(
        [
            radius * cos(t),
            radius * sin(t),
            h,
        ],
        axis=1,
    )

    # Arrows point outward and slightly up
    directions = stack(
        [
            0.3 * cos(t),
            0.3 * sin(t),
            0.15 * (1 + sin(t)),
        ],
        axis=1,
    )

    return starts, directions


def sample_points(n: int = 12):
    """Generate sample points on a sphere."""
    from numpy import arccos

    indices = linspace(0, 1, n, endpoint=False)
    phi = 2 * pi * indices
    theta = arccos(1 - 2 * indices)

    r = 0.5
    x = r * sin(theta) * cos(phi) + 1.0
    y = r * sin(theta) * sin(phi) + 0.5
    z = r * cos(theta) + 0.5

    return stack([x, y, z], axis=1)


# =============================================================================
# Demo Runner
# =============================================================================


def demo_theme(theme_name: str, block: bool = True) -> Canvas:
    """Create a visualization showcasing one theme."""
    canvas = Canvas(
        theme=theme_name,
        title=f"Theme: {theme_name}",
        show_basis=True,
    )

    # Arrows cycling through palette colors
    starts, directions = spiral_arrows(n=6)
    for k in range(len(starts)):
        canvas.arrow(starts[k], directions[k])

    # Curve using next palette color
    curve_points = helix(radius=0.25, pitch=0.12, turns=2.5)
    curve_points[:, 0] += 0.6
    curve_points[:, 1] += 0.6
    canvas.curve(curve_points)

    # Points using next palette color
    pts = sample_points(n=8)
    canvas.points(pts, radius=0.025)

    # Single accent arrow
    canvas.arrow(
        [0.0, 0.0, 0.0],
        [0.9, 0.4, 0.6],
        color=canvas.theme.accent,
        shaft_radius=0.02,
    )

    # Muted reference plane
    canvas.plane(
        center=[0.5, 0.5, 0.0],
        normal=[0, 0, 1],
        size=1.5,
        color=canvas.theme.muted,
        opacity=0.15,
    )

    canvas.camera(
        position=(2.5, -1.5, 1.8),
        focal_point=(0.5, 0.4, 0.4),
    )

    canvas.show(block=block)

    return canvas


def main():
    """Launch all theme demos sequentially."""
    print()
    print("=" * 50)
    print("  VISUALIZATION THEME SHOWCASE")
    print("=" * 50)
    print()

    themes = list_themes()
    print(f"Available themes: {', '.join(themes)}")
    print()
    print("Close each window to see the next theme.")
    print()

    for theme_name in themes:
        print(f"Showing {theme_name}...")
        demo_theme(theme_name, block=True)

    print()
    print("All themes shown.")
    print()


if __name__ == "__main__":
    main()
