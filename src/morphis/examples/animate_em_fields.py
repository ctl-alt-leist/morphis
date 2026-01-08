"""
Electromagnetic Field Animation

Visualizes oscillating E and B field vectors at two sensor locations near a
multi-circuit three-phase transmission line, with accompanying time series plots.

Physical setup:
- Two circuits (C1, C2) with three phases each (120 degree separation)
- Operating frequency: 60 Hz (depicted as 1 Hz for visibility)
- C1: Constant power flow direction
- C2: Power flow reverses at t=10s

Run with: uv run python -m morphis.examples.em_field_animation
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyvista as pv
import vtk


# =============================================================================
# Color Palette (Marco Bucci inspired - temperature shifts)
# =============================================================================

BACKGROUND = (0.12, 0.13, 0.14)  # Deep charcoal with warm undertones

# E field: Bright blue (both locations)
E_COLOR = (0.35, 0.65, 1.0)  # Vibrant blue
E_COLOR_LIGHT = (0.55, 0.78, 1.0)  # Lighter blue for Bx in time series
E_COLOR_DARK = (0.25, 0.50, 0.85)  # Darker blue for Bz in time series

# B field: Bright red (both locations)
B_COLOR = (1.0, 0.40, 0.35)  # Vibrant red
B_COLOR_LIGHT = (1.0, 0.60, 0.55)  # Lighter red for Bx in time series
B_COLOR_MID = (1.0, 0.40, 0.35)  # Medium red for By in time series
B_COLOR_DARK = (0.85, 0.25, 0.25)  # Darker red for Bz in time series

# Poynting vector: Silver
S_COLOR = (0.78, 0.78, 0.82)  # Silver

# UI colors
LABEL_COLOR = (0.82, 0.84, 0.86)  # Light gray
GRID_COLOR = (0.25, 0.26, 0.28)  # Subtle grid
TIME_MARKER_COLOR = (0.95, 0.85, 0.40)  # Amber accent


# =============================================================================
# Circuit Parameters
# =============================================================================


@dataclass
class CircuitParams:
    """Parameters for a single transmission line circuit."""

    E0: float  # Electric field amplitude
    B0: float  # Magnetic field amplitude
    psi_E: float  # E field phase offset
    psi_B: float  # B field phase offset
    # E field component phase offsets (for 3D E vector)
    E_alpha_x: float
    E_alpha_y: float
    E_alpha_z: float
    # B field component phase offsets
    B_alpha_x: float
    B_alpha_y: float
    B_alpha_z: float


# Circuit 1: Constant power flow
CIRCUIT_1 = CircuitParams(
    E0=1.0,
    B0=0.5,
    psi_E=0.0,
    psi_B=np.pi / 6,  # 30 degree phase for visible power flow
    # E field: primarily vertical but with some horizontal components
    E_alpha_x=np.pi / 3,
    E_alpha_y=2 * np.pi / 3,
    E_alpha_z=0.0,  # Dominant component
    # B field component offsets
    B_alpha_x=0.0,
    B_alpha_y=np.pi / 2,
    B_alpha_z=np.pi,
)

# Circuit 2: Power reverses at t=10s
CIRCUIT_2 = CircuitParams(
    E0=0.8,
    B0=0.4,
    psi_E=np.pi / 4,
    psi_B=np.pi / 3,  # Initial phase (will shift by pi at t=10s)
    # E field component offsets
    E_alpha_x=np.pi / 2,
    E_alpha_y=np.pi,
    E_alpha_z=np.pi / 6,  # Dominant component
    # B field component offsets
    B_alpha_x=np.pi / 6,
    B_alpha_y=2 * np.pi / 3,
    B_alpha_z=5 * np.pi / 6,
)

# Animation parameters
OMEGA = 2 * np.pi  # 1 Hz depicted frequency (representing 60 Hz)
REAL_FREQUENCY = 60.0  # Hz - actual signal frequency
TOTAL_DURATION = 20.0  # seconds (animation time)
FPS = 30
POWER_REVERSAL_TIME = 10.0  # seconds (animation time)

# Time conversion: animation runs at 1 Hz, real signal is 60 Hz
# 1 animation second = 1/60 real second = 16.667 ms
TIME_SCALE_MS = 1000.0 / REAL_FREQUENCY  # ~16.667 ms per animation second

# Time series display window (middle third to reduce clutter)
TIME_WINDOW_START = 7.0  # animation seconds
TIME_WINDOW_END = 13.0  # animation seconds

# Arrow styling
SHAFT_RADIUS = 0.02  # Thinner for more elegant arrows
S_SHAFT_RADIUS = 0.028  # Slightly bolder for Poynting vector


# =============================================================================
# Field Computation
# =============================================================================


def phase_angles(t: float) -> Tuple[float, float, float]:
    """Compute phase angles for three phases A, B, C at time t."""
    phi_A = OMEGA * t
    phi_B = OMEGA * t - 2 * np.pi / 3
    phi_C = OMEGA * t - 4 * np.pi / 3
    return phi_A, phi_B, phi_C


def compute_E_field(
    t: float,
    params: CircuitParams,
    distance_weights: tuple = (1.0, 0.7, 0.5),
) -> np.ndarray:
    """
    Compute E field vector (Ex, Ey, Ez) for a circuit at time t.

    Each component has its own phase offset creating 3D oscillation.
    Asymmetric distance weighting prevents perfect three-phase cancellation.

    Args:
        t: Time in seconds
        params: Circuit parameters
        distance_weights: Relative contribution from each phase (A, B, C)
            based on distance to sensor. Unequal weights create net field.
    """
    phi_A, phi_B, phi_C = phase_angles(t)
    w_A, w_B, w_C = distance_weights

    E_x = (
        params.E0
        * 0.3
        * (  # Smaller horizontal components
            w_A * np.cos(phi_A + params.psi_E + params.E_alpha_x)
            + w_B * np.cos(phi_B + params.psi_E + params.E_alpha_x)
            + w_C * np.cos(phi_C + params.psi_E + params.E_alpha_x)
        )
    )

    E_y = (
        params.E0
        * 0.3
        * (
            w_A * np.cos(phi_A + params.psi_E + params.E_alpha_y)
            + w_B * np.cos(phi_B + params.psi_E + params.E_alpha_y)
            + w_C * np.cos(phi_C + params.psi_E + params.E_alpha_y)
        )
    )

    E_z = params.E0 * (  # Dominant vertical component
        w_A * np.cos(phi_A + params.psi_E + params.E_alpha_z)
        + w_B * np.cos(phi_B + params.psi_E + params.E_alpha_z)
        + w_C * np.cos(phi_C + params.psi_E + params.E_alpha_z)
    )

    return np.array([E_x, E_y, E_z])


def compute_B_field(
    t: float,
    params: CircuitParams,
    psi_B_override: float | None = None,
    distance_weights: tuple = (1.0, 0.7, 0.5),
) -> np.ndarray:
    """
    Compute B field vector (Bx, By, Bz) for a circuit at time t.

    Each component has its own phase offset creating 3D rotation.
    Asymmetric distance weighting prevents perfect three-phase cancellation.
    """
    phi_A, phi_B, phi_C = phase_angles(t)
    psi_B = psi_B_override if psi_B_override is not None else params.psi_B
    w_A, w_B, w_C = distance_weights

    B_x = params.B0 * (
        w_A * np.cos(phi_A + psi_B + params.B_alpha_x)
        + w_B * np.cos(phi_B + psi_B + params.B_alpha_x)
        + w_C * np.cos(phi_C + psi_B + params.B_alpha_x)
    )

    B_y = params.B0 * (
        w_A * np.cos(phi_A + psi_B + params.B_alpha_y)
        + w_B * np.cos(phi_B + psi_B + params.B_alpha_y)
        + w_C * np.cos(phi_C + psi_B + params.B_alpha_y)
    )

    B_z = params.B0 * (
        w_A * np.cos(phi_A + psi_B + params.B_alpha_z)
        + w_B * np.cos(phi_B + psi_B + params.B_alpha_z)
        + w_C * np.cos(phi_C + psi_B + params.B_alpha_z)
    )

    return np.array([B_x, B_y, B_z])


def compute_poynting_vector(E: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute Poynting vector S = E × B."""
    return np.cross(E, B)


def compute_total_fields(t: float, location: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute total E and B fields at a sensor location.

    Location 1: Closer to phase A of circuit 1
    Location 2: More centered, different distance pattern

    Circuit 2 phase reverses at t >= POWER_REVERSAL_TIME.
    """
    # Circuit 2 phase shift after power reversal
    c2_psi_B = CIRCUIT_2.psi_B
    if t >= POWER_REVERSAL_TIME:
        c2_psi_B = CIRCUIT_2.psi_B + np.pi

    # Location-dependent distance weights for three-phase contributions
    # These represent inverse-distance weighting to phase conductors A, B, C
    if location == 1:
        # Location 1: closest to phase A, farther from B and C
        dist_weights_c1 = (1.0, 0.6, 0.4)
        dist_weights_c2 = (0.7, 0.5, 0.8)
        circuit_mix = (1.0, 0.7)  # Circuit 1 dominates
    else:
        # Location 2: more centered, different geometry
        dist_weights_c1 = (0.5, 0.9, 0.6)
        dist_weights_c2 = (0.8, 0.6, 1.0)
        circuit_mix = (0.8, 0.9)  # Circuits more balanced

    w1, w2 = circuit_mix

    # Compute E fields (full 3D vector)
    E1 = compute_E_field(t, CIRCUIT_1, distance_weights=dist_weights_c1)
    E2 = compute_E_field(t, CIRCUIT_2, distance_weights=dist_weights_c2)
    E_total = w1 * E1 + w2 * E2

    # Compute B fields (full 3D vector)
    B1 = compute_B_field(t, CIRCUIT_1, distance_weights=dist_weights_c1)
    B2 = compute_B_field(t, CIRCUIT_2, psi_B_override=c2_psi_B, distance_weights=dist_weights_c2)
    B_total = w1 * B1 + w2 * B2

    return E_total, B_total


# =============================================================================
# Arrow Creation and Transform Utilities
# =============================================================================


def create_unit_arrow(
    shaft_radius: float = SHAFT_RADIUS,
    tip_ratio: float = 0.12,
    resolution: int = 24,
) -> pv.PolyData:
    """
    Create a unit arrow mesh pointing along +Z axis, base at origin.

    The arrow has length 1.0, from z=0 to z=1.
    Use VTK transforms to rotate and scale to target direction/length.
    """
    shaft_length = 1.0 * (1 - tip_ratio)
    tip_length = 1.0 * tip_ratio
    tip_radius = shaft_radius * 2.5

    # Shaft from z=0 to z=(1 - tip_length)
    shaft_start_z = 0.0
    shaft_end_z = 1.0 - tip_length

    shaft = pv.Cylinder(
        center=(0, 0, (shaft_start_z + shaft_end_z) / 2),
        direction=(0, 0, 1),
        radius=shaft_radius,
        height=shaft_length,
        resolution=resolution,
        capping=True,
    )

    tip = pv.Cone(
        center=(0, 0, shaft_end_z + tip_length / 2),
        direction=(0, 0, 1),
        height=tip_length,
        radius=tip_radius,
        resolution=resolution,
        capping=True,
    )

    return shaft.merge(tip)


def direction_to_rotation(direction: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute rotation angle and axis to rotate +Z to target direction.

    Returns:
        (angle_deg, axis): Angle in degrees and rotation axis
    """
    direction = np.asarray(direction, dtype=float)
    length = np.linalg.norm(direction)

    if length < 1e-10:
        return 0.0, np.array([0.0, 0.0, 1.0])

    direction_norm = direction / length
    z_axis = np.array([0.0, 0.0, 1.0])

    # Dot product gives cos(angle)
    cos_angle = np.dot(z_axis, direction_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Cross product gives rotation axis
    axis = np.cross(z_axis, direction_norm)
    axis_length = np.linalg.norm(axis)

    if axis_length < 1e-10:
        # Vectors are parallel or anti-parallel
        if cos_angle > 0:
            return 0.0, np.array([0.0, 0.0, 1.0])
        else:
            return 180.0, np.array([1.0, 0.0, 0.0])

    axis = axis / axis_length
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg, axis


def update_arrow_transform(transform: vtk.vtkTransform, direction: np.ndarray):
    """
    Update a VTK transform to point arrow in given direction with correct length.

    The arrow mesh is assumed to be a unit arrow along +Z with base at origin.
    """
    length = np.linalg.norm(direction)
    angle_deg, axis = direction_to_rotation(direction)

    transform.Identity()

    # Rotate from +Z to target direction first
    if abs(angle_deg) > 1e-6:
        transform.RotateWXYZ(angle_deg, axis[0], axis[1], axis[2])

    # Scale to target length (arrow base stays at origin)
    transform.Scale(length, length, length)


@dataclass
class AnimatedArrow:
    """Container for an arrow with its VTK transform for smooth animation."""

    mesh: pv.PolyData
    actor: vtk.vtkActor
    transform: vtk.vtkTransform

    def update(self, direction: np.ndarray):
        """Update arrow to point in new direction with correct length."""
        update_arrow_transform(self.transform, direction)


def create_animated_arrow(
    plotter: pv.Plotter,
    initial_direction: np.ndarray,
    color: Tuple[float, float, float],
    shaft_radius: float = SHAFT_RADIUS,
) -> AnimatedArrow:
    """
    Create an arrow that can be smoothly animated using VTK transforms.

    Returns an AnimatedArrow containing the mesh, actor, and transform.
    """
    mesh = create_unit_arrow(shaft_radius=shaft_radius)
    actor = plotter.add_mesh(mesh, color=color, smooth_shading=True)

    transform = vtk.vtkTransform()
    actor.SetUserTransform(transform)

    # Set initial direction
    update_arrow_transform(transform, initial_direction)

    return AnimatedArrow(mesh=mesh, actor=actor, transform=transform)


# =============================================================================
# Time Series Chart
# =============================================================================


def create_time_series_chart(
    t_array: np.ndarray,
    E_z_series: np.ndarray,
    B_series: np.ndarray,
    S_y_series: np.ndarray,
    location_label: str,
) -> pv.Chart2D:
    """
    Create a 2D chart for time series data.

    Shows E_z, all three B components (Bx, By, Bz), and S_y (power flow direction).
    Displays middle third of time range to reduce visual clutter.
    """
    chart = pv.Chart2D()

    # Scale colors to 0-255 for chart
    def to_255(c):
        return tuple(int(v * 255) for v in c)

    # Filter to time window (middle third)
    mask = (t_array >= TIME_WINDOW_START) & (t_array <= TIME_WINDOW_END)
    t_window = t_array[mask]
    # Convert to real milliseconds (animation at 1 Hz represents 60 Hz signal)
    t_window_ms = t_window * TIME_SCALE_MS
    E_z_window = E_z_series[mask]
    B_window = B_series[mask]
    S_y_window = S_y_series[mask]

    # Add E_z line (bright blue)
    chart.line(t_window_ms, E_z_window, color=to_255(E_COLOR), width=2.5, label="$E_z$")

    # Add B component lines (red shades)
    chart.line(t_window_ms, B_window[:, 0], color=to_255(B_COLOR_LIGHT), width=2.0, label="$B_x$")
    chart.line(t_window_ms, B_window[:, 1], color=to_255(B_COLOR_MID), width=2.0, label="$B_y$")
    chart.line(t_window_ms, B_window[:, 2], color=to_255(B_COLOR_DARK), width=2.0, label="$B_z$")

    # Add Poynting vector y-component (silver - power flow direction)
    chart.line(t_window_ms, S_y_window, color=to_255(S_COLOR), width=2.5, label="$S_y$")

    # Configure axes - ensure all text/lines are visible on dark background
    chart.x_axis.label = "Time (ms)"
    chart.y_axis.label = "Field (a.u.)"
    chart.x_axis.range = [TIME_WINDOW_START * TIME_SCALE_MS, TIME_WINDOW_END * TIME_SCALE_MS]

    # Set colors using VTK text properties (0-1 range for VTK)
    for axis in [chart.x_axis, chart.y_axis]:
        # Axis label text color
        axis.GetLabelProperties().SetColor(*LABEL_COLOR)
        # Tick label text color (uses same property as axis label in VTK)
        axis.GetTitleProperties().SetColor(*LABEL_COLOR)
        # Axis line color
        axis.pen.color = to_255(LABEL_COLOR)

    # Grid (subtle)
    chart.x_axis.grid = True
    chart.y_axis.grid = True
    for axis in [chart.x_axis, chart.y_axis]:
        axis.grid_pen.color = to_255(GRID_COLOR)

    # Add legend
    chart.legend_visible = True

    # Add title (with proper color for dark theme)
    chart.title = location_label
    chart.GetTitleProperties().SetColor(*LABEL_COLOR)

    return chart


# =============================================================================
# Main Animation
# =============================================================================


def precompute_time_series(t_array: np.ndarray) -> dict:
    """Pre-compute all field values for the animation."""
    n_frames = len(t_array)

    # Location 1 data
    E1_z_series = np.zeros(n_frames)  # Only E_z for time series
    B1_series = np.zeros((n_frames, 3))
    S1_y_series = np.zeros(n_frames)  # Poynting vector y-component (power flow direction)

    # Location 2 data
    E2_z_series = np.zeros(n_frames)  # Only E_z for time series
    B2_series = np.zeros((n_frames, 3))
    S2_y_series = np.zeros(n_frames)  # Poynting vector y-component (power flow direction)

    for i, t in enumerate(t_array):
        E1_vec, B1_vec = compute_total_fields(t, location=1)
        E2_vec, B2_vec = compute_total_fields(t, location=2)

        E1_z_series[i] = E1_vec[2]  # E_z component only for time series
        B1_series[i] = B1_vec
        S1_vec = compute_poynting_vector(E1_vec, B1_vec)
        S1_y_series[i] = S1_vec[1]  # y-component indicates power flow direction

        E2_z_series[i] = E2_vec[2]  # E_z component only for time series
        B2_series[i] = B2_vec
        S2_vec = compute_poynting_vector(E2_vec, B2_vec)
        S2_y_series[i] = S2_vec[1]  # y-component indicates power flow direction

    return {
        "E1_z": E1_z_series,
        "B1": B1_series,
        "S1_y": S1_y_series,
        "E2_z": E2_z_series,
        "B2": B2_series,
        "S2_y": S2_y_series,
    }


def _draw_axis_arrow(
    plotter: pv.Plotter,
    direction: np.ndarray,
    color: Tuple[float, float, float],
    shaft_radius: float = 0.004,
    tip_ratio: float = 0.08,
):
    """Draw a coordinate axis arrow from origin."""
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return

    dir_norm = direction / length
    tip_length = length * tip_ratio
    shaft_length = length - tip_length
    tip_radius = shaft_radius * 2.5

    shaft_end = dir_norm * shaft_length

    # Shaft cylinder
    shaft = pv.Cylinder(
        center=shaft_end / 2,
        direction=dir_norm,
        radius=shaft_radius,
        height=shaft_length,
        resolution=20,
        capping=True,
    )
    plotter.add_mesh(shaft, color=color, smooth_shading=True)

    # Tip cone
    tip = pv.Cone(
        center=shaft_end + dir_norm * (tip_length / 2),
        direction=dir_norm,
        height=tip_length,
        radius=tip_radius,
        resolution=20,
        capping=True,
    )
    plotter.add_mesh(tip, color=color, smooth_shading=True)


def setup_3d_subplot(plotter: pv.Plotter, location: int):
    """Set up a 3D subplot with coordinate axes matching trivector animation style."""
    axis_length = 0.8
    axis_color = (0.5, 0.52, 0.55)
    label_offset = 0.08

    # Axis directions and label offsets (outside positive octant)
    axes = [
        (np.array([axis_length, 0, 0]), "$e_1$", np.array([0, -1, -1])),
        (np.array([0, axis_length, 0]), "$e_2$", np.array([-1, 0, -1])),
        (np.array([0, 0, axis_length]), "$e_3$", np.array([-1, -1, 0])),
    ]

    for direction, label, offset_dir in axes:
        # Draw arrow
        _draw_axis_arrow(plotter, direction, axis_color)

        # Add label (offset outside positive octant)
        offset = offset_dir / np.linalg.norm(offset_dir) * label_offset
        label_pos = direction * 0.5 + offset
        plotter.add_point_labels(
            [label_pos],
            [label],
            font_size=12,
            text_color=LABEL_COLOR,
            point_size=0,
            shape=None,
            show_points=False,
            always_visible=True,
        )

    # Add location label
    loc_label = f"Location {location}"
    plotter.add_text(
        loc_label,
        position="upper_left",
        font_size=10,
        color=LABEL_COLOR,
    )

    # Set camera (10% more zoomed in)
    plotter.camera.position = (2.25, -1.8, 1.62)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)


def adjust_layout_widths(plotter: pv.Plotter):
    """
    Adjust subplot viewports for 1/3 - 2/3 width split.

    Default 2x2 layout gives 50-50 split. We want:
    - Column 0 (3D animation): 1/3 width
    - Column 1 (time series): 2/3 width
    """
    # Viewport format: (xmin, ymin, xmax, ymax) in normalized coords [0, 1]
    split = 1.0 / 3.0  # 1/3 for 3D animation

    # Row 0 (top): Location 1
    # Subplot (0, 0): 3D animation - left 1/3, top half
    plotter.renderers[0].SetViewport(0.0, 0.5, split, 1.0)
    # Subplot (0, 1): Time series - right 2/3, top half
    plotter.renderers[1].SetViewport(split, 0.5, 1.0, 1.0)

    # Row 1 (bottom): Location 2
    # Subplot (1, 0): 3D animation - left 1/3, bottom half
    plotter.renderers[2].SetViewport(0.0, 0.0, split, 0.5)
    # Subplot (1, 1): Time series - right 2/3, bottom half
    plotter.renderers[3].SetViewport(split, 0.0, 1.0, 0.5)


def main_interactive():
    """Run animation in interactive mode."""
    print()
    print("=" * 60)
    print("  ELECTROMAGNETIC FIELD ANIMATION")
    print("=" * 60)
    print()
    print("Physical setup:")
    print("  - Two circuits with three-phase power (120 deg separation)")
    print("  - Frequency: 1 Hz displayed (representing 60 Hz)")
    print("  - Circuit 2 power reverses at t=10s")
    print()
    print("Close window to exit.")
    print()

    # Pre-compute time series
    n_frames = int(TOTAL_DURATION * FPS)
    t_array = np.linspace(0, TOTAL_DURATION, n_frames)
    data = precompute_time_series(t_array)

    # Create plotter with 2x2 layout (10% smaller)
    plotter = pv.Plotter(
        shape=(2, 2),
        window_size=(1728, 972),
        title="EM Field Animation",
    )

    # Adjust layout for 1/3 - 2/3 split
    adjust_layout_widths(plotter)

    # Set background for all subplots
    plotter.set_background(BACKGROUND)

    # --- Subplot (0, 0): Location 1 - 3D vectors ---
    plotter.subplot(0, 0)
    setup_3d_subplot(plotter, location=1)

    # Create animated arrows with VTK transforms (blue E, red B, silver S)
    E1_vec, B1_vec = compute_total_fields(0, location=1)
    S1_vec = compute_poynting_vector(E1_vec, B1_vec)
    E1_arrow = create_animated_arrow(plotter, E1_vec, E_COLOR)
    B1_arrow = create_animated_arrow(plotter, B1_vec, B_COLOR)
    S1_arrow = create_animated_arrow(plotter, S1_vec, S_COLOR, shaft_radius=S_SHAFT_RADIUS)

    # --- Subplot (0, 1): Location 1 - Time series ---
    plotter.subplot(0, 1)
    chart1 = create_time_series_chart(
        t_array,
        data["E1_z"],
        data["B1"],
        data["S1_y"],
        "Location 1",
    )
    plotter.add_chart(chart1)

    # --- Subplot (1, 0): Location 2 - 3D vectors ---
    plotter.subplot(1, 0)
    setup_3d_subplot(plotter, location=2)

    E2_vec, B2_vec = compute_total_fields(0, location=2)
    S2_vec = compute_poynting_vector(E2_vec, B2_vec)
    E2_arrow = create_animated_arrow(plotter, E2_vec, E_COLOR)
    B2_arrow = create_animated_arrow(plotter, B2_vec, B_COLOR)
    S2_arrow = create_animated_arrow(plotter, S2_vec, S_COLOR, shaft_radius=S_SHAFT_RADIUS)

    # --- Subplot (1, 1): Location 2 - Time series ---
    plotter.subplot(1, 1)
    chart2 = create_time_series_chart(
        t_array,
        data["E2_z"],
        data["B2"],
        data["S2_y"],
        "Location 2",
    )
    plotter.add_chart(chart2)

    # Show and animate
    plotter.show(interactive_update=True, auto_close=False)

    import time

    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > TOTAL_DURATION:
            elapsed = TOTAL_DURATION

        # Compute current fields (full 3D vectors)
        E1_vec, B1_vec = compute_total_fields(elapsed, location=1)
        E2_vec, B2_vec = compute_total_fields(elapsed, location=2)
        S1_vec = compute_poynting_vector(E1_vec, B1_vec)
        S2_vec = compute_poynting_vector(E2_vec, B2_vec)

        # Update arrows via transforms (no mesh recreation - smooth!)
        E1_arrow.update(E1_vec)
        B1_arrow.update(B1_vec)
        S1_arrow.update(S1_vec)
        E2_arrow.update(E2_vec)
        B2_arrow.update(B2_vec)
        S2_arrow.update(S2_vec)

        plotter.render()
        time.sleep(1.0 / FPS)

        if elapsed >= TOTAL_DURATION:
            print("Animation complete!")
            break

    plotter.close()


def main_save(filename: str = "em_field_animation.mp4"):
    """Render animation to video file."""
    print()
    print("=" * 60)
    print("  ELECTROMAGNETIC FIELD ANIMATION - RENDER TO VIDEO")
    print("=" * 60)
    print()

    # Pre-compute time series
    n_frames = int(TOTAL_DURATION * FPS)
    t_array = np.linspace(0, TOTAL_DURATION, n_frames)
    data = precompute_time_series(t_array)

    print(f"Rendering {n_frames} frames to {filename}...")
    print()

    # Create plotter with 2x2 layout (10% smaller)
    plotter = pv.Plotter(
        shape=(2, 2),
        window_size=(1728, 972),
        off_screen=True,
    )

    # Adjust layout for 1/3 - 2/3 split
    adjust_layout_widths(plotter)

    plotter.set_background(BACKGROUND)

    # --- Subplot (0, 0): Location 1 - 3D vectors ---
    plotter.subplot(0, 0)
    setup_3d_subplot(plotter, location=1)

    E1_vec, B1_vec = compute_total_fields(0, location=1)
    S1_vec = compute_poynting_vector(E1_vec, B1_vec)
    E1_arrow = create_animated_arrow(plotter, E1_vec, E_COLOR)
    B1_arrow = create_animated_arrow(plotter, B1_vec, B_COLOR)
    S1_arrow = create_animated_arrow(plotter, S1_vec, S_COLOR, shaft_radius=S_SHAFT_RADIUS)

    # --- Subplot (0, 1): Location 1 - Time series ---
    plotter.subplot(0, 1)
    chart1 = create_time_series_chart(
        t_array,
        data["E1_z"],
        data["B1"],
        data["S1_y"],
        "Location 1",
    )
    plotter.add_chart(chart1)

    # --- Subplot (1, 0): Location 2 - 3D vectors ---
    plotter.subplot(1, 0)
    setup_3d_subplot(plotter, location=2)

    E2_vec, B2_vec = compute_total_fields(0, location=2)
    S2_vec = compute_poynting_vector(E2_vec, B2_vec)
    E2_arrow = create_animated_arrow(plotter, E2_vec, E_COLOR)
    B2_arrow = create_animated_arrow(plotter, B2_vec, B_COLOR)
    S2_arrow = create_animated_arrow(plotter, S2_vec, S_COLOR, shaft_radius=S_SHAFT_RADIUS)

    # --- Subplot (1, 1): Location 2 - Time series ---
    plotter.subplot(1, 1)
    chart2 = create_time_series_chart(
        t_array,
        data["E2_z"],
        data["B2"],
        data["S2_y"],
        "Location 2",
    )
    plotter.add_chart(chart2)

    # Initialize render window before recording (fixes cropping in off-screen mode)
    plotter.show(auto_close=False)

    # Start recording
    plotter.open_movie(filename, framerate=FPS, quality=9)

    # Render each frame
    for frame_idx, t in enumerate(t_array):
        # Compute current fields (full 3D vectors)
        E1_vec, B1_vec = compute_total_fields(t, location=1)
        E2_vec, B2_vec = compute_total_fields(t, location=2)
        S1_vec = compute_poynting_vector(E1_vec, B1_vec)
        S2_vec = compute_poynting_vector(E2_vec, B2_vec)

        # Update arrows via transforms (smooth, no jitter)
        E1_arrow.update(E1_vec)
        B1_arrow.update(B1_vec)
        S1_arrow.update(S1_vec)
        E2_arrow.update(E2_vec)
        B2_arrow.update(B2_vec)
        S2_arrow.update(S2_vec)

        # Write frame
        plotter.write_frame()

        # Progress update
        if (frame_idx + 1) % (FPS * 2) == 0:
            print(f"  {t:.1f}s / {TOTAL_DURATION:.1f}s ({frame_idx + 1}/{n_frames} frames)")

    plotter.close()
    print()
    print(f"Saved to {filename}")


# =============================================================================
# Measurement Analysis
# =============================================================================


@dataclass
class MeasurementStats:
    """RMS and average statistics for a measurement period."""

    E_rms: float  # RMS of E field magnitude
    B_rms: float  # RMS of B field magnitude
    S_y_avg: float  # Average of S_y (power flow direction indicator)
    period_label: str  # "Measurement 1" or "Measurement 2"
    time_range_ms: Tuple[float, float]  # Real time range in ms


def compute_measurement_stats(
    t_start: float,
    t_end: float,
    location: int,
    n_samples: int = 1000,
) -> MeasurementStats:
    """
    Compute field statistics for a measurement period.

    Args:
        t_start: Start time (animation seconds)
        t_end: End time (animation seconds)
        location: Sensor location (1 or 2)
        n_samples: Number of samples for integration

    Returns:
        MeasurementStats with RMS and average values
    """
    t_array = np.linspace(t_start, t_end, n_samples)

    E_mag_sq_sum = 0.0
    B_mag_sq_sum = 0.0
    S_y_sum = 0.0

    for t in t_array:
        E_vec, B_vec = compute_total_fields(t, location=location)
        S_vec = compute_poynting_vector(E_vec, B_vec)

        E_mag_sq_sum += np.dot(E_vec, E_vec)
        B_mag_sq_sum += np.dot(B_vec, B_vec)
        S_y_sum += S_vec[1]

    E_rms = np.sqrt(E_mag_sq_sum / n_samples)
    B_rms = np.sqrt(B_mag_sq_sum / n_samples)
    S_y_avg = S_y_sum / n_samples

    # Determine period label
    period_label = "Measurement 1" if t_start < POWER_REVERSAL_TIME else "Measurement 2"

    # Convert to real time (ms)
    time_range_ms = (t_start * TIME_SCALE_MS, t_end * TIME_SCALE_MS)

    return MeasurementStats(
        E_rms=E_rms,
        B_rms=B_rms,
        S_y_avg=S_y_avg,
        period_label=period_label,
        time_range_ms=time_range_ms,
    )


def print_measurement_summary():
    """Print a formatted summary of measurements for both periods and locations."""
    print()
    print("=" * 70)
    print("  ELECTROMAGNETIC FIELD MEASUREMENTS")
    print("  Dual-Circuit Three-Phase Transmission Line")
    print("=" * 70)
    print()
    print(f"  Signal frequency: {REAL_FREQUENCY:.0f} Hz")
    print(f"  Power reversal (Circuit 2): {POWER_REVERSAL_TIME * TIME_SCALE_MS:.1f} ms")
    print()

    # Compute stats for both periods and locations
    half_time = TOTAL_DURATION / 2

    stats = {
        "loc1_m1": compute_measurement_stats(0, half_time, location=1),
        "loc1_m2": compute_measurement_stats(half_time, TOTAL_DURATION, location=1),
        "loc2_m1": compute_measurement_stats(0, half_time, location=2),
        "loc2_m2": compute_measurement_stats(half_time, TOTAL_DURATION, location=2),
    }

    # Helper to format change
    def format_change(m1_val, m2_val, name):
        delta = m2_val - m1_val
        pct = (delta / abs(m1_val) * 100) if abs(m1_val) > 1e-10 else 0
        direction = "+" if delta > 0 else ""
        return f"  Delta {name}: {direction}{delta:.4f} ({direction}{pct:.1f}%)"

    # Print Location 1
    print("-" * 70)
    print("  LOCATION 1")
    print("-" * 70)
    m1, m2 = stats["loc1_m1"], stats["loc1_m2"]
    print(f"  {m1.period_label} ({m1.time_range_ms[0]:.1f} - {m1.time_range_ms[1]:.1f} ms):")
    print(f"    E_RMS  = {m1.E_rms:.4f}    B_RMS = {m1.B_rms:.4f}    S_y = {m1.S_y_avg:+.4f}")
    print(f"  {m2.period_label} ({m2.time_range_ms[0]:.1f} - {m2.time_range_ms[1]:.1f} ms):")
    print(f"    E_RMS  = {m2.E_rms:.4f}    B_RMS = {m2.B_rms:.4f}    S_y = {m2.S_y_avg:+.4f}")
    print()
    print(format_change(m1.S_y_avg, m2.S_y_avg, "S_y"))
    print()

    # Print Location 2
    print("-" * 70)
    print("  LOCATION 2")
    print("-" * 70)
    m1, m2 = stats["loc2_m1"], stats["loc2_m2"]
    print(f"  {m1.period_label} ({m1.time_range_ms[0]:.1f} - {m1.time_range_ms[1]:.1f} ms):")
    print(f"    E_RMS  = {m1.E_rms:.4f}    B_RMS = {m1.B_rms:.4f}    S_y = {m1.S_y_avg:+.4f}")
    print(f"  {m2.period_label} ({m2.time_range_ms[0]:.1f} - {m2.time_range_ms[1]:.1f} ms):")
    print(f"    E_RMS  = {m2.E_rms:.4f}    B_RMS = {m2.B_rms:.4f}    S_y = {m2.S_y_avg:+.4f}")
    print()
    print(format_change(m1.S_y_avg, m2.S_y_avg, "S_y"))
    print()

    print("=" * 70)
    print("  Key insight: When Circuit 2 power reverses, S_y changes as the")
    print("  two circuits' contributions add or partially cancel.")
    print("=" * 70)
    print()

    return stats


def create_measurement_bar_chart(stats: dict, filename: str = "measurement_comparison.png"):
    """
    Create a grouped bar chart comparing measurements.

    Args:
        stats: Dictionary of MeasurementStats from print_measurement_summary
        filename: Output filename for the chart
    """
    import matplotlib.pyplot as plt

    # Enable LaTeX-style rendering
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"

    # Set up the figure with dark theme (15% smaller: 14*0.85=11.9, 6*0.85=5.1)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(11.9, 5.1))
    fig.patch.set_facecolor(BACKGROUND)

    # Thin bars, touching, with tight spacing
    bar_width = 0.08
    x_spacing = 0.2  # Tight grouping
    x = np.arange(3) * x_spacing  # E, B, S_y positions

    for _idx, (ax, loc_idx) in enumerate(zip(axes, [1, 2], strict=False)):
        ax.set_facecolor(BACKGROUND)

        # Get stats for this location
        m1_key = f"loc{loc_idx}_m1"
        m2_key = f"loc{loc_idx}_m2"
        m1 = stats[m1_key]
        m2 = stats[m2_key]

        # Data for bars
        m1_vals = [m1.E_rms, m1.B_rms, m1.S_y_avg]
        m2_vals = [m2.E_rms, m2.B_rms, m2.S_y_avg]

        # Colors (slightly muted for scientific look)
        m1_color = (0.35, 0.60, 0.85)  # Steel blue
        m2_color = (0.85, 0.45, 0.38)  # Muted coral

        # Create bars (touching: offset by bar_width/2)
        ax.bar(x - bar_width / 2, m1_vals, bar_width, label=r"Measurement at $t_1$", color=m1_color, edgecolor="none")
        ax.bar(x + bar_width / 2, m2_vals, bar_width, label=r"Measurement at $t_2$", color=m2_color, edgecolor="none")

        # Styling with LaTeX
        ax.set_ylabel(r"Value (a.u.)", color=LABEL_COLOR, fontsize=10)
        ax.set_title(rf"Location {loc_idx}", color=LABEL_COLOR, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [r"$\langle E \rangle$", r"$\langle B \rangle$", r"$\langle S_y \rangle$"], color=LABEL_COLOR, fontsize=11
        )
        ax.tick_params(colors=LABEL_COLOR, labelsize=9)
        ax.legend(facecolor=BACKGROUND, edgecolor=(0.4, 0.4, 0.4), labelcolor=LABEL_COLOR, fontsize=9, framealpha=0.8)

        # Horizontal line at y=0
        ax.axhline(y=0, color=LABEL_COLOR, linestyle="-", alpha=0.3, linewidth=0.5)

        # Grid (subtle)
        ax.yaxis.grid(True, alpha=0.15, color=LABEL_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)

        # Spine colors
        for spine in ax.spines.values():
            spine.set_color(LABEL_COLOR)
            spine.set_alpha(0.25)

    plt.suptitle(r"Field Measurements: Before vs After Power Flow Reversal", color=LABEL_COLOR, fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=150, facecolor=BACKGROUND, edgecolor="none", bbox_inches="tight")
    plt.close()

    print(f"Saved bar chart to {filename}")


def main_analyze():
    """Generate measurement analysis: printed summary and bar chart."""
    stats = print_measurement_summary()
    create_measurement_bar_chart(stats)


def main():
    """Entry point - run interactive, save, or analyze based on args."""
    import sys

    if "--analyze" in sys.argv:
        main_analyze()
    elif "--save" in sys.argv:
        filename = "em_field_animation.mp4"
        for arg in sys.argv:
            if arg.endswith(".mp4") or arg.endswith(".gif"):
                filename = arg
                break
        main_save(filename)
    else:
        main_interactive()


if __name__ == "__main__":
    main()
