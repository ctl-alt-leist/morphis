"""
Astronaut Animated

NASA's Extravehicular Mobility Unit on a rollercoaster path with rotations.

Run: uv run python -m morphis.examples.astronaut_animated
"""

from numpy import arcsin, arctan2, array, clip, cos, pi, sin
from numpy.linalg import norm

from morphis.elements import Surface
from morphis.visuals import SMALL_SQUARE, Scene


def rotation_matrix_z(angle):
    """Rotation around Z axis."""
    c, s = cos(angle), sin(angle)
    return array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_matrix_y(angle):
    """Rotation around Y axis."""
    c, s = cos(angle), sin(angle)
    return array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_x(angle):
    """Rotation around X axis."""
    c, s = cos(angle), sin(angle)
    return array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rollercoaster_path(t, path_radius, scale):
    """Rollercoaster path with loops."""
    tau = t * 2 * pi
    x = path_radius * cos(tau)
    y = path_radius * sin(tau)
    z = scale * 1.5 * (1 - cos(2 * tau)) / 2 + scale * 0.3 * sin(3 * tau)
    return array([x, y, z])


def create_scene():
    """Create the astronaut animation scene."""
    # Load the model
    surface = Surface.from_file("data/models-3d/nasa_extravehicular-mobility-unit.glb")
    print(f"Loaded model: {surface.n_vertices} vertices, {surface.n_faces} faces", flush=True)

    # Get model center and scale
    center = surface.vertices.data.mean(axis=0)
    extent = surface.vertices.data.max(axis=0) - surface.vertices.data.min(axis=0)
    scale = norm(extent)

    # Store initial vertices centered at origin
    initial_vertices = surface.vertices.data.copy() - center

    # Animation parameters
    duration = 10.0
    fps = 30
    n_frames = int(duration * fps)
    path_radius = scale * 4

    # Create scene
    scene = Scene(theme="obsidian", show_basis=False, size=SMALL_SQUARE)
    scene.add(surface, color=(0.85, 0.85, 0.9))

    # Set up lighting
    scene.add_light(position=(15, -10, 12), intensity=1.0, directional=True)
    scene.add_light(position=(-10, 5, 8), intensity=0.4, directional=True)
    scene.add_light(position=(0, 0, -10), intensity=0.2, directional=True)

    # Set camera
    cam_dist = scale * 12
    scene.camera(
        position=(cam_dist * 0.7, -cam_dist * 0.7, cam_dist * 0.5),
        focal_point=(0, 0, scale * 1.5),
    )
    scene.set_clipping_range(near=0.1, far=cam_dist * 5)

    # Capture frames
    for frame in range(n_frames):
        t_normalized = frame / n_frames
        t = frame / fps

        position = rollercoaster_path(t_normalized, path_radius, scale)

        # Compute tangent
        dt = 0.001
        tangent = rollercoaster_path(min(1, t_normalized + dt), path_radius, scale) - rollercoaster_path(
            max(0, t_normalized - dt), path_radius, scale
        )
        tangent /= norm(tangent) + 1e-10

        # Rotation angles
        heading_angle = arctan2(tangent[1], tangent[0]) + pi
        pitch_angle = arcsin(clip(tangent[2], -1, 1)) * 0.7
        roll_angle = t_normalized * 4 * pi * 0.3

        # Combined rotation
        R = rotation_matrix_z(heading_angle) @ rotation_matrix_x(pitch_angle) @ rotation_matrix_y(roll_angle)

        # Transform vertices
        transformed = (R @ initial_vertices.T).T + position
        surface.vertices.data[:] = transformed

        scene.capture(t)

    return scene


if __name__ == "__main__":
    scene = create_scene()
    scene.show()
