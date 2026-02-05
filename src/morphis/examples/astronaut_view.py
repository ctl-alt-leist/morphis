"""
Astronaut Static Viewer

Displays NASA's Extravehicular Mobility Unit (spacesuit) model.

Run: uv run python -m morphis.examples.astronaut_view
"""

from morphis.elements import Surface
from morphis.visuals import Scene


surface = Surface.from_file("data/models-3d/nasa_extravehicular-mobility-unit.glb")
print(f"Loaded model: {surface.n_vertices} vertices, {surface.n_faces} faces", flush=True)

scene = Scene(theme="obsidian", show_basis=False)
scene.add(surface, color=(0.9, 0.9, 0.9))

# Three-point lighting setup
scene.add_light(position=(10, -8, 10), intensity=1.0, directional=True)  # Key
scene.add_light(position=(-8, 5, 6), intensity=0.4, directional=True)  # Fill
scene.add_light(position=(0, 10, -5), intensity=0.2, directional=True)  # Rim

scene.reset_camera()
scene.show()
