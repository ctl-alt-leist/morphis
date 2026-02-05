"""
Visualization Backends

Pluggable rendering backends for Scene visualization. Currently supports:
- PyVista (VTK-based 3D rendering)

Future backends may include matplotlib, plotly, moderngl, etc.
"""

from morphis.visuals.backends.protocol import RenderBackend
from morphis.visuals.backends.pyvista import PyVistaBackend


def get_backend(name: str = "pyvista") -> RenderBackend:
    """
    Get a rendering backend by name.

    Args:
        name: Backend name ("pyvista")

    Returns:
        Backend instance implementing RenderBackend protocol
    """
    backends = {
        "pyvista": PyVistaBackend,
    }

    if name not in backends:
        available = ", ".join(backends.keys())
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")

    return backends[name]()


__all__ = [
    "RenderBackend",
    "PyVistaBackend",
    "get_backend",
]
