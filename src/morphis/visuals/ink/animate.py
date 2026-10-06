"""
Ink Animation

An ink animation is a sequence of sketches, one per time, built by a function
of time from the true geometry at that instant. Every frame shares one page
rectangle (the union over all frames, fitted to the figure aspect), so the
space and anything held still stay registered from frame to frame, and every
mark keeps its seeded stipple and pen wobble. Frames are written through the
same Recording as Scene.record, so the suffix picks .mp4 or .gif.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from numpy import asarray, stack

from morphis.visuals.ink.sketch import Sketch
from morphis.visuals.recording import Recording


def animate(
    build: Callable[[float], Sketch],
    times: Sequence[float],
    path: str | Path,
    frame_rate: float = 30,
    dpi: int = 150,
) -> Path:
    """
    Render an ink animation to a video or GIF.

    Args:
        build: Returns the sketch of the scene at time t
        times: Times of the frames, in order
        path: Output file; .mp4 for video, .gif for an animated GIF
        frame_rate: Frames per second
        dpi: Resolution of each frame

    Returns:
        The written path
    """
    sketches = [build(t) for t in times]

    corners = [sketch.page_bounds() for sketch in sketches]
    lower = stack([low for low, _ in corners]).min(axis=0)
    upper = stack([high for _, high in corners]).max(axis=0)
    size = asarray(sketches[0].size)
    scale = ((upper - lower) / size).max()
    middle = 0.5 * (lower + upper)
    bounds = (middle - 0.5 * scale * size, middle + 0.5 * scale * size)

    with Recording(path, frame_rate) as recording:
        for sketch in sketches:
            recording.append(sketch.frame(dpi=dpi, bounds=bounds))

    return recording.path
