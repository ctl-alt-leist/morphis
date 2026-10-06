"""
Recording

One writer for every animated output in morphis.visuals: frames go in as RGB
image arrays, and the file suffix picks the format. `.mp4` is H.264 video;
`.gif` is an animated GIF that loops. Scene.record and ink.animate both write
through it.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from imageio.v2 import get_writer
from numpy import asarray, uint8
from numpy.typing import NDArray


FORMATS = (".mp4", ".gif")


class Recording:
    """
    A video or GIF being written, one frame at a time.

    Use as a context manager; the file is finalized on exit:

        with Recording("figures/orbit/orbit.mp4", frame_rate=30) as recording:
            for image in images:
                recording.append(image)

    Args:
        path: Output file, ending in .mp4 or .gif
        frame_rate: Frames per second of the output

    Raises:
        ValueError: If the suffix is not .mp4 or .gif
    """

    def __init__(self, path: str | Path, frame_rate: float):
        target = Path(path).expanduser()
        suffix = target.suffix.lower()
        if suffix not in FORMATS:
            raise ValueError(f"Unsupported recording format '{suffix}'; use .mp4 or .gif")

        is_gif = suffix == ".gif"
        options = (
            {"duration": 1000.0 / frame_rate, "loop": 0}
            if is_gif
            else {"fps": frame_rate, "codec": "libx264", "quality": 8, "macro_block_size": 2}
        )
        target.parent.mkdir(parents=True, exist_ok=True)

        self.path = target
        self.frame_rate = frame_rate
        self.frame_count = 0
        self._writer = get_writer(target, **options)

    def append(self, image: NDArray) -> None:
        """Append one RGB (or RGBA) image of shape (height, width, 3 or 4)."""
        frame = asarray(image)[..., :3].astype(uint8)
        self._writer.append_data(frame)
        self.frame_count += 1

    def close(self) -> None:
        """Finalize the file."""
        self._writer.close()

    def __enter__(self) -> Recording:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()
