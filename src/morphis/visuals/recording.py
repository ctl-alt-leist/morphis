"""
Recording

One writer for every animated output in morphis.visuals: frames go in as RGB
image arrays, and the file suffix picks the format. `.mp4` is H.264 video;
`.gif` is an animated GIF that loops. Scene.record and ink.animate both write
through it.

GIF frame delays are whole centiseconds, and browsers slow any delay under
20 ms, so a GIF plays at most 50 frames per second. When frames arrive faster,
the GIF keeps the frames that fall on its own ticks, so the clip keeps its
duration.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from imageio.v2 import get_writer
from numpy import asarray, uint8
from numpy.typing import NDArray


FORMATS = (".mp4", ".gif")

# Shortest GIF frame delay that plays as written, in centiseconds
GIF_MIN_DELAY = 2


class Recording:
    """
    A video or GIF being written, one frame at a time.

    Use as a context manager; the file is finalized on exit:

        with Recording("figures/orbit/orbit.mp4", frame_rate=30) as recording:
            for image in images:
                recording.append(image)

    Args:
        path: Output file, ending in .mp4 or .gif
        frame_rate: Frames per second at which frames are appended

    Raises:
        ValueError: If the suffix is not .mp4 or .gif
    """

    def __init__(self, path: str | Path, frame_rate: float):
        target = Path(path).expanduser()
        suffix = target.suffix.lower()
        if suffix not in FORMATS:
            raise ValueError(f"Unsupported recording format '{suffix}'; use .mp4 or .gif")

        is_gif = suffix == ".gif"
        delay = max(GIF_MIN_DELAY, round(100.0 / frame_rate))
        options = (
            {"duration": 10.0 * delay, "loop": 0}
            if is_gif
            else {"fps": frame_rate, "codec": "libx264", "quality": 8, "macro_block_size": 2}
        )
        target.parent.mkdir(parents=True, exist_ok=True)

        self.path = target
        self.frame_rate = frame_rate
        self.frame_count = 0
        self.written_count = 0
        self._output_period = delay / 100.0 if is_gif else 1.0 / frame_rate
        self._writer = get_writer(target, **options)

    def append(self, image: NDArray) -> None:
        """
        Append one RGB (or RGBA) image of shape (height, width, 3 or 4).

        Frames arrive at the recording's frame rate. Each is written when it
        reaches the output's next tick, which for video is every frame.
        """
        arrival = self.frame_count / self.frame_rate
        is_due = arrival + 1e-9 >= self.written_count * self._output_period
        if is_due:
            frame = asarray(image)[..., :3].astype(uint8)
            self._writer.append_data(frame)
            self.written_count += 1
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
