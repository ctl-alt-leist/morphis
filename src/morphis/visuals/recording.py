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

A GIF is built in two passes once every frame is known: the frames are kept
losslessly in a temporary video, one 256-color palette is chosen for the whole
clip, and each frame stores only the rectangle that changed. With one palette
a still part of the picture has the same pixels in every frame, so a figure
whose background holds still costs little more than its moving parts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from types import TracebackType

from imageio.v2 import get_writer
from imageio_ffmpeg import get_ffmpeg_exe
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
        target.parent.mkdir(parents=True, exist_ok=True)

        self.path = target
        self.frame_rate = frame_rate
        self.frame_count = 0
        self.written_count = 0
        self._output_period = delay / 100.0 if is_gif else 1.0 / frame_rate

        # A GIF's frames wait, lossless, in a scratch video until its palette can be chosen
        self._scratch = Path(mkdtemp(prefix="morphis-gif-")) if is_gif else None
        self._writer = (
            get_writer(
                self._scratch / "frames.mkv",
                fps=1.0 / self._output_period,
                codec="ffv1",
                pixelformat="bgr0",
                macro_block_size=1,
            )
            if is_gif
            else get_writer(target, fps=frame_rate, codec="libx264", quality=8, macro_block_size=2)
        )

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
        """Finalize the file; a GIF is encoded here, from every frame at once."""
        self._writer.close()
        if self._scratch is not None:
            try:
                if self.written_count > 0:
                    self._encode_gif()

            finally:
                rmtree(self._scratch, ignore_errors=True)
                self._scratch = None

    def _encode_gif(self) -> None:
        """One palette for the whole clip, then each frame as the rectangle that changed."""
        frames, palette = self._scratch / "frames.mkv", self._scratch / "palette.png"
        ffmpeg = [get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(frames)]
        subprocess.run([*ffmpeg, "-vf", "palettegen=stats_mode=diff", str(palette)], check=True)
        subprocess.run(
            [
                *ffmpeg,
                "-i",
                str(palette),
                "-lavfi",
                "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                "-loop",
                "0",
                str(self.path),
            ],
            check=True,
        )

    def __enter__(self) -> Recording:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()
