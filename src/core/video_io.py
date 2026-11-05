"""Video input / output utilities for PitchVision.

Functions
---------
frame_generator
    Lazy iterator that yields ``(index, timestamp_ms, frame)`` tuples
    from a video file.
get_video_properties
    Returns a metadata dict (fps, resolution, duration, …).

Classes
-------
VideoExporter
    Thin wrapper around :class:`cv2.VideoWriter` for annotated output.
"""

from __future__ import annotations

import logging
from typing import Generator

import cv2
import numpy as np

from src.config import VIDEO_EXPORT_CODEC

logger = logging.getLogger(__name__)


def frame_generator(
    video_path: str,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Yield frames from *video_path* as ``(index, timestamp_ms, frame)`` tuples.

    Parameters
    ----------
    video_path : str
        Filesystem path to the input video.

    Yields
    ------
    tuple[int, float, np.ndarray]
        ``(frame_index, timestamp_ms, bgr_frame)``

    Raises
    ------
    FileNotFoundError
        If *video_path* cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms: float = cap.get(cv2.CAP_PROP_POS_MSEC)
            yield frame_index, timestamp_ms, frame
            frame_index += 1
    finally:
        cap.release()
        logger.debug("Released VideoCapture for %s after %d frames", video_path, frame_index)


def get_video_properties(video_path: str) -> dict:
    """Return metadata for *video_path*.

    Parameters
    ----------
    video_path : str
        Filesystem path to the input video.

    Returns
    -------
    dict
        Keys: ``fps`` (float), ``width`` (int), ``height`` (int),
        ``total_frames`` (int), ``duration_s`` (float).

    Raises
    ------
    FileNotFoundError
        If *video_path* cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0.0
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration_s": duration_s,
        }
    finally:
        cap.release()


class VideoExporter:
    """Write annotated frames to a video file.

    Parameters
    ----------
    output_path : str
        Destination file path (e.g. ``"output/result.mp4"``).
    fps : float
        Frames per second for the output video.
    frame_size : tuple[int, int]
        ``(width, height)`` of each frame in pixels.

    Example
    -------
    >>> exporter = VideoExporter("out.mp4", fps=30.0, frame_size=(1920, 1080))
    >>> exporter.write_frame(annotated_bgr)
    >>> exporter.release()
    """

    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int]) -> None:
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_EXPORT_CODEC)
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise IOError(f"Failed to open VideoWriter at {output_path}")
        logger.info("VideoExporter opened: %s @ %.1f fps, %s", output_path, fps, frame_size)

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single BGR frame to the output video.

        Parameters
        ----------
        frame : np.ndarray
            BGR image with the same dimensions as *frame_size*.
        """
        self._writer.write(frame)

    def release(self) -> None:
        """Flush and close the underlying :class:`cv2.VideoWriter`."""
        self._writer.release()
        logger.info("VideoExporter released: %s", self.output_path)
