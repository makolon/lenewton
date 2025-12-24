import os
import logging

import imageio
import numpy as np

log = logging.getLogger(__name__)


###
# Video Recorder
###


class VideoRecorder:
    """Helper class for recording video frames during environment execution."""

    def __init__(self, enabled: bool = True, fps: int = 30):
        self.enabled = enabled
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self.step_count = 0

    def record_frame(self, image: np.ndarray):
        """Record a given image frame directly."""
        if not self.enabled:
            return
        self.frames.append(image)

    def save_video(
        self, filename: str = "execution_video.mp4", output_path: str = "outputs"
    ) -> str | None:
        """Save captured frames as MP4 video."""
        if not self.frames:
            log.warning("No frames to save")
            return None

        # Determine output directory
        out_dir = os.path.join(os.getcwd(), output_path)
        os.makedirs(out_dir, exist_ok=True)

        video_path = os.path.join(out_dir, filename)

        with imageio.get_writer(video_path, fps=self.fps, codec="libx264") as writer:
            for frame in self.frames:
                writer.append_data(frame)

        log.info(f"Video saved to: {video_path}")
        return video_path

    def clear_frames(self):
        """Clear captured frames to free memory."""
        self.frames = []
        self.step_count = 0