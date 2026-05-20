import numpy as np


class BagSource:
    """Reads aligned color + depth frames from Intel RealSense .bag files."""

    def __init__(self, bag_path: str, real_time: bool = False):
        self.bag_path = bag_path
        self.real_time = real_time
        self.pipeline = None
        self.align = None
        self._start_time_ms = None

    def open(self) -> bool:
        import pyrealsense2 as rs

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            rs.config.enable_device_from_file(config, self.bag_path, repeat_playback=False)
            config.enable_stream(rs.stream.color)
            config.enable_stream(rs.stream.depth)

            profile = self.pipeline.start(config)

            playback = profile.get_device().as_playback()
            playback.set_real_time(self.real_time)

            self.align = rs.align(rs.stream.color)
            self._start_time_ms = None
            return True
        except Exception as e:
            print(f"[BagSource] Failed to open {self.bag_path}: {e}")
            self.pipeline = None
            return False

    def read(self) -> tuple[bool, np.ndarray | None, object | None, float]:
        """Read next frame. Returns (success, color_image, depth_frame, timestamp_ms).

        depth_frame is a raw pyrealsense2 depth_frame for get_distance() calls,
        or None if depth is unavailable.
        """
        if self.pipeline is None:
            return False, None, None, 0.0

        while True:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                return False, None, None, 0.0

            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            if color_frame and color_frame.get_data_size() > 0:
                depth_frame = aligned.get_depth_frame()
                break

        timestamp_ms = color_frame.get_timestamp()
        if self._start_time_ms is None:
            self._start_time_ms = timestamp_ms
        relative_ms = timestamp_ms - self._start_time_ms

        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_out = depth_frame if depth_frame else None

        return True, color_image, depth_out, relative_ms

    def release(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    @property
    def fps(self) -> float:
        return 30.0
