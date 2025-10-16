import os
from typing import Optional

import daft
from daft import DataType
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_splitter import split_video_ffmpeg


@daft.udf(return_dtype=DataType.list(DataType.list(DataType.string())))
class SceneSegmentation:
    def __init__(
        self,
        output_dir: Optional[str] = None,
        threshold: float = 3.0,
        minimum_length: int = 15,
        downscale_factor: int = 64,
        filter_by_seconds: Optional[str] = "1s",
        trim_frames: Optional[int] = 6,
    ):
        super().__init__()

        self.threshold = threshold
        self.minimum_length = minimum_length
        self.downscale_factor = downscale_factor
        self.filter_by_seconds = filter_by_seconds
        self.trim_frames = trim_frames
        self.output_path = output_dir

    def __call__(self, video_path: str):
        video_path = video_path.to_pylist()[0]
        print(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if self.output_path:
            output_dirs = os.path.join(
                self.output_path, os.path.splitext(os.path.basename(video_path))[0]
            )
            os.makedirs(output_dirs, exist_ok=True)

        video = open_video(video_path)

        scene_manager = SceneManager()

        detector = AdaptiveDetector(
            adaptive_threshold=self.threshold, min_scene_len=self.minimum_length
        )

        scene_manager.add_detector(detector=detector)

        scene_manager.auto_downscale = False
        scene_manager.downscale = video.frame_size[0] // self.downscale_factor

        scene_manager.detect_scenes(video)

        scenes = scene_manager.get_scene_list()

        if self.trim_frames:
            trimmed_scenes = []
            for start, end in scenes:
                trim_start = start + self.trim_frames
                trim_end = end - self.trim_frames
                if trim_start < trim_end:
                    trimmed_scenes.append((trim_start, trim_end))
            scenes = trimmed_scenes

        if self.filter_by_seconds is not None and self.filter_by_seconds.endswith("s"):
            filter_short_videos = FrameTimecode(
                timecode=float(self.filter_by_seconds[:-1]), fps=video.frame_rate
            )

        if filter_short_videos:
            filtered_scenes = []
            for start, end in scenes:
                duration = end.get_frames() - start.get_frames()
                if duration >= filter_short_videos.get_frames():
                    filtered_scenes.append((start, end))
            scenes = filtered_scenes

        if self.output_path:
            split_video_ffmpeg(
                input_video_path=video_path,
                scene_list=scenes,
                output_dir=output_dirs,
                show_progress=False,
            )

        return [[[str(start), str(end)] for start, end in scenes]]
