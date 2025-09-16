from typing import List

import cv2
import daft
import numpy as np
from daft import DataType, col
from video_reader import PyVideoReader


class FrameExtractor:
    def __init__(
        self,
        input_column: str = "image",
        output_column: str = "image",
        number_frames: int = 8,
        encoding: str = ".webp",
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.number_frames = number_frames
        self.encoding = encoding

    def _extract(self, filepath: str) -> List:
        try:
            vr = PyVideoReader(filepath)
            info_dict = vr.get_info()
            total_frames = int(info_dict["frame_count"])

            if total_frames == 0:
                return []

            num_to_sample = min(total_frames, self.number_frames)
            indices = np.random.choice(total_frames, size=num_to_sample, replace=False)
            indices.sort()
            frames = vr.get_batch(indices.tolist())

            images = []
            for frame in frames:
                _, encoded_image = cv2.imencode(self.encoding, frame)
                image = encoded_image.tobytes()
                images.append(image)
            return images

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return []

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        df = df.with_column(
            "image_bytes",
            col(self.input_column).apply(
                self._extract, return_dtype=DataType.list(dtype=DataType.binary())
            ),
        )
        df = df.explode("image_bytes")
        df = df.with_column(self.output_column, col("image_bytes").image.decode())
        return df
