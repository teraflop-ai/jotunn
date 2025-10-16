from typing import List

import cv2
import daft
import numpy as np
from daft import DataType, col
from video_reader import PyVideoReader


class RustFrameExtractor:
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
        df = df.exclude("image_bytes")
        return df


class OpencvFrameExtractor:
    def __init__(
        self,
        input_column: str = "filepath",
        output_column: str = "image",
        number_frames: int = 8,
        encoding: str = ".webp",
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.number_frames = number_frames
        self.encoding = encoding

    def _extract_random_n(self, filepath: str) -> List:
        try:
            cap = cv2.VideoCapture(filepath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                return []
            indices = np.sort(
                np.random.choice(
                    total_frames, min(self.number_frames, total_frames), replace=False
                )
            )
            images = []
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    _, encoded_image = cv2.imencode(self.encoding, frame)
                    images.append(encoded_image.tobytes())
            cap.release()
            return images
        except:
            return []

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        df = df.with_column(
            "image_bytes",
            col(self.input_column).apply(
                self._extract_random_n,
                return_dtype=DataType.list(dtype=DataType.binary()),
            ),
        )
        df = df.explode("image_bytes")
        df = df.with_column(self.output_column, col("image_bytes").image.decode())
        df = df.exclude("image_bytes")
        return df


class FrameExtractor:
    def __init__(self, input_column="filepath", output_column="image"):
        self.input_column = input_column
        self.output_column = output_column

    def _extract_frame(self, filepath):
        try:
            cap = cv2.VideoCapture(filepath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                return None
            frame_idx = np.random.randint(0, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        except:
            return None

    def __call__(self, df):
        df = df.with_column(
            self.output_column,
            col(self.input_column).apply(
                self._extract_frame, return_dtype=DataType.image()
            ),
        )
        return df
