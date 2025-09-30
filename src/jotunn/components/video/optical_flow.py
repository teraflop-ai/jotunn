from typing import Optional

import cv2
import numpy as np
from daft import DataType

from jotunn.components.base import ScoreFilter


class FarnebackOpticalFlow(ScoreFilter):
    def __init__(
        self,
        input_column: str = "filepath",
        output_column: str = "optical_flow_score",
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
        downsample_size=None,
        daft_dtype=DataType.float32(),
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        self.downsample_size = downsample_size

    def _score(self, filename: str) -> float:
        cap = cv2.VideoCapture(filename)
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return 0.0

        magnitudes = []

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            flow = cv2.calcOpticalFlowFarneback(
                self.preprocess(prev_frame),
                self.preprocess(curr_frame),
                None,
                self.pyr_scale,
                self.levels,
                self.winsize,
                self.iterations,
                self.poly_n,
                self.poly_sigma,
                self.flags,
            )
            magnitude = np.linalg.norm(flow, axis=-1).mean()
            magnitudes.append(magnitude)
            prev_frame = curr_frame

        cap.release()
        return float(np.mean(magnitudes)) if magnitudes else 0.0

    def preprocess(self, frame):
        out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return out
