from typing import Optional

import cv2
import daft
import numpy as np
from daft import DataType

from jotunn.components.base import ScoreFilter


class Saturation(ScoreFilter):
    def __init__(
        self,
        input_column: str = "image",
        output_column: str = "saturation_score",
        daft_dtype: DataType = DataType.float32(),
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

    def _score(self, image: np.array) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = cv2.mean(hsv)[1] / 255.0
        return saturation
