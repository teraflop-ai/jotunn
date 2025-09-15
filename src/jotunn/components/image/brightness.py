from typing import Optional

import cv2
import daft
import numpy as np
from daft import DataType

from jotunn.components.base import ScoreFilter


class Brightness(ScoreFilter):
    def __init__(
        self,
        input_column: str = "image",
        output_column: Optional[str] = "brightness_score",
        daft_dtype: DataType = DataType.float32(),
        threshold: Optional[float] = None,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            max_threshold=threshold,
        )

    def _score(self, image: np.array) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = cv2.mean(hsv)[2] / 255.0
        return brightness
