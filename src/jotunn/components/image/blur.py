from typing import Optional

import cv2
import daft
import numpy as np
from daft import DataType

from jotunn.components.base import ScoreFilter


class Blur(ScoreFilter):
    def __init__(
        self,
        input_column: str = "image",
        output_column: Optional[str] = "blur_score",
        daft_dtype: DataType = DataType.float32(),
        threshold: Optional[float] = 100.0,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            min_threshold=threshold,
        )

    def _score(self, image: np.array) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(blur_score)
