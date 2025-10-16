from typing import Optional

import cv2
import daft
import numpy as np
from daft import DataType
from skimage.exposure import is_low_contrast

from jotunn.components.base import ScoreFilter


class Contrast(ScoreFilter):
    def __init__(
        self,
        input_column: str = "image",
        output_column: str = "contrast_score",
        daft_dtype: DataType = DataType.int8(),
        threshold: Optional[float] = 1,
        contrast: Optional[float] = 0.05,
    ):
        self.contrast = contrast
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            min_threshold=threshold,
        )

    def _score(self, image: np.array) -> int:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        low_contrast = is_low_contrast(gray, fraction_threshold=self.contrast)
        return 0 if low_contrast else 1
