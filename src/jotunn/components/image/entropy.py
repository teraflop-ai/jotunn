from typing import Optional

import cv2
import daft
import numpy as np
from daft import DataType
from skimage.measure import shannon_entropy

from jotunn.components.base import ScoreFilter


class Entropy(ScoreFilter):
    def __init__(
        self,
        input_column: str = None,
        output_column: Optional[str] = "entropy_score",
        daft_dtype: DataType = DataType.float32(),
        threshold: Optional[float] = None,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            threshold=threshold,
        )

    def _score(self, image: np.array) -> float:
        if image is None:
            return -1.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entropy = shannon_entropy(gray)
        return entropy

    def _filter(self, df: daft.DataFrame, threshold: float) -> daft.DataFrame:
        df = df.where(
            (df[self.output_column] >= threshold) | (df[self.output_column] == -1)
        )
        return df
