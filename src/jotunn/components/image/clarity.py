from typing import Optional

import cv2
import daft
import numpy as np
from daft import DataType

from jotunn.components.base import ScoreFilter


class Clarity(ScoreFilter):
    def __init__(
        self,
        input_column: str = "image",
        output_column: Optional[str] = "clarity_score",
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clarity_score = cv2.Laplacian(gray, cv2.CV_64F).var() * gray.std()
        return float(clarity_score)

    def _filter(self, df: daft.DataFrame, threshold: float) -> daft.DataFrame:
        df = df.where(df[self.output_column] >= threshold)
        return df
