from typing import Optional

import daft
import numpy as np
from daft import DataType
from PIL import Image

from jotunn.components.base import ScoreFilter


class Resolution(ScoreFilter):
    def __init__(
        self,
        input_column: str = None,
        output_column: Optional[str] = "image_resolution",
        daft_dtype: DataType = DataType.struct(
            {
                "width": DataType.int32(),
                "height": DataType.int32(),
            }
        ),
        threshold: Optional[int] = None,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            threshold=threshold,
        )

    def _score(self, image: np.array) -> dict:
        image = Image.fromarray(image)
        width, height = image.size
        return dict(width=width, height=height)

    def _filter(self, df: daft.DataFrame, threshold: int) -> daft.DataFrame:
        df = df.where(
            (df[self.output_column]["width"] > threshold)
            & (df[self.output_column]["height"] > threshold)
        )
        return df
