from typing import Optional

import daft
import numpy as np
from daft import DataType
from PIL import Image

from jotunn.components.base import ScoreFilter


class Resolution(ScoreFilter):
    def __init__(
        self,
        input_column: str = "image",
        output_column: str = "image_resolution",
        daft_dtype: DataType = DataType.struct(
            {
                "width": DataType.int32(),
                "height": DataType.int32(),
            }
        ),
        min_width: Optional[int] = 256,
        min_height: Optional[int] = 256,
    ):
        self.min_width = min_width
        self.min_height = min_height
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
        )

    def _score(self, image: np.array) -> dict:
        image = Image.fromarray(image)
        width, height = image.size
        return dict(width=width, height=height)

    def _filter(self, df: daft.DataFrame) -> daft.DataFrame:
        df = df.where(
            (df[self.output_column]["width"] >= self.min_width)
            & (df[self.output_column]["height"] >= self.min_height)
        )
        return df

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        df = df.with_column(
            self.output_column,
            daft.col(self.input_column).apply(
                lambda x: self._score(x), return_dtype=self.daft_dtype
            ),
        )

        if self.min_width is not None or self.min_height is not None:
            df = self._filter(df)

        return df
