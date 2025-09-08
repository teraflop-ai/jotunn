from typing import Optional

import daft
from daft import DataType

from jotunn.components.base import ScoreFilter


class ImageSize(ScoreFilter):
    def __init__(
        self,
        input_column: str = None,
        output_column: Optional[str] = "image_size",
        daft_dtype: DataType = DataType.int32(),
        threshold: Optional[float] = None,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            threshold=threshold,
        )

    def _score(self, image: bytes) -> int:
        return len(image)

    def _filter(self, df: daft.DataFrame, threshold: float) -> daft.DataFrame:
        df = df.where(df[self.output_column] >= threshold)
        return df
