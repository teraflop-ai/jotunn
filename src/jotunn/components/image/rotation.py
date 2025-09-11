from io import BytesIO
from typing import Optional

import daft
import exifread
from daft import DataType

from jotunn.components.base import ScoreFilter


class Rotation(ScoreFilter):
    def __init__(
        self,
        input_column: str = "image_bytes",
        output_column: Optional[str] = "image_orientation",
        daft_dtype: DataType = DataType.int8(),
        threshold: Optional[int] = 1,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            threshold=threshold,
        )

    def _score(self, image_bytes: bytes) -> int:
        tags = exifread.process_file(BytesIO(image_bytes))
        if "Image Orientation" in tags:
            orientation = tags["Image Orientation"].values[0]
            return orientation
        return -1

    def _filter(self, df: daft.DataFrame, threshold: int) -> daft.DataFrame:
        df = df.where(
            (df[self.output_column] == threshold) | (df[self.output_column] == -1)
        )
        return df
