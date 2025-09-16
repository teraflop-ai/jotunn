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
        output_column: str = "image_orientation",
        daft_dtype: DataType = DataType.int8(),
        orientation: Optional[int] = 1,
    ):
        self.orientation = orientation
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
        )

    def _score(self, image_bytes: bytes) -> int:
        tags = exifread.process_file(BytesIO(image_bytes))
        if "Image Orientation" in tags:
            orientation = tags["Image Orientation"].values[0]
            return orientation
        return -1

    def _filter(self, df: daft.DataFrame) -> daft.DataFrame:
        df = df.where((df[self.output_column] == self.orientation))
        return df

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        """Apply the score and filter to the dataframe."""
        df = df.with_column(
            self.output_column,
            daft.col(self.input_column).apply(
                lambda x: self._score(x), return_dtype=self.daft_dtype
            ),
        )
        if self.orientation:
            df = self._filter(df)
        return df
