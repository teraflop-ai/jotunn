from typing import Optional

import daft
from daft import col


class BatchDecode:
    def __init__(
        self,
        input_column: str = "image_bytes",
        output_column: str = "image",
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.input_column = input_column
        self.output_column = output_column
        self.batch_size = batch_size

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        if self.batch_size:
            df = df.into_batches(self.batch_size)
        df = df.with_column(self.output_column, col(self.input_column).image.decode())
        return df
