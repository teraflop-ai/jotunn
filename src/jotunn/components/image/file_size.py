from typing import Optional

import daft
from daft import DataType

from jotunn.components.base import ScoreFilter


class FileSize(ScoreFilter):
    def __init__(
        self,
        input_column: str = "image",
        output_column: str = "image_size",
        daft_dtype: DataType = DataType.int32(),
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )

    def _score(self, image: bytes) -> int:
        return len(image)
