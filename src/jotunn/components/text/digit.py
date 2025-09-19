from typing import Optional

import re2
from daft import DataType

from jotunn.components.base import ScoreFilter

pattern = re2.compile("[[:digit:]]")


class Digits(ScoreFilter):
    def __init__(
        self,
        input_column: str = "text",
        output_column: str = "text_length",
        daft_dtype: DataType = DataType.float32(),
        max_threshold: Optional[float] = 0.15,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            max_threshold=max_threshold,
        )

    def _score(self, text: str) -> float:
        if len(text) == 0:
            return 0.0
        digits = len(pattern.findall(text))
        return digits / len(text)
