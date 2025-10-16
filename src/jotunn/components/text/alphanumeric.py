from typing import Optional

import re2
from daft import DataType

from jotunn.components.base import ScoreFilter

pattern = re2.compile(r"[[:alnum:]]")


class AlphanumericText(ScoreFilter):
    def __init__(
        self,
        input_column: str = "text",
        output_column: str = "text_length",
        daft_dtype: DataType = DataType.float32(),
        min_threshold: Optional[float] = 0.75,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            min_threshold=min_threshold,
        )

    def _score(self, text: str) -> float:
        if len(text) == 0:
            return 0.0
        alphanumeric = len(pattern.findall(text))
        return alphanumeric / len(text)
