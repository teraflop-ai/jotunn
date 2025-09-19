from typing import Optional

from daft import DataType

from jotunn.components.base import ScoreFilter


class LongWords(ScoreFilter):
    def __init__(
        self,
        input_column: str = "text",
        output_column: str = "text_length",
        daft_dtype: DataType = DataType.float32(),
        max_threshold: Optional[float] = 1000,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            max_threshold=max_threshold,
        )

    def _score(self, text) -> float:
        words = text.strip().split()
        if not words:
            return 0.0
        return float(max(len(word) for word in words))
