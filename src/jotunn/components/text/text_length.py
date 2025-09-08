from typing import Optional

import daft
from daft import DataType

from jotunn.components.base import ScoreFilter


class TextLength(ScoreFilter):
    def __init__(
        self,
        input_column: str = None,
        output_column: Optional[str] = "text_length",
        daft_dtype: DataType = DataType.int32(),
        threshold: Optional[int] = None,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            threshold=threshold,
        )

    def _score(self, text) -> float:
        return len(text)

    def _filter(self, df: daft.DataFrame, threshold: float) -> daft.DataFrame:
        df = df.where(df[self.output_column] >= threshold)
        return df
