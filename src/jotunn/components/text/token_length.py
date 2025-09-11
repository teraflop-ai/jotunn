from typing import List, Optional

import daft
from daft import DataType
from transformers import AutoTokenizer

from jotunn.components.base import ScoreFilter


class TokenLength(ScoreFilter):
    def __init__(
        self,
        tokenizer_name: str,
        input_column: str = "text",
        output_column: Optional[str] = "token_length",
        daft_dtype=DataType.int32(),
        threshold: Optional[int] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            threshold=threshold,
        )

    def _score(self, text: str) -> List[int]:
        tokens = self.tokenizer(text, return_tensors=None)
        return len(tokens["input_ids"])

    def _filter(self, df: daft.DataFrame, threshold: int) -> daft.DataFrame:
        df = df.where(df[self.output_column] >= threshold)
        return df
