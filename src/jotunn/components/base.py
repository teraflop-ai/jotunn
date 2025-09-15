from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import daft
from daft import DataType, col


class ScoreFilter(ABC):
    """Base class for applying Score and Filter to DataFrame."""

    def __init__(
        self,
        input_column: str,
        output_column: Optional[str] = None,
        daft_dtype: DataType = None,
        min_threshold: Optional[Union[float, int]] = None,
        max_threshold: Optional[Union[float, int]] = None,
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.daft_dtype = daft_dtype
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    @abstractmethod
    def _score(self) -> Any:
        """Create and return the score."""
        pass

    def _filter(self, df: daft.DataFrame) -> daft.DataFrame:
        """Fitler the Dataframe depending on the score."""
        if self.min_threshold is not None:
            df = df.where(df[self.output_column] >= self.min_threshold)
        if self.max_threshold is not None:
            df = df.where(df[self.output_column] <= self.max_threshold)
        return df

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        """Apply the score and filter to the dataframe."""
        df = df.with_column(
            self.output_column,
            col(self.input_column).apply(
                lambda x: self._score(x), return_dtype=self.daft_dtype
            ),
        )
        if self.min_threshold is not None or self.max_threshold is not None:
            df = self._filter(df)
        return df
