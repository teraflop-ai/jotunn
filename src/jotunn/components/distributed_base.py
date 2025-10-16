from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import daft
from daft import col


class Distributed(ABC):
    """Base class for applying UDFs to DataFrame."""

    def __init__(
        self,
        input_columns: Union[str, List[str]],
        output_column: Optional[str] = None,
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
    ):
        if isinstance(input_columns, str):
            self.input_columns = [input_columns]
        else:
            self.input_columns = input_columns
        self.output_column = output_column
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus

    @abstractmethod
    def _udf(self) -> Any:
        """Create and return the UDF."""
        pass

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        """Apply the UDF to the dataframe."""
        df = df.with_column(
            self.output_column, self._udf()(*[col(x) for x in self.input_columns])
        )
        return df
