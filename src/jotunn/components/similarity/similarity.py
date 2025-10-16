import daft
import numpy as np
from daft import DataType, col


class CosineSimilarity:
    def __init__(
        self,
        col1: str,
        col2: str,
        output_column: str = "similarity_score",
        threshold: float = 0.0,
        daft_dtype: DataType = DataType.float32(),
    ):
        super().__init__()
        self.col1 = col1
        self.col2 = col2
        self.output_column = output_column
        self.threshold = threshold
        self.daft_dtype = daft_dtype

    def _score(self, embedding1, embedding2) -> float:
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        return float(np.dot(embedding1, embedding2))

    def _filter(self, df: daft.DataFrame) -> daft.DataFrame:
        return df.where(col(self.output_column) >= self.threshold)

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        @daft.udf(return_dtype=self.daft_dtype)
        def compute_similarity(embedding1_batch, embedding2_batch):
            results = []
            for emb1, emb2 in zip(
                embedding1_batch.to_pylist(), embedding2_batch.to_pylist()
            ):
                results.append(self._score(emb1, emb2))
            return results

        df = df.with_column(
            self.output_column, compute_similarity(col(self.col1), col(self.col2))
        )
        if self.threshold:
            df = self._filter(df)
        return df
