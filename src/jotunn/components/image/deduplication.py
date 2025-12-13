import numpy as np
from tqdm import tqdm
from usearch.index import Index


class Deduplication:
    def __init__(
        self,
        ndim: int = 768,
        dtype: str = "f32",
        metric: str = "cos",
        hash_col: str = "hash",
        embedding_col: str = "embedding",
        threshold: float = 0.90,
        batch_size: int = 8192,
        search_k: int = 100,
        threads: int = 8,
    ):
        self.index = Index(ndim=ndim, dtype=dtype, metric=metric)
        self.threshold = threshold
        self.batch_size = batch_size
        self.search_k = search_k
        self.threads = threads
        self.hash_col = hash_col
        self.embedding_col = embedding_col

    def build_index(self, df):
        for partition in df.iter_partitions():
            pdf = partition.to_pandas()
            embeddings = np.stack(pdf[self.embedding_col].to_numpy())
            ids = pdf[self.hash_col].to_numpy()
            self.index.add(ids, embeddings)

    def deduplicate(self):
        duplicates_to_remove = set()

        keys = np.array(self.index.keys)
        n = len(keys)

        for i in tqdm(range(0, n, self.batch_size), desc="Scanning index in batches"):
            query_keys = keys[i : i + self.batch_size]
            query_vecs = self.index.get(query_keys)

            results = self.index.search(
                query_vecs, count=self.search_k, threads=self.threads
            )

            for q_idx, query_key in enumerate(query_keys):
                matches = results.keys[q_idx]
                distances = results.distances[q_idx]

                for match_key, distance in zip(matches, distances):
                    if match_key == query_key:
                        continue
                    if int(match_key) <= int(query_key):
                        continue

                    similarity = 1 - distance
                    if similarity >= self.threshold:
                        duplicates_to_remove.add(int(match_key))

        return duplicates_to_remove
