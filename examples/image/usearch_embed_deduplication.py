from pathlib import Path

import daft
import numpy as np
from daft import col
from usearch.index import Index

from jotunn.components.image.embedding import ImageEmbedding

df = daft.read_parquet(f"{Path.home()}/Downloads/train-00000-of-00072.parquet")
df = df.with_column("image", col("image")["bytes"].image.decode())

embed = ImageEmbedding(input_column="image", batch_size=2048, num_gpus=1, concurrency=1)
index = Index(ndim=512)

df = embed(df)

embeddings = df.select("image_embedding").to_pylist()

for i, embedding in enumerate(embeddings):
    index.add(i, np.array(embedding["image_embedding"]))

clustering = index.cluster()
centroid_keys, sizes = clustering.centroids_popularity
print(f"Centroid keys: {centroid_keys} | Cluster sizes: {sizes}")
clustering.plot_centroids_popularity()
