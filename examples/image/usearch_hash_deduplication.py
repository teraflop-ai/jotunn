from pathlib import Path

import daft
import numpy as np
from daft import col
from usearch.index import Index

from jotunn import ImageHasher

df = daft.read_parquet(f"{Path.home()}/Downloads/train-00000-of-00072.parquet")
df = df.with_column("image", col("image")["bytes"].image.decode())

hasher = ImageHasher(
    input_column="image",
    hashing_algorithm="perceptual",
)
index = Index(ndim=64, metric="hamming", dtype="b1")
df = hasher(df)

for i, hash_list in enumerate(df.iter_rows()):
    index.add(i, np.packbits(np.array(hash_list["image_hash"], dtype=np.bool)))

clustering = index.cluster()
centroid_keys, sizes = clustering.centroids_popularity
print(f"Centroid keys: {centroid_keys} | Cluster sizes: {sizes}")
clustering.plot_centroids_popularity()
