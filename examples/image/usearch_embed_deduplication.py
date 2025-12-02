import time
from pathlib import Path

import daft
import numpy as np
from daft import col
from tqdm import tqdm
from usearch.index import Index

from jotunn.components.image.embedding import ImageEmbedding

df = daft.read_parquet(f"{Path.home()}/Downloads/train-00000-of-00072.parquet")
df = df.limit(100)
df = df.with_column("image", col("image")["bytes"].image.decode())
df = df.with_column("hash", col("image").hash())
df = df.unique("hash")

embed = ImageEmbedding(
    embedder="openclip",
    model_name="MobileCLIP2-S3",
    pretrained="dfndr2b",
    input_column="image",
    output_column="image_embedding",
    batch_size=256,
    concurrency=1,
    num_gpus=1,
    reparameterize=True,
)
df = embed(df)

index = Index(ndim=768, dtype="f32")

data = df.select("hash", "image_embedding").to_pydict()

hashes = data["hash"]
embs = data["image_embedding"]

for h, emb in zip(hashes, embs):
    index.add(h, np.array(emb, dtype=np.float32))

threshold = 0.90
batch_size = 8192
search_k = 100

duplicates_to_remove = set()

keys = np.array(index.keys)
n = len(keys)

start_time = time.time()

for i in tqdm(range(0, n, batch_size), desc="Scanning index in batches"):
    query_keys = keys[i : i + batch_size]
    query_vecs = np.stack([index[k] for k in query_keys])

    results = index.search(query_vecs, count=search_k, threads=8)

    for q_idx, query_key in enumerate(query_keys):
        matches = results.keys[q_idx]
        distances = results.distances[q_idx]

        for match_key, distance in zip(matches, distances):
            if match_key == query_key:
                continue
            if int(match_key) <= int(query_key):
                continue

            similarity = 1 - distance
            if similarity >= threshold:
                duplicates_to_remove.add(int(match_key))

end_time = time.time()

print("\n-------------------------------------")
print("Search complete.")
print(f"Total hashes to remove: {len(duplicates_to_remove)}")
print(f"Total time taken: {end_time - start_time:.2f} seconds.\n")
