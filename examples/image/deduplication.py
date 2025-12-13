from pathlib import Path

import daft
from daft import col

from jotunn import Deduplication, ImageEmbedding

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
    output_column="embedding",
    batch_size=256,
    concurrency=1,
    num_gpus=1,
    reparameterize=True,
)
df = embed(df)

dedup = Deduplication()

dedup.build_index(df)
hashes = dedup.deduplicate()

print(hashes)
