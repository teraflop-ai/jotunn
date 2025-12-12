from pathlib import Path

import daft
from daft import col

from jotunn import ImageEmbedding

df = daft.read_parquet(f"{Path.home()}/wikiart/data/**")
df = df.with_column("image", col("image")["bytes"].image.decode())

siglip = ImageEmbedding(
    embedder="siglip",
    model_name="nielsr/siglip-base-patch16-224",
    input_column="image",
    output_column="siglip_image_embedding",
    batch_size=8,
    concurrency=1,
    num_gpus=1,
)

clip = ImageEmbedding(
    embedder="clip",
    model_name="openai/clip-vit-base-patch32",
    input_column="image",
    output_column="clip_image_embedding",
    batch_size=8,
    concurrency=1,
    num_gpus=1,
)

openclip = ImageEmbedding(
    embedder="openclip",
    model_name="MobileCLIP2-S3",
    pretrained="dfndr2b",
    input_column="image",
    output_column="openclip_image_embedding",
    batch_size=256,
    num_gpus=1,
    reparameterize=True,
)

df = siglip(df)
df = clip(df)
df = openclip(df)
df.show()
