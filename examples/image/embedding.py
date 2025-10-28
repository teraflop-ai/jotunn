import daft
from daft import col
from pathlib import Path

from jotunn.components.image.embedding import ImageEmbedding
from jotunn.components.image.resize import Resize

df = daft.read_parquet(f"{Path.home()}/wikiart/data/**")
df = df.with_column("image", col("image")["bytes"].image.decode())

resize = Resize(output_column="image", width=224, height=224)

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
    concurrency=1,
    num_gpus=1,
    reparameterize=True,
)

df = resize(df)

df = siglip(df)
df = clip(df)
df = openclip(df)
df.write_parquet("embeddings")
