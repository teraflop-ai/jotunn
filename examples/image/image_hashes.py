import daft
from daft import col

from jotunn.components.image.image_hashing import ImageHasher

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

hasher = ImageHasher(
    input_column="image",
    hashing_algorithm="perceptual",
)

df = hasher(df)
df.show()
