import daft
from daft import col

from jotunn import Contrast

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

contrast_filter = Contrast(input_column="image", contrast=0.35)

df = contrast_filter(df)
df.show()
