import daft
from daft import col

from jotunn import FileSize

df = daft.read_huggingface("huggan/wikiart")

image_size_filter = FileSize(input_column="image_bytes")

df = df.with_column("image_bytes", col("image")["bytes"])
df = image_size_filter(df)
df.show()
