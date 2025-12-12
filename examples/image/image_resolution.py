import daft
from daft import col

from jotunn import Resolution

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

resolution_filter = Resolution(input_column="image", min_width=300, min_height=300)

df = resolution_filter(df)
df.show()
