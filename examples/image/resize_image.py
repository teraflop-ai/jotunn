import daft
from daft import col

from jotunn import Resize

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

resized = Resize(input_column="image")

df = resized(df)
df.show()
