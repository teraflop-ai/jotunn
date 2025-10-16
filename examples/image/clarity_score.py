import daft
from daft import col

from jotunn.components.image.clarity import Clarity

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

clarity_filter = Clarity(input_column="image")

df = clarity_filter(df)
df.show()
