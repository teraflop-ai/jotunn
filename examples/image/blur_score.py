import daft
from daft import col

from jotunn.components.image.blur import Blur

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

blur_filter = Blur(input_column="image")

df = blur_filter(df)
df.show()
