import daft
from daft import col

from jotunn import Brightness

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

brightness_filter = Brightness(input_column="image")

df = brightness_filter(df)
df.show()
