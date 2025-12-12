import daft
from daft import col

from jotunn import Saturation

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

saturation_filter = Saturation(input_column="image")

df = saturation_filter(df)
df.show()
