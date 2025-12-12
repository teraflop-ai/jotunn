import daft
from daft import col

from jotunn import Exposure

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

exposure_filter = Exposure(input_column="image")

df = exposure_filter(df)
df.show()
