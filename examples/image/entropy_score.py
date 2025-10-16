import daft
from daft import col

from jotunn.components.image.entropy import Entropy

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

entropy_filter = Entropy(input_column="image")

df = entropy_filter(df)
df.show()
