import daft
from daft import col

from jotunn.components.image.intensive_text import IntensiveText

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

ocr_filter = IntensiveText(max_threshold=0.2)

df = ocr_filter(df)
df.show()
