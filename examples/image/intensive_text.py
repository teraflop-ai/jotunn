from pathlib import Path

import daft
from daft import col

from jotunn import IntensiveText

df = daft.read_parquet(f"{Path.home()}/wikiart/data/**")
df = df.with_column("image", col("image")["bytes"].image.decode()).limit(100)

ocr_filter = IntensiveText(max_threshold=0.2)

df = ocr_filter(df)
df.show()
