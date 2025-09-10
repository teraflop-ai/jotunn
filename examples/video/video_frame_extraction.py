import cv2
import daft
import numpy as np
from daft import DataType, col

from jotunn.components.video.frame_extractor import extract_random_frame_sample

df = daft.from_pydict({"filepath": ["/video/my.mp4"]})
df = df.with_column(
    "image_bytes",
    col("filepath").apply(
        extract_random_frame_sample, return_dtype=DataType.list(dtype=DataType.binary())
    ),
)
df = df.explode("image_bytes")
df = df.with_column("image", col("image_bytes").image.decode())
df = df.with_column("image", df["image"].image.resize(256, 256))
df.write_parquet("output_frames")

for i, img in enumerate(df.to_pydict()["image"]):
    cv2.imwrite(
        f"extracted_frames/frame_{i}.webp",
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
    )
