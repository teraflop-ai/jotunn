import os

import cv2
import daft
import numpy as np

from jotunn.components.image.resize import Resize
from jotunn.components.video.frame_extractor import FrameExtractor

daft.set_execution_config(parquet_target_filesize=5 * 1024 * 1024 * 1024)

df = daft.from_pydict(
    {
        "filepath": [
            "/Parallel Computer Architecture and Programming, Lecture 4 (Tsinghua⧸CMU 2017 Summer Course) [yNP6RsiZNXQ].mkv"
        ]
    }
)

extractor = FrameExtractor(input_column="filepath", number_frames=8, encoding=".webp")
resized = Resize(input_column="image", output_column="image")

df = extractor(df)
df = resized(df)

os.makedirs("/output_frames", exist_ok=True)
df.write_parquet("/output_frames")

df.show()

os.makedirs("/extracted_frames", exist_ok=True)
for i, img in enumerate(df.to_pydict()["image"]):
    cv2.imwrite(
        f"/extracted_frames/frame_{i}.webp",
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
    )
