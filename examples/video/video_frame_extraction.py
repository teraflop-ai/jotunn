import os
from pathlib import Path

import cv2
import daft
import numpy as np

from jotunn.components.image.resize import Resize
from jotunn.components.video.frame_extractor import FrameExtractor

df = daft.from_pydict(
    {
        "filepath": [
            f"{Path.home()}/Downloads/apple.mp4",
        ]
    }
)

extractor = FrameExtractor(input_column="filepath")
resized = Resize(input_column="image", output_column="image")

df = extractor(df)
df = resized(df)

os.makedirs(f"{Path.home()}/output_frames", exist_ok=True)
df.write_parquet(f"{Path.home()}/output_frames")

df.show()

os.makedirs(f"{Path.home()}/extracted_frames", exist_ok=True)
for i, img in enumerate(df.to_pydict()["image"]):
    cv2.imwrite(f"{Path.home()}/extracted_frames/frame_{i}.webp", np.array(img))
