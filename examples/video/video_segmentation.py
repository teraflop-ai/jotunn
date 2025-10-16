from pathlib import Path

import daft
from daft import col

from jotunn.components.video.video_segmentation import SceneSegmentation

df = daft.from_pydict(
    {
        "filepath": [
            f"{Path.home()}/Downloads/apple.mp4",
        ]
    }
)

video_segmentation = SceneSegmentation.with_init_args()

df = df.with_column(
    "output_paths",
    video_segmentation(
        col("filepath"),
    ),
)
df.show()
