from pathlib import Path

import daft

from jotunn.components.video.optical_flow import FarnebackOpticalFlow

df = daft.from_pydict(
    {
        "filepath": [
            f"{Path.home()}/Downloads/apple.mp4",
        ]
    }
)

farneback = FarnebackOpticalFlow(input_column="filepath")

df = farneback(df)
df.show()
