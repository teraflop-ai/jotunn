from pathlib import Path

import daft

from jotunn.components.video.cotracker import Cotracker

df = daft.from_pydict(
    {
        "filepath": [
            f"{Path.home()}/Downloads/apple.mp4",
        ]
    }
)

cotracker = Cotracker(
    input_column="filepath",
    batch_size=1,
    concurrency=1,
    num_cpus=6,
    num_gpus=1,
)

df = cotracker(df)
df.show()
