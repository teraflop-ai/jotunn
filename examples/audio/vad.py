from pathlib import Path

import daft

from jotunn.components.audio.vad import VAD

df = daft.from_pydict(
    {
        "filepath": [
            f"{Path.home()}/Downloads/hahaha2.wav",
        ],
    }
)

vad = VAD()
df = vad(df)
df.show()
