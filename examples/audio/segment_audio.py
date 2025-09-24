import os

import daft
from daft import col

from jotunn.components.audio.segment_audio import SegmentAudio
from jotunn.components.audio.vad import VAD

df = daft.from_pydict(
    {
        "filepath": [
            "/Downloads/hahaha2.wav",
        ],
    }
)

output_dir = "audio_chunks"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

vad = VAD()
audio_segment = SegmentAudio.with_init_args(output_dir)

df = vad(df)

df = df.with_column(
    "output_paths",
    audio_segment(
        col("filepath"),
        col("vad_timestamps"),
    ),
)
df.show()
