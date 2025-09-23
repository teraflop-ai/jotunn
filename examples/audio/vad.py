import daft

from jotunn.components.audio.vad import VAD

df = daft.from_pydict(
    {
        "filepath": [
            "/Downloads/hahaha2.wav",
        ],
    }
)

vad = VAD()
df = vad(df)
df.show()
