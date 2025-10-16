import daft

from jotunn.components.audio.transcription import NemoTranscription

df = daft.from_pydict(
    {
        "filepath": [
            "/Downloads/hahaha2.wav",
        ],
    }
)

transcribe = NemoTranscription()
df = transcribe(df)
df.show()
