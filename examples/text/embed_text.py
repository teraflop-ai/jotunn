import daft

from jotunn.components.text.embedding import (
    ClipTextEmbedding,
    SentenceTransformersEmbed,
)

df = daft.from_pydict(
    {
        "text": [
            "My mother told me",
            "Someday I will buy",
            "Galleys with good oars",
            "Sail to distant shores",
        ],
    }
)

clip_embedder = ClipTextEmbedding(
    input_column="text",
    batch_size=4,
    concurrency=1,
    num_cpus=6,
    num_gpus=1,
)

embedder = SentenceTransformersEmbed(
    input_column="text",
    model_name="all-MiniLM-L6-v2",
    max_seq_length=4,
    batch_size=4,
    concurrency=1,
    num_cpus=6,
    num_gpus=1,
)
df = embedder(df)
df = clip_embedder(df)
df.show()
