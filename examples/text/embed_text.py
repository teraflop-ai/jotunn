import daft

from jotunn import TextEmbedding

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

clip_embedder = TextEmbedding(
    embedder="clip",
    model_name="openai/clip-vit-base-patch32",
    input_column="text",
    output_column="clip_text_embedding",
    batch_size=4,
    num_gpus=1,
)

st_embedder = TextEmbedding(
    embedder="sentence-transformers",
    input_column="text",
    output_column="st_text_embedding",
    model_name="all-MiniLM-L6-v2",
    max_seq_length=4,
    batch_size=4,
    num_gpus=1,
)
df = st_embedder(df)
df = clip_embedder(df)
df.show()
