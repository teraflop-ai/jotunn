import daft

from jotunn.components.text.token_length import TokenLength

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

token_length_filter = TokenLength(
    input_column="text", tokenizer_name="gpt2", min_threshold=5
)
df = token_length_filter(df)
df.show()
