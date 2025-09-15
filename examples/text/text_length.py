import daft

from jotunn.components.text.text_length import TextLength

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

text_length_filter = TextLength(input_column="text", min_threshold=20)
df = text_length_filter(df)
df.show()
