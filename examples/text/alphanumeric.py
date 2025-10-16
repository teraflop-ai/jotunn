import daft

from jotunn.components.text.alphanumeric import AlphanumericText

df = daft.from_pydict(
    {
        "text": [
            "My mother told me",
            "Someday I will buy",
            "Galleys with good oars",
            "Sail to distant shores",
            "!@#$%^&*(,)_+=:;",
        ],
    }
)

text_length_filter = AlphanumericText(input_column="text", min_threshold=0.75)
df = text_length_filter(df)
df.show()
