import daft

from jotunn.components.text.long_words import LongWords

df = daft.from_pydict(
    {
        "text": [
            "My mother told me",
            "Someday I will buy",
            "Galleys with good oars",
            "Sail to distant shores",
            "a" * 10000,
        ],
    }
)

text_length_filter = LongWords(input_column="text")
df = text_length_filter(df)
df.show()
