import daft

from jotunn.components.text.digit import Digits

df = daft.from_pydict(
    {
        "text": [
            "My mother told me",
            "Someday I will buy",
            "Galleys with good oars",
            "Sail to distant shores",
            "11111111111",
        ],
    }
)

text_length_filter = Digits(input_column="text", max_threshold=0.5)
df = text_length_filter(df)
df.show()
