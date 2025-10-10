import daft

from jotunn.components.code.filters import (
    AverageLineLengthFilter,
    LongLinesFilter,
    MaximumLineLengthFilter,
)

df = daft.from_pydict(
    {
        "text": [
            "My mother told me",
            "Someday I will buy",
            "Galleys with good oars",
            "Sail to distant shores",
            "a" * 10000,
            "short line\nanother short line",
            "this is a very long line " * 60,
            "normal\nok\nfine",
            "This is a line.\n" * 100000,
        ],
    }
)

long_lines = LongLinesFilter()
maximum_line_length = MaximumLineLengthFilter()
average_line_length = AverageLineLengthFilter()

df = long_lines(df)
df = maximum_line_length(df)
df = average_line_length(df)

df.show()
