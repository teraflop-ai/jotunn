import daft
from daft import col
from nupunkt import para_tokenize


class TextSegmentation:
    def __init__(self, input_column="text", output_column="paragraphs"):
        self.input_column = input_column
        self.output_column = output_column

    def __call__(self, df):
        df = df.with_column(
            self.output_column,
            col(self.input_column).apply(
                lambda x: para_tokenize(x),
                return_dtype=daft.DataType.list(daft.DataType.string()),
            ),
        )
        return df
