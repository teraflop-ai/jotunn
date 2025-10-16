import daft


class Resize:
    def __init__(
        self,
        input_column: str = "image",
        output_column: str = "image_resized",
        width: int = 256,
        height: int = 256,
    ):
        super().__init__()
        self.input_column = input_column
        self.output_column = output_column
        self.width = width
        self.height = height

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        df = df.with_column(
            self.output_column,
            df[self.input_column].image.resize(self.width, self.height),
        )
        return df
