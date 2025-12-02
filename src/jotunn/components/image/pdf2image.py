import pymupdf
import daft
from daft import col, DataType

class PDF2Image:
    def __init__(
        self,
        input_column: str = "pdf",
        output_column: str = "image",
        dpi: int = 144,
        image_format: str = "png",
        daft_dtype: DataType = DataType.list(DataType.binary()),
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.daft_dtype = daft_dtype
        self.dpi = dpi
        self.image_format = image_format

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        df = df.with_column(
            self.output_column,
            col(self.input_column).apply(
                lambda x: self.pdf_to_images(x, self.dpi, self.image_format),
                return_dtype=self.daft_dtype
            )
        )
        df = df.explode(col(self.output_column))
        df = df.with_column(self.output_column, col(self.output_column).image.decode())
        return df

    def pdf_to_images(self, pdf_path, dpi, image_format):
        images = []

        pdf_document = pymupdf.open(pdf_path)

        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img_bytes = pixmap.tobytes(image_format)
            images.append(img_bytes)

        pdf_document.close()
        return images
