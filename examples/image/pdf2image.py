import daft

from jotunn.components.image.pdf2image import PDF2Image

df = daft.from_pydict(
    {
        "pdf": [
            "/Z_Image_Report.pdf",
        ],
    }
)

pdf2image = PDF2Image()

df = pdf2image(df)
df.show()
