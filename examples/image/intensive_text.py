import daft
from daft import col

from jotunn.components.image.intensive_text import IntensiveText

df = daft.from_pydict(
    {
        "urls": [
            "https://images.squarespace-cdn.com/content/v1/657f48ea60c71d001aba5971/1702925423373-9UPJWOUWIE3FFEZEJ13N/Death-by-PowerPoint-presentation-slide-example",
            "https://jeroen.github.io/images/testocr.png",
            "https://blog.richardmillwood.net/wp-content/uploads/2017/11/subtitles.png",
            "https://live.staticflickr.com/65535/53670606332_1ea5f2ce68_o.jpg",
        ],
    }
)

ocr_filter = IntensiveText(max_threshold=0.2)

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())
df = ocr_filter(df)
df.show()
