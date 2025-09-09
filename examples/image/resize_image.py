import daft
from daft import col

from jotunn.components.image.resize import Resize

df = daft.from_pydict(
    {
        "urls": [
            "https://live.staticflickr.com/2498/4065367744_54865768dc_b.jpg",
            "https://live.staticflickr.com/65535/53671700073_2c9441422e_o.jpg",
            "https://live.staticflickr.com/65535/53670606332_1ea5f2ce68_o.jpg",
            "https://live.staticflickr.com/65535/53671838039_b97411a441_o.jpg",
            "https://live.staticflickr.com/65535/53671698613_0230f8af3c_o.jpg",
            "https://media-cdn.tripadvisor.com/media/photo-s/01/c4/97/a6/a-very-bright-brighton.jpg",
            "https://www.shutterstock.com/shutterstock/videos/1010107970/thumb/1.jpg?ip=x480",
        ],
    }
)

resized = Resize(input_column="image")

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())
df = resized(df)
df.show()
