import daft
from daft import col

from jotunn.components.image.saturation import Saturation

df = daft.from_pydict(
    {
        "urls": [
            "https://live.staticflickr.com/2498/4065367744_54865768dc_b.jpg",
            "https://live.staticflickr.com/65535/53671700073_2c9441422e_o.jpg",
            "https://live.staticflickr.com/65535/53670606332_1ea5f2ce68_o.jpg",
            "https://live.staticflickr.com/65535/53671838039_b97411a441_o.jpg",
            "https://live.staticflickr.com/65535/53671698613_0230f8af3c_o.jpg",
            "https://cdn.visualwilderness.com/wp-content/uploads/2022/12/new_mexico_9791.jpg",
            "https://f64academy.com/~f64academy/wp-content/uploads/2012/11/Before-512x385.jpg",
        ],
    }
)

saturation_filter = Saturation(input_column="image")

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())
df = saturation_filter(df)
df.show()
