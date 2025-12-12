import daft
from daft import col

from jotunn import Rotation

df = daft.from_pydict(
    {
        "urls": [
            "https://live.staticflickr.com/65535/53671838774_03ba68d203_o.jpg",
            "https://live.staticflickr.com/65535/53671700073_2c9441422e_o.jpg",
            "https://live.staticflickr.com/65535/53670606332_1ea5f2ce68_o.jpg",
            "https://live.staticflickr.com/65535/53671838039_b97411a441_o.jpg",
            "https://live.staticflickr.com/65535/53671698613_0230f8af3c_o.jpg",
            "https://github.com/ianare/exif-samples/blob/master/jpg/Kodak_CX7530.jpg?raw=true",
            "https://github.com/ianare/exif-samples/blob/master/jpg/Canon_40D_photoshop_import.jpg?raw=true",
        ],
    }
)

rotation_filter = Rotation(input_column="image_bytes", orientation=1)

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = rotation_filter(df)
df.show()
