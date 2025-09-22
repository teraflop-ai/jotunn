import daft
from daft import col

from jotunn.components.image.embedding import ClipImageEmbedding, SiglipEmbedding

df = daft.from_pydict(
    {
        "urls": [
            "https://live.staticflickr.com/65535/53671838774_03ba68d203_o.jpg",
            "https://live.staticflickr.com/65535/53671700073_2c9441422e_o.jpg",
            "https://live.staticflickr.com/65535/53670606332_1ea5f2ce68_o.jpg",
            "https://live.staticflickr.com/65535/53671838039_b97411a441_o.jpg",
            "https://live.staticflickr.com/65535/53671698613_0230f8af3c_o.jpg",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxNcZf5QCNIARnmYAmWfTso4_OkQ1WB_L0mQ&s",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQdJcekXmVj22d55ETnKABLMSckfVnaOAHEw&s",
        ],
    }
)

siglip = SiglipEmbedding(
    input_column="image", batch_size=8, concurrency=1, num_cpus=6, num_gpus=1
)

clip = ClipImageEmbedding(
    input_column="image", batch_size=8, concurrency=1, num_cpus=6, num_gpus=1
)

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())
df = siglip(df)
df = clip(df)
df.show()
