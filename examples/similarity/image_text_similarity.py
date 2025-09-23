import daft
from daft import col

from jotunn.components.image.embedding import ClipImageEmbedding
from jotunn.components.similarity.similarity import CosineSimilarity
from jotunn.components.text.embedding import ClipTextEmbedding
from jotunn.pipeline import Pipeline

df = daft.from_pydict(
    {
        "urls": [
            "https://live.staticflickr.com/65535/53671838774_03ba68d203_o.jpg",
            "https://live.staticflickr.com/65535/53671700073_2c9441422e_o.jpg",
            "https://live.staticflickr.com/65535/53670606332_1ea5f2ce68_o.jpg",
            "https://live.staticflickr.com/65535/53671838039_b97411a441_o.jpg",
            "https://live.staticflickr.com/65535/53671698613_0230f8af3c_o.jpg",
        ],
        "text": [
            "dog",
            "dog",
            "dog",
            "dog",
            "dog",
        ],
    }
)

clip_text = ClipTextEmbedding(
    input_column="text", batch_size=8, concurrency=1, num_cpus=6, num_gpus=1
)

clip_image = ClipImageEmbedding(
    input_column="image", batch_size=8, concurrency=1, num_cpus=6, num_gpus=1
)

cos_sim = CosineSimilarity("clip_image_embedding", "clip_text_embedding")

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())

pipeline = Pipeline(
    ops=[clip_text, clip_image, cos_sim],
)

df = pipeline(df)
df.show()
