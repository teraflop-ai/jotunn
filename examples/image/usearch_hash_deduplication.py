import daft
import numpy as np
from daft import col
from usearch.index import Index

from jotunn.components.image.image_hashing import ImageHasher

df = daft.from_pydict(
    {
        "urls": [
            "https://live.staticflickr.com/65535/53671838774_03ba68d203_o.jpg",
            "https://live.staticflickr.com/65535/53671700073_2c9441422e_o.jpg",
            "https://live.staticflickr.com/65535/53670606332_1ea5f2ce68_o.jpg",
            "https://live.staticflickr.com/65535/53671838039_b97411a441_o.jpg",
            "https://live.staticflickr.com/65535/53671698613_0230f8af3c_o.jpg",
        ],
    }
)

hasher = ImageHasher(
    input_column="image",
    hashing_algorithm="perceptual",
)
index = Index(ndim=64, metric="hamming", dtype="b1")

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())
df = hasher(df)

df = df.with_column("duplicate", col("urls").apply(lambda x: list(range(100)), return_dtype=daft.DataType.list(daft.DataType.int64())))
df = df.explode("duplicate")
df = df.select("urls", "image_bytes", "image", "image_hash")

for i, hash_list in enumerate(df.iter_rows()):
    index.add(i, np.packbits(np.array(hash_list["image_hash"], dtype=np.bool)))

clustering = index.cluster()
centroid_keys, sizes = clustering.centroids_popularity
print(f"Centroid keys: {centroid_keys} | Cluster sizes: {sizes}")
clustering.plot_centroids_popularity()