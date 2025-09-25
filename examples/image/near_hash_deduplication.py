import daft
from daft import col
from rapidfuzz.distance import Hamming

from jotunn.components.image.image_hashing import ImageHasher

df = daft.from_pydict(
    {
        "urls": [
            "https://cms.bbcearth.com//sites/default/files/factfiles/2024-07/el2.jpg",
            "https://media.istockphoto.com/id/166673845/photo/elephant-approaching.jpg?s=612x612&w=0&k=20&c=Qiz6i7hpv7xppQxAXSvei4QlXZHGdtkjHusIMGFZzMs=",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbB2OhIp2QfwBopb3Vo56nIplwhGqG21QWImhEViPB-eB-jmlFU5OA5otyqfPf75uDMEU&usqp=CAU",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQzanE5KR9ltAkpExLo35vJINnd_i-i--42JQig0AC0uMpzPLf5fRyZnqcx8eO-Fx23PHE&usqp=CAU",
        ],
    }
)

hasher = ImageHasher(
    input_column="image",
    hashing_algorithm="perceptual",
)

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())
df = df.with_column("uid", col("urls").hash(hash_function="xxhash"))

df = hasher(df)


@daft.func(return_dtype=daft.DataType.int8())
def hamming_distance_udf(hash1: str, hash2: str) -> int:
    return Hamming.distance(hash1, hash2)


df = df.join(
    df.select(
        col("uid").alias("uid_right"), col("image_hash").alias("image_hash_right")
    ),
    how="cross",
)

duplicates = (
    df.where(col("uid") < col("uid_right"))
    .with_column(
        "distance", hamming_distance_udf(col("image_hash"), col("image_hash_right"))
    )
    .where(col("distance") <= 15)
)

duplicates.show()
