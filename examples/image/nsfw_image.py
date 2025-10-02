import daft
from daft import col

from jotunn.components.image.nsfw import FalconsNSFWClassifier

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

classifier = FalconsNSFWClassifier(
    input_column="image", batch_size=8, concurrency=1, num_cpus=6, num_gpus=1
)

df = classifier(df)
df.show()
