import daft
from daft import col

from jotunn import NSFWClassifier

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

marqo = NSFWClassifier(
    classifier="marqo",
    model_name="hf_hub:Marqo/nsfw-image-detection-384",
    input_column="image",
    output_column="marqo_nsfw",
    batch_size=12,
    concurrency=1,
    num_cpus=6,
    num_gpus=1,
)

falcons = NSFWClassifier(
    classifier="falcons",
    model_name="Falconsai/nsfw_image_detection",
    input_column="image",
    output_column="falcons_nsfw",
    batch_size=12,
    concurrency=1,
    num_cpus=6,
    num_gpus=1,
)

df = marqo(df)
df = falcons(df)
df.show()
