import daft
from daft import col

from jotunn.components.image.aesthetic import AestheticClassifier

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

laion = AestheticClassifier(
    model_name="openai/clip-vit-large-patch14",
    classifier="laion",
    input_column="image",
    output_column="laion_aesthetic_score",
    batch_size=48,
    num_gpus=1,
)

simple = AestheticClassifier(
    model_name="shunk031/aesthetics-predictor-v1-vit-large-patch14",
    classifier="simple",
    input_column="image",
    output_column="simple_aesthetic_score",
    batch_size=48,
    num_gpus=1,
)

df = laion(df)
df = simple(df)
df.show()
