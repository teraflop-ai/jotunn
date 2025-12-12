import daft
from daft import col

from jotunn import ImageClassifier

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

classifier = ImageClassifier(
    model_name="hf_hub:TeraflopAI/compression-detection-224",
    classifier="timm",
    input_column="image",
    output_column="compression-detection",
    use_compile=False,
    batch_size=1,
    num_gpus=1,
)

df = classifier(df)
df.show()
