import daft
from daft import col

from jotunn.components.image.tagger import ImageTagger

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())


florence = ImageTagger(
    tagger="florence",
    model_name="florence-community/Florence-2-base-ft",
    input_column="image",
    batch_size=8,
    num_gpus=1,
)

weeb = ImageTagger(
    tagger="weeb",
    model_name="animetimm/eva02_large_patch14_448.dbv4-full",
    input_column="image",
    batch_size=8,
    num_gpus=1,
)

df = weeb(df)
df = florence(df)
df.show()
