import daft
from daft import col

from jotunn.components.image.tagger import ImageTagger

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

tags = ["trees", "person", "suit", "hat", "landscape", "sky", "boat", "flowers"]

prompt = f"""
<image>

You must return ONLY the tags from the allowed tag list that truly apply to the image.

Return the output ONLY in EXACT JSON format:

{{
  "tags": ["tag1", "tag2"]
}}

Where each tag must be one of the allowed tags below and must NOT include anything else. There should only be one of each tag.

Allowed tags: {tags}
"""

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

vllm = ImageTagger(
    tagger="vllm",
    model_name="OpenGVLab/InternVL3_5-4B-Instruct",
    prompt=prompt,
    tags=tags,
    max_tokens=128,
    temperature=0.0,
    batch_size=1,
    concurrency=1,
    num_cpus=6,
    num_gpus=1,
)

df = weeb(df)
df = florence(df)
df = vllm(df)
df.show()
