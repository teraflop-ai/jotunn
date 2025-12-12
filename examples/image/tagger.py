import daft
from daft import col

from jotunn import ImageTagger
from jotunn.utils.prompt_templates import tagging_prompts

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

tags = ["trees", "person", "suit", "hat", "landscape", "sky", "boat", "flowers"]

prompt = tagging_prompts(template="qwen", tags=tags)

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
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    prompt=prompt,
    tags=tags,
    max_tokens=128,
    max_model_len=4096,
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
