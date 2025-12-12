import daft
from daft import col

from jotunn import VllmImageCaption

df = daft.read_huggingface("huggan/wikiart")
df = df.with_column("image", col("image")["bytes"].image.decode())

prompt = """\
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\
Write a detailed description of what is in the image?<|im_end|>\n\
<|im_start|>assistant\n"""

captioner = VllmImageCaption(
    input_column="image",
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    prompt=prompt,
    max_tokens=128,
    batch_size=1,
    concurrency=1,
    num_cpus=6,
    num_gpus=1,
)

df = captioner(df)
df.show()
