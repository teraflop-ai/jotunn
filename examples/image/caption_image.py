import daft
from daft import col

from jotunn.components.image.caption import VllmImageCaption

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

prompt = """\
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\
f"Write a detailed description of what is in the image?<|im_end|>\n\
<|im_start|>assistant\n"""

captioner = VllmImageCaption(
    input_column="image",
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    prompt=prompt,
    batch_size=4,
    concurrency=1,
    num_cpus=6,
    num_gpus=1,
)

df = df.with_column("image_bytes", col("urls").url.download(on_error="null"))
df = df.with_column("image", col("image_bytes").image.decode())
df = captioner(df)
df.show()
