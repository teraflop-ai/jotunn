from typing import Optional

import daft
import timm
import torch
from daft import DataType
from PIL import Image

from jotunn.components.distributed_base import Distributed


def create_timm_udf(
    model_name: str,
    pretrained: str,
    use_compile: bool,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.int8(),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class TimmUDF:
        def __init__(
            self,
            model_name: str = model_name,
            device: str = "cuda",
            pretrained: str = pretrained,
            use_compile: bool = use_compile,
        ):
            self.device = torch.device(device=device)
            if self.device.type == "cpu":
                self.dtype = torch.float32
            else:
                self.dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )

            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
            )
            self.model.to(device=device)
            if use_compile:
                self.model = torch.compile(self.model)
            self.model.eval()

            data_config = timm.data.resolve_model_data_config(self.model)
            self.transforms = timm.data.create_transform(
                **data_config, is_training=False
            )

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            with torch.inference_mode():
                inputs = self.processor(images.to_pylist())
                outputs = self.classify_images(inputs)
                pred_scores = torch.argmax(outputs, dim=1).cpu().tolist()
                return pred_scores

        def classify_images(self, inputs):
            if self.device.type == "cpu":
                return self.model(inputs)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                return self.model(inputs)

        def processor(self, images):
            inputs = [self.transforms(Image.fromarray(img)) for img in images]
            batch = torch.stack(inputs).to(device=self.device)
            return batch

    return TimmUDF.with_init_args(
        model_name=model_name,
        pretrained=pretrained,
        use_compile=use_compile,
    )


class ImageClassifier(Distributed):
    def __init__(
        self,
        classifier: str,
        model_name: str,
        batch_size: int = 1,
        input_column: str = "image",
        output_column: str = "image_embedding",
        concurrency: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        pretrained: bool = True,
        use_compile: bool = True,
    ):
        super().__init__(
            input_columns=input_column,
            output_column=output_column,
            batch_size=batch_size,
            concurrency=concurrency,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )
        self.classifier = classifier
        self.model_name = model_name
        self.pretrained = pretrained
        self.use_compile = use_compile

    def _udf(self):
        if self.classifier == "timm":
            return create_timm_udf(
                model_name=self.model_name,
                pretrained=self.pretrained,
                use_compile=self.use_compile,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        else:
            raise NotImplementedError()
