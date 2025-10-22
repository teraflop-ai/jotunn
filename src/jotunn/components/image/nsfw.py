from typing import Optional

import daft
import timm
import torch
from daft import DataType
from PIL import Image

from jotunn.components.distributed_base import Distributed


def create_falcon_nsfw_udf(
    model_name: str,
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
    class FalconsNSFWUDF:
        def __init__(
            self,
            model_name: str = model_name,
            device: str = "cuda",
        ):
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self.device = device

            self.processor = AutoImageProcessor.from_pretrained(
                model_name, use_fast=True
            )
            self.model = AutoModelForImageClassification.from_pretrained(model_name).to(
                self.device
            )
            self.model.compile()
            self.model.eval()

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            inputs = self.processor(images=images.to_pylist(), return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model(**inputs).logits
            predicted_labels = outputs.argmax(-1)
            scores = [p.cpu().item() for p in predicted_labels]
            return scores

    return FalconsNSFWUDF.with_init_args(
        model_name=model_name,
    )


def create_maqro_nsfw_udf(
    model_name: str,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.string(),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class MarqoNSFWUDF:
        def __init__(
            self,
            model_name: str = model_name,
            device: str = "cuda",
        ):
            self.device = torch.device(device=device)
            if self.device.type == "cpu":
                self.dtype = torch.float32
            else:
                self.dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )

            self.model = timm.create_model(model_name=model_name, pretrained=True).to(
                self.device, self.dtype
            )
            self.model.compile()
            self.model.eval()

            data_config = timm.data.resolve_model_data_config(self.model)
            self.transforms = timm.data.create_transform(
                **data_config, is_training=False
            )

            self.class_names = self.model.pretrained_cfg["label_names"]

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            inputs = torch.stack(
                [self.transforms(Image.fromarray(img)) for img in images]
            ).to(self.device, dtype=self.dtype)
            with torch.no_grad():
                outputs = self.model(inputs).softmax(dim=-1).cpu()
            scores = [self.class_names[p.argmax().item()] for p in outputs]
            return scores

    return MarqoNSFWUDF.with_init_args(
        model_name=model_name,
    )


class NSFWClassifier(Distributed):
    def __init__(
        self,
        classifier: str,
        model_name: str,
        batch_size: int = 1,
        input_column: str = "image",
        output_column: str = "nsfw_score",
        concurrency: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
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

    def _udf(self):
        if self.classifier == "falcons":
            return create_falcon_nsfw_udf(
                model_name=self.model_name,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        elif self.classifier == "marqo":
            return create_maqro_nsfw_udf(
                model_name=self.model_name,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        else:
            return NotImplementedError()
