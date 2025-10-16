from typing import Optional

import daft
import torch
from aesthetics_predictor import AestheticsPredictorV1
from daft import DataType
from transformers import (
    CLIPProcessor,
    CLIPVisionModelWithProjection,
)

from jotunn.components.distributed_base import Distributed
from jotunn.models.laion import MLP


def create_laion_aesthetic_udf(
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
    class LaionAestheticUDF:
        def __init__(
            self,
            model_name: str = model_name,
            device: str = "cuda",
        ):
            self.device = torch.device(device=device)

            self.clip_processor = CLIPProcessor.from_pretrained(
                model_name, use_fast=True
            )
            self.clip_model = CLIPVisionModelWithProjection.from_pretrained(model_name)
            self.clip_model.to(self.device)
            self.clip_model.eval()

            self.score_model = MLP()
            self.score_model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://huggingface.co/TeraflopAI/ddpo-aesthetic-predictor/resolve/main/aesthetic-model.pth",
                    map_location="cpu",
                    progress=False,
                )
            )
            self.score_model.to(self.device)
            self.score_model.eval()

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            inputs = self.clip_processor(images=images.to_pylist(), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embed = self.clip_model(**inputs).image_embeds
                normalize_embeds = embed / torch.linalg.vector_norm(
                    embed, dim=-1, keepdim=True
                )
                scores = self.score_model(normalize_embeds)
            scores = [pred.item() for pred in scores]
            return scores

    return LaionAestheticUDF.with_init_args(
        model_name=model_name,
    )


def create_simple_aesthetic_udf(
    model_name: str,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.float32(),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class SimpleAestheticUDF:
        def __init__(
            self,
            model_name: str = model_name,
            device: str = "cuda",
        ):
            self.device = torch.device(device=device)
            self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

            self.model = AestheticsPredictorV1.from_pretrained(
                model_name,
            ).to(self.device)
            self.model = torch.compile(self.model)
            self.model.eval()

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            inputs = self.processor(images=images.to_pylist(), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            scores = outputs.logits.squeeze().float().cpu().numpy().tolist()
            return scores

    return SimpleAestheticUDF.with_init_args(
        model_name=model_name,
    )


class AestheticClassifier(Distributed):
    def __init__(
        self,
        model_name: str,
        classifier: str = "simple",
        batch_size: int = 1,
        input_column: str = "image",
        output_column: str = "aesthetic_score",
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
        if self.classifier == "laion":
            return create_laion_aesthetic_udf(
                model_name=self.model_name,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        elif self.classifier == "simple":
            return create_simple_aesthetic_udf(
                model_name=self.model_name,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        else:
            raise NotImplementedError()
