from typing import Optional

import daft
import torch
import torch.nn.functional as F
from daft import DataType
from transformers import (
    CLIPProcessor,
    CLIPVisionModelWithProjection,
)

from jotunn.components.distributed_base import Distributed
from jotunn.models.laion import MLP


def create_laion_aesthetic_udf(
    clip_model_name: str,
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
            clip_model_name: str = clip_model_name,
            device: str = "cuda",
        ):
            self.device = device

            self.clip_processor = CLIPProcessor.from_pretrained(
                clip_model_name, use_fast=True
            )
            self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
                clip_model_name
            )
            self.clip_model.to(self.device)
            self.clip_model.eval()

            self.score_model = MLP()
            self.score_model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://huggingface.co/TeraflopAI/ddpo-aesthetic-predictor/resolve/main/aesthetic-model.pth",
                    map_location="cpu",
                    progress=True,
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
        clip_model_name=clip_model_name,
    )


class LaionAestheticClassifier(Distributed):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        batch_size: int = 1,
        input_column: str = "image",
        output_column: str = "laion_aesthetic_score",
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
        self.clip_model_name = clip_model_name

    def _udf(self):
        return create_laion_aesthetic_udf(
            clip_model_name=self.clip_model_name,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
