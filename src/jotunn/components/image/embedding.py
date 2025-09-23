from typing import Optional

import daft
import torch
from daft import DataType

from jotunn.components.distributed_base import Distributed


def create_siglip_udf(
    model_name: str,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.list(DataType.float32()),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class SiglipUDF:
        def __init__(
            self,
            model_name: str = model_name,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
            attn_implementation: str = "sdpa",
        ):
            from transformers import AutoImageProcessor, SiglipModel

            self.device = device

            self.processor = AutoImageProcessor.from_pretrained(
                model_name, use_fast=True
            )

            self.model = SiglipModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            ).to(self.device)
            self.model.compile()
            self.model.eval()

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            with torch.no_grad():
                inputs = self.processor(
                    images=images.to_pylist(), return_tensors="pt"
                ).to(self.device)
                image_features = self.model.get_image_features(**inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.float().cpu().numpy()
                return image_embs

    return SiglipUDF.with_init_args(
        model_name=model_name,
    )


class SiglipEmbedding(Distributed):
    def __init__(
        self,
        model_name: str = "nielsr/siglip-base-patch16-224",
        batch_size: int = 1,
        input_column: str = "image",
        output_column: str = "siglip_image_embedding",
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
        self.model_name = model_name

    def _udf(self):
        return create_siglip_udf(
            model_name=self.model_name,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )


def create_clip_udf(
    model_name: str,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.list(DataType.float32()),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class ClipUDF:
        def __init__(
            self,
            model_name: str = model_name,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
            attn_implementation: str = "sdpa",
        ):
            from transformers import CLIPModel, CLIPProcessor

            self.device = device

            self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

            self.model = CLIPModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            ).to(self.device)
            self.model.compile()
            self.model.eval()

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            with torch.no_grad():
                inputs = self.processor(
                    images=images.to_pylist(), return_tensors="pt"
                ).to(self.device)
                image_features = self.model.get_image_features(**inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.float().cpu().numpy()
                return image_embs

    return ClipUDF.with_init_args(
        model_name=model_name,
    )


class ClipImageEmbedding(Distributed):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 1,
        input_column: str = "image",
        output_column: str = "clip_image_embedding",
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
        self.model_name = model_name

    def _udf(self):
        return create_clip_udf(
            model_name=self.model_name,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
