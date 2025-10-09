import copy
from typing import Optional

import daft
import open_clip
import torch
from daft import DataType
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, SiglipModel

from jotunn.components.distributed_base import Distributed


def create_siglip_udf(
    model_name: str,
    attn_implementation: str = "sdpa",
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
            attn_implementation=attn_implementation,
        ):
            self.device = torch.device(device=device)
            if self.device.type == "cpu":
                self.dtype = torch.float32
            else:
                self.dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )

            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

            self.model = SiglipModel.from_pretrained(
                model_name,
                attn_implementation=attn_implementation,
                dtype=self.dtype,
            ).to(self.device)
            self.model = torch.compile(self.model)
            self.model.eval()

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            with torch.no_grad():
                inputs = self.processor(
                    images=images.to_pylist(), return_tensors="pt"
                ).to(self.device)
                image_features = self.encode_images(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.float().cpu().numpy()
                return image_embs

        def encode_images(self, inputs):
            if self.device.type == "cpu":
                return self.model.get_image_features(**inputs)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                return self.model.get_image_features(**inputs)

    return SiglipUDF.with_init_args(
        model_name=model_name,
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
        ):
            self.device = torch.device(device=device)
            if self.device.type == "cpu":
                self.dtype = torch.float32
            else:
                self.dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )

            self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

            self.model = CLIPModel.from_pretrained(
                model_name,
            ).to(self.device)
            self.model = torch.compile(self.model)
            self.model.eval()

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            with torch.no_grad():
                inputs = self.processor(
                    images=images.to_pylist(), return_tensors="pt"
                ).to(self.device)
                image_features = self.encode_images(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.float().cpu().numpy()
                return image_embs

        def encode_images(self, inputs):
            if self.device.type == "cpu":
                return self.model.get_image_features(**inputs)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                return self.model.get_image_features(**inputs)

    return ClipUDF.with_init_args(
        model_name=model_name,
    )


def create_openclip_udf(
    model_name: str,
    repo_id: str,
    filename: str,
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
    class OpenClipUDF:
        def __init__(
            self,
            model_name: str = model_name,
            device: str = "cuda",
            repo_id: str = repo_id,
            filename: str = filename,
        ):
            self.device = torch.device(device=device)
            if self.device.type == "cpu":
                self.dtype = torch.float32
            else:
                self.dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )

            if repo_id and filename:
                checkpoint = hf_hub_download(repo_id=repo_id, filename=filename)
                model_name = repo_id.split("/")[1]
            else:
                model_name, checkpoint = model_name.split("/")

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=model_name,  # 'ViT-B-32'
                pretrained=checkpoint,  # 'laion2b_s34b_b79k'
                device=self.device,
            )
            if "mobileclip" in filename:
                self.model = self.reparameterize_model(self.model)
            self.model = torch.compile(self.model)
            self.model.eval()

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            with torch.no_grad():
                inputs = self.processor(images.to_pylist())
                image_features = self.encode_images(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.float().cpu().numpy()
                return image_embs

        def encode_images(self, inputs):
            if self.device.type == "cpu":
                return self.model.encode_image(inputs)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                return self.model.encode_image(inputs)

        def processor(self, images):
            inputs = [self.preprocess(Image.fromarray(img)) for img in images]
            batch = torch.stack(inputs).to(device=self.device, dtype=self.dtype)
            return batch

        def reparameterize_model(self, model: torch.nn.Module) -> torch.nn.Module:
            """Method returns a model where a multi-branched structure
                used in training is re-parameterized into a single branch
                for inference.

            Args:
                model: MobileOne model in train mode.

            Returns:
                MobileOne model in inference mode.
            """
            # Avoid editing original graph
            model = copy.deepcopy(model)
            for module in model.modules():
                if hasattr(module, "reparameterize"):
                    module.reparameterize()
            return model

    return OpenClipUDF.with_init_args(
        model_name=model_name,
    )


class ImageEmbedding(Distributed):
    def __init__(
        self,
        embedder: str = "clip",  # siglip
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 1,
        input_column: str = "image",
        output_column: str = "image_embedding",
        concurrency: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        super().__init__(
            input_columns=input_column,
            output_column=output_column,
            batch_size=batch_size,
            concurrency=concurrency,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )
        self.embedder = embedder
        self.model_name = model_name
        self.repo_id = repo_id
        self.filename = filename

    def _udf(self):
        if self.embedder == "clip":
            return create_clip_udf(
                model_name=self.model_name,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        elif self.embedder == "siglip":
            return create_siglip_udf(
                model_name=self.model_name,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        elif self.embedder == "openclip":
            return create_openclip_udf(
                model_name=self.model_name,
                repo_id=self.repo_id,
                filename=self.filename,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        else:
            raise NotImplementedError()
