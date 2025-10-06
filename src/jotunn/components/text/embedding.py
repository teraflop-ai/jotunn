from typing import Optional

import daft
import numpy as np
import torch
from daft import DataType
from loguru import logger

from jotunn.components.distributed_base import Distributed


def create_sentence_transformer_udf(
    model_name: str,
    max_seq_length: Optional[int] = None,
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
    class SentenceTransformersUDF:
        def __init__(
            self,
            model_name: str = model_name,
            batch_size: int = batch_size,
            device: str = "cuda",
            convert_to_tensor: bool = False,
            dtype: torch.dtype = torch.bfloat16,
            attn_implementation: str = "sdpa",
            max_seq_length: Optional[int] = max_seq_length,
            token: str = None,
            show_progress_bar: bool = False,
        ):
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                device=device,
                token=token,
                model_kwargs={
                    "torch_dtype": dtype,
                    "attn_implementation": attn_implementation,
                },
            )
            self.model = torch.compile(self.model)

            if max_seq_length is not None:
                logger.info(f"Max sequence length is set to: {max_seq_length}")
                self.model.max_seq_length = max_seq_length

            self.convert_to_tensor = convert_to_tensor
            self.device = device
            self.show_progress_bar = show_progress_bar
            self.batch_size = batch_size

        def __call__(self, text_col: daft.DataFrame) -> daft.DataFrame:
            embeddings = self.model.encode(
                text_col.to_pylist(),
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor,
                device=self.device,
            )
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            embeddings = embeddings.astype(np.float32)
            return embeddings

    return SentenceTransformersUDF.with_init_args(
        model_name=model_name,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
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
            from transformers import AutoTokenizer, CLIPModel

            self.device = device

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.model = CLIPModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            ).to(self.device)
            self.model = torch.compile(self.model)
            self.model.eval()

        def __call__(self, text: daft.DataFrame) -> daft.DataFrame:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text=text.to_pylist(), return_tensors="pt", padding=True
                ).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_embs = text_features.float().cpu().numpy()
                return text_embs

    return ClipUDF.with_init_args(
        model_name=model_name,
    )


class TextEmbedding(Distributed):
    def __init__(
        self,
        model_name: str,
        embedder: str = "clip",
        batch_size: int = 1,
        input_column: str = "text",
        output_column: str = "text_embedding",
        concurrency: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        max_seq_length: Optional[int] = None,
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
        self.max_seq_length = max_seq_length

    def _udf(self):
        if self.embedder == "sentence-transformers":
            return create_sentence_transformer_udf(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        elif self.embedder == "clip":
            return create_clip_udf(
                model_name=self.model_name,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        else:
            raise NotImplementedError()
