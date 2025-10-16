from typing import Optional

import daft
import torch
from daft import DataType

from jotunn.components.distributed_base import Distributed


def create_nemo_asr_udf(
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
    class NemoTranscriptionUDF:
        def __init__(
            self,
            model_name: str = model_name,
            batch_size: int = batch_size,
            device: str = "cuda",
        ):
            try:
                from nemo.collections.asr.models import ASRModel
            except ImportError:
                raise "pip install -U nemo_toolkit['asr']"

            self.asr_ast_model = ASRModel.from_pretrained(model_name=model_name)

            self.device = device
            self.batch_size = batch_size

        def __call__(self, audios: daft.DataFrame) -> daft.DataFrame:
            with torch.inference_mode():
                outputs = self.asr_ast_model.transcribe(
                    audios.to_pylist(), batch_size=self.batch_size
                )
            return [output[0].text for output in outputs]

    return NemoTranscriptionUDF.with_init_args(
        model_name=model_name,
        batch_size=batch_size,
    )


class NemoTranscription(Distributed):
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        batch_size: int = 1,
        input_column: str = "filepath",
        output_column: str = "nemo_asr_transcription",
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
        return create_nemo_asr_udf(
            model_name=self.model_name,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
