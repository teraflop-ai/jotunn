from typing import List, Optional

import daft
from daft import DataType
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

from jotunn.components.distributed_base import Distributed


def create_vad_udf(
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.list(
            DataType.struct({"start": DataType.float32(), "end": DataType.float32()})
        ),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class VadUDF:
        def __init__(
            self,
        ):
            self.model = load_silero_vad()

        def __call__(self, filepaths: str) -> List[List[dict]]:
            results = []
            for filepath in filepaths.to_pylist():
                wav = read_audio(filepath)
                speech_timestamps = get_speech_timestamps(
                    wav,
                    model=self.model,
                    return_seconds=True,
                )
                results.append(speech_timestamps)
            return results

    return VadUDF.with_init_args()


class VAD(Distributed):
    def __init__(
        self,
        batch_size: int = 1,
        input_column: str = "filepath",
        output_column: str = "vad_timestamps",
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

    def _udf(self):
        return create_vad_udf(
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
