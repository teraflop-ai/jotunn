from typing import Optional

import cv2
import daft
import numpy as np
import torch
from daft import DataType

from jotunn.components.distributed_base import Distributed


def create_cotracker_udf(
    model_name: str,
    grid_size: int,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.struct(
            {
                "pred_tracks": DataType.list(
                    DataType.list(DataType.list(DataType.list(DataType.float32())))
                ),  # B T N 2
                "pred_visibility": DataType.list(
                    DataType.list(DataType.list(DataType.float32()))
                ),  # B T N
            }
        ),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class CotrackerUDF:
        def __init__(
            self,
            model_name: str = model_name,
            grid_size: int = grid_size,
            device: str = "cuda",
        ):
            self.grid_size = grid_size
            self.device = device

            self.model = torch.hub.load("facebookresearch/co-tracker", model_name).to(
                device
            )
            self.model.compile()
            self.model.eval()

        def __call__(self, filepath: daft.DataFrame) -> list:
            results = []
            video_paths = filepath.to_pylist()

            for path in video_paths:
                cap = cv2.VideoCapture(path)
                frames = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

                cap.release()

                video = (
                    torch.tensor(np.array(frames))
                    .permute(0, 3, 1, 2)[None]
                    .float()
                    .to(self.device)
                )  # B T C H W

                with torch.no_grad():
                    pred_tracks, pred_visibility = self.model(
                        video, grid_size=self.grid_size
                    )  # B T N 2,  B T N 1

                results.append(
                    dict(
                        pred_tracks=pred_tracks.cpu().numpy().tolist(),
                        pred_visibility=pred_visibility.cpu().numpy().tolist(),
                    )
                )
            return results

    return CotrackerUDF.with_init_args(
        model_name=model_name,
    )


class Cotracker(Distributed):
    def __init__(
        self,
        model_name: str = "cotracker3_offline",
        grid_size: int = 10,
        batch_size: int = 1,
        input_column: str = "text",
        output_column: str = "cotracker",
        concurrency: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
    ):
        self.model_name = model_name
        self.grid_size = grid_size
        super().__init__(
            input_columns=input_column,
            output_column=output_column,
            batch_size=batch_size,
            concurrency=concurrency,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )

    def _udf(self):
        return create_cotracker_udf(
            model_name=self.model_name,
            grid_size=self.grid_size,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )
