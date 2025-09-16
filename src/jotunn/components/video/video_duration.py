import json
import subprocess
from typing import Optional

from daft import DataType

from jotunn.components.base import ScoreFilter


class VideoDuration(ScoreFilter):
    def __init__(
        self,
        input_column: str = "filepath",
        output_column: str = "video_duration",
        daft_dtype=DataType.float32(),
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            min_threshold=min_duration,
            max_threshold=max_duration,
        )

    def _score(self, filename: str) -> float:
        result = subprocess.check_output(
            f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"',
            shell=True,
        ).decode()
        fields = json.loads(result)["streams"][0]
        return float(fields["duration"])
