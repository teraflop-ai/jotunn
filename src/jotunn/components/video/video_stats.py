from typing import List

import daft
from daft import DataType, col
from video_reader import PyVideoReader


class VideoStats:
    def __init__(
        self,
        input_column: str = "filepath",
        output_column: str = "video_stats",
    ):
        self.input_column = input_column
        self.output_column = output_column

    def _get_video_stats(self, filepath: str) -> List:
        try:
            vr = PyVideoReader(filepath)
            info_dict = vr.get_info()
            return info_dict

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return []

    def __call__(self, df: daft.DataFrame) -> daft.DataFrame:
        df = df.with_column(
            self.output_column,
            col(self.input_column).apply(
                self._get_video_stats,
                return_dtype=DataType.struct(
                    {
                        "aspect_ratio": DataType.string(),
                        "fps": DataType.string(),
                        "frame_count": DataType.string(),
                        "duration": DataType.string(),
                        "width": DataType.string(),
                        "height": DataType.string(),
                    }
                ),
            ),
        )
        return df
