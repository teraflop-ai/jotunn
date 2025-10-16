from typing import List, Optional

import easyocr
import numpy as np
from daft import DataType
from PIL import Image

from jotunn.components.base import ScoreFilter


class IntensiveText(ScoreFilter):
    def __init__(
        self,
        languages: List[str] = ["ch_sim", "en"],
        gpu: bool = True,
        input_column: str = "image",
        output_column: str = "intensive_text_score",
        daft_dtype: DataType = DataType.float32(),
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
    ):
        super().__init__(
            input_column=input_column,
            output_column=output_column,
            daft_dtype=daft_dtype,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def _score(self, image: np.array) -> float:
        pil_image = Image.fromarray(image)
        width, height = pil_image.size
        image_area = width * height

        rectangles, free = self.reader.detect(image)

        quad_area = sum(
            self.PolyArea([point[0] for point in box], [point[1] for point in box])
            for box in free[0]
        )

        rect_area = sum(
            (xmax - xmin) * (ymax - ymin)
            for xmin, xmax, ymin, ymax in rectangles[0]
            if xmax > xmin and ymax > ymin
        )

        text_area = rect_area + quad_area
        total_area_ratio = text_area / image_area
        return total_area_ratio

    def PolyArea(self, x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
