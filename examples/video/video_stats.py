import daft
from daft import DataType, col

from jotunn.components.video.video_stats import get_video_stats

df = daft.from_pydict({"filepath": ["/videos/my.mp4"]})
df = df.with_column(
    "video_stats",
    col("filepath").apply(
        get_video_stats,
        return_dtype=DataType.struct(
            {
                "aspect_ratio": DataType.string(),
                "fps": DataType.string(),
                "frame_count": DataType.string(),
                "duration": DataType.string(),
                "width": DataType.string(),
                "heigt": DataType.string(),
            }
        ),
    ),
)
df.show()
