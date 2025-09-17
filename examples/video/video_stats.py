import daft

from jotunn.components.video.video_stats import VideoStats

df = daft.from_pydict(
    {
        "filepath": [
            "/video/IYJgv2Cuf1E.mp4",
            "/video/bwcedHNNcVk.mp4",
            "/video/UGi_x_L3InY.mp4",
        ]
    }
)

video_stats = VideoStats()
df = video_stats(df)
df.show()
