import daft

from jotunn import VideoDuration

df = daft.from_pydict(
    {
        "filepath": [
            "/video/IYJgv2Cuf1E.mp4",
            "/video/bwcedHNNcVk.mp4",
            "/video/UGi_x_L3InY.mp4",
        ]
    }
)

video_duration = VideoDuration()

df = video_duration(df)
df.show()
