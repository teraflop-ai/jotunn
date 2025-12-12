from pathlib import Path

from jotunn import TransNetV2Segmentation

out = TransNetV2Segmentation().predict_video(
    f"{Path.home()}/Videos/axH8WxYAf2o.mp4", show_progressbar=True
)
TransNetV2Segmentation.save_scenes_from_video(
    f"{Path.home()}/Videos/axH8WxYAf2o.mp4", out, show_progressbar=True
)
