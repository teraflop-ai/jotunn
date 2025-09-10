from typing import List

import cv2
import numpy as np
from video_reader import PyVideoReader


def extract_random_frame_sample(filepath: str) -> List:
    try:
        vr = PyVideoReader(filepath)
        info_dict = vr.get_info()
        total_frames = int(info_dict["frame_count"])

        if total_frames == 0:
            return []

        num_to_sample = min(total_frames, 4)
        indices = np.random.choice(total_frames, size=num_to_sample, replace=False)
        indices.sort()
        frames = vr.get_batch(indices.tolist())

        images = []
        for frame in frames:
            _, encoded_image = cv2.imencode(".webp", frame)
            image = encoded_image.tobytes()
            images.append(image)
        return images

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []
