from typing import List

from video_reader import PyVideoReader


def get_video_stats(filepath: str) -> List:
    try:
        vr = PyVideoReader(filepath)
        info_dict = vr.get_info()
        return info_dict

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []
