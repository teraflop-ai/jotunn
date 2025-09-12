import json
import subprocess

def get_video_duration(filename):
    result = subprocess.check_output(
        f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"',
        shell=True,
    ).decode()
    fields = json.loads(result)["streams"][0]
    return float(fields["duration"])