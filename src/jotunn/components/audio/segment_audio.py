import os

import daft
import librosa
import soundfile as sf
from daft import DataType


@daft.udf(return_dtype=DataType.list(DataType.string()))
class SegmentAudio:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def __call__(self, filepath, vad_timestamps):
        y, sr = librosa.load(filepath.to_pylist()[0], sr=None)
        output_paths = []
        for i, segment in enumerate(vad_timestamps.to_pylist()[0]):
            start_time = segment["start"]
            end_time = segment["end"]

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            chunk = y[start_sample:end_sample]

            output_file = os.path.join(
                self.output_dir, f"chunk_{i:04d}_{start_time:.1f}s-{end_time:.1f}s.wav"
            )

            sf.write(output_file, chunk, sr)
            output_paths.append(output_file)
        return [output_paths]
