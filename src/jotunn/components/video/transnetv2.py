import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from jotunn.models.transnetv2 import TransNetV2

class TransNetV2Segmentation:
    """
    Taken from: https://github.com/soCzech/TransNetV2/issues/54
    """
    def __init__(self, url = "https://github.com/SlimRG/transnetv2pt/raw/master/transnetv2pt/transnetv2-pytorch-weights.pth"):
        self.model = TransNetV2()
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to("cuda")
        self.model.eval()

    def extract_frames_with_opencv(self,video_path: str, target_height: int = 27, target_width: int = 48, show_progressbar: bool = False):
        """
        Extracts frames from a video using OpenCV with optional CUDA support and progress tracking.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        # Initialize progress bar
        progress_bar = tqdm(total=total_frames, desc="Extracting frames", unit="frame") if show_progressbar else None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame
            frame_resized = cv2.resize(frame_rgb, (target_width, target_height))
            frames.append(frame_resized)
            if progress_bar:
                progress_bar.update(1)

        cap.release()
        if progress_bar:
            progress_bar.close()
        return np.array(frames)

    def input_iterator(self, frames):
        """
        Generator that yields batches of 100 frames, with padding at the beginning and end.
        """
        no_padded_frames_start = 25
        no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)

        start_frame = np.expand_dims(frames[0], 0)
        end_frame = np.expand_dims(frames[-1], 0)
        padded_inputs = np.concatenate(
            [start_frame] * no_padded_frames_start +
            [frames] +
            [end_frame] * no_padded_frames_end, 0
        )

        ptr = 0
        while ptr + 100 <= len(padded_inputs):
            out = padded_inputs[ptr:ptr + 100]
            ptr += 50
            yield out[np.newaxis]

    def predictions_to_scenes(self, predictions: np.ndarray, threshold: float = 0.5):
        """
        Converts model predictions to scene boundaries based on a threshold.
        """
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    def predict_raw(self, model, video, device=torch.device('cuda:0')):
        """
        Performs inference on the video using the TransNetV2 model.
        """

        with torch.no_grad():
            predictions = []
            for inp in self.input_iterator(video):
                video_tensor = torch.from_numpy(inp).to(device)
                single_frame_pred, all_frame_pred = model(video_tensor)
                single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
                all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()
                predictions.append(
                    (single_frame_pred[0, 25:75, 0], all_frame_pred[0, 25:75, 0]))
            single_frame_pred = np.concatenate([single_ for single_, _ in predictions])
            return video.shape[0], single_frame_pred

    def predict_video(self, video_path: str, device: str = 'cuda', show_progressbar: bool = False):
        """
        Detects shot boundaries in a video file using the TransNetV2 model.
        """
        frames = self.extract_frames_with_opencv(video_path, show_progressbar=show_progressbar)
        _, single_frame_pred = self.predict_raw(self.model, frames, device=device)
        scenes = self.predictions_to_scenes(single_frame_pred)
        return scenes

    def save_scenes_from_video(video_path: str, scenes: np.ndarray, output_dir: str = "scenes", codec: str = "mp4v", show_progressbar: bool = False):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scenes = np.clip(scenes, 0, total_frames - 1)
        saved_paths = []

        progress_bar = tqdm(total=len(scenes), desc="Saving scenes", unit="clip") if show_progressbar else None

        for idx, (start, end) in enumerate(scenes):
            clip_path = os.path.join(output_dir, f"scene_{idx:04d}.mp4")
            out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for frame_idx in range(start, end + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()
            saved_paths.append(clip_path)
            if progress_bar:
                progress_bar.update(1)

        cap.release()
        if progress_bar:
            progress_bar.close()

        return saved_paths
