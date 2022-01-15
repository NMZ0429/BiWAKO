from pathlib import Path
from typing import Optional

import cv2 as cv
from tqdm import tqdm

from ..model.base_inference import BaseInference

__all__ = ["VideoPredictor"]


class VideoPredictor:
    def __init__(self, model: BaseInference) -> None:
        self.model = model

    def run(self, video_path: str, title: Optional[str] = None) -> None:
        self.predict_video(video_path, self.model, title)

    def predict_video(
        self, video_path: str, model: BaseInference, title: Optional[str] = None
    ) -> None:
        """Load video and predict all frames by model. The result is saved at the same directory as video_path.
        
        Return the width, height and fps of the video.

        Args:
            video_path (str): Path to the video to predict.
            model (BiWAKO.MODNet): Model to use.
            title (str, optional): Title of the video. Defaults to None.
        """
        # Set up video capture
        input_video = cv.VideoCapture(video_path)
        w, h = (
            int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )
        fps = input_video.get(cv.CAP_PROP_FPS)
        num_frames = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))

        # Set up video writer
        v_title = title or model.__class__.__name__ + "_prediction.mp4"
        v_title = v_title if v_title.endswith(".mp4") else v_title + ".mp4"
        output_video = str(Path(video_path).parent / v_title)
        fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
        output_video = cv.VideoWriter(output_video, fourcc, fps, (w, h))

        print(f"Predicting {Path(video_path).name} ...")
        with tqdm(total=num_frames) as pbar:
            while True:
                ret, frame = input_video.read()
                if not ret:
                    break
                pred = model.predict(frame)
                img = model.render(pred, frame)
                output_video.write(img)
                pbar.update(1)

        cv.destroyAllWindows()
        output_video.release()

    def make_video(self, video_path: str, img_size: tuple, fps=25.0) -> None:
        """Make video from frames in frame_path. The result is saved at the same directory as video_path.

        Args:
            frame_path (str): Path to the directory containing frames.
            video_path (str): Path to the video to save.
        """
        # encoder(for mp4)
        fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")

        # output file name, encoder, fps, size(fit to image size)
        frame_path = Path(video_path).parent / "predictions"
        video_path = str(Path(frame_path).parent / "predictions.mp4")
        video = cv.VideoWriter(video_path, fourcc, fps, img_size)

        # read frames in frame_path
        # frames = natsorted([str(p) for p in Path(frame_path).glob("*.png")])
        frames = [str(p) for p in Path(frame_path).glob("*.png")]
        for f in frames:
            img = cv.imread(str(f))
            video.write(img)

        cv.destroyAllWindows()
        video.release()

    def clean_up(self, video_path: str):
        # delete all files in predictions and remove predictions directory
        predictions = Path(video_path).parent / "predictions"
        preds = Path(predictions).glob("*")
        for p in preds:
            p.unlink()
        Path(predictions).rmdir()

