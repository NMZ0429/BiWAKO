from .base_inference import BaseInference
from onnxruntime import InferenceSession
from .utils import Image, maybe_download_weight
import numpy as np
import cv2 as cv


WEIGHT_PATH = {
    "ferplus8": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/ferplus8.onnx"
}


class FerPlus(BaseInference):
    def __init__(self, model: str):
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.session = InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.emotion_table = [
            "neutral",
            "happiness",
            "surprise",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "contempt",
        ]

    def predict(self, image: Image) -> np.ndarray:

        image = self._read_image(image)
        image = self._preprocess(image)
        prediction = self.session.run([self.output_name], {self.input_name: image})
        prediction = self._postprocess(prediction)

        return prediction

    def render(self, prediction: np.ndarray, image: Image) -> np.ndarray:
        # convert prediction to emotion_table
        rtn = []
        for k, i in enumerate(prediction):
            rtn.append(f"Prediction {k}: {self.emotion_table[i]}")

        return rtn  # type: ignore

    def _preprocess(self, image: np.ndarray) -> np.ndarray:

        # bgr to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = self.resize_and_gray(image)

        return image

    def _postprocess(self, prediction: np.ndarray) -> np.ndarray:
        prob = self.softmax(prediction)
        prob = np.squeeze(prob)
        classes = np.argsort(prob)[::-1]

        return classes

    def softmax(self, x):

        y = np.exp(x - np.max(x))
        f_x = y / np.sum(np.exp(x))
        return f_x

    def rgb2gray(self, rgb):
        """Convert the input image into grayscale"""
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def resize_img(self, img_to_resize):
        """Resize image to FER+ model input dimensions"""
        r_img = cv.resize(img_to_resize, dsize=(64, 64), interpolation=cv.INTER_AREA)
        r_img.resize((1, 1, 64, 64))
        return r_img

    def resize_and_gray(self, img_to_preprocess):
        """Resize input images and convert them to grayscale."""
        """if img_to_preprocess.shape == (64, 64):
            img_to_preprocess.resize((1, 1, 64, 64))
            return img_to_preprocess"""

        grayscale = self.rgb2gray(img_to_preprocess)
        processed_img = self.resize_img(grayscale)
        return processed_img.astype(np.float32)
