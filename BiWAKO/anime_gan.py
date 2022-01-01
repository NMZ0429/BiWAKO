from typing import Literal, Tuple

import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

from .base_inference import BaseInference
from .utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "animeGAN512": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/animeGAN512.onnx"
}


class AnimeGAN(BaseInference):
    def __init__(self, model: Literal["animeGAN512"] = "animeGAN512") -> None:
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.model = InferenceSession(self.model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.ouput_name = self.model.get_outputs()[0].name
        self.input_size = 512

    def predict(self, image: Image) -> np.ndarray:
        img = self._read_image(image)
        img = self._preprocess(img)
        pred = self.model.run([self.ouput_name], {self.input_name: img})[0]
        ani = self._postprocess(pred)

        return ani

    def render(
        self,
        prediction: np.ndarray,
        image: Image = None,
        input_size: Tuple[int, int] = None,
        **kwargs
    ) -> np.ndarray:
        if image:
            h, w = self._read_image(image).shape[:2]
        elif input_size:
            h, w = input_size
        else:
            raise ValueError("Either image or input_size must be given.")

        return cv.resize(prediction, (w, h))

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv.resize(image, (self.input_size, self.input_size))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img = img * 2 - 1
        img = img.reshape((1, 3, self.input_size, self.input_size))

        return img

    def _postprocess(self, image: np.ndarray) -> np.ndarray:
        ani = image[0]
        ani = (ani * 0.5 + 0.5).clip(0, 1) * 255
        ani = ani.transpose((1, 2, 0)).astype(np.uint8)
        ani = cv.cvtColor(ani, cv.COLOR_RGB2BGR)

        return ani

