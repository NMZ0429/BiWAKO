from typing import Literal, Optional, Tuple
from onnxruntime import InferenceSession
import numpy as np
import cv2 as cv
from .base_inference import BaseInference
from .utils import maybe_download_weight, Image

WEIGHT_PATH = {
    "denoise_320_480": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/denoise_320_480.onnx"
}


class HINet(BaseInference):
    def __init__(self, model: str = "") -> None:
        # model_path = maybe_download_weight(model)
        self.model = InferenceSession("denoise_320_480.onnx")
        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = (480, 320)

    def predict(self, image: Image) -> np.ndarray:
        img = self._read_image(image)
        img = self._preprocess(img)
        result = self.model.run(None, {self.input_name: img})
        result = np.array(result)
        return result

    def render(
        self,
        prediction: np.ndarray,
        image: Image = None,
        output_type: Literal[0, 1] = 0,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        img_1, img_2 = self._postprocess(prediction)
        if image:
            h, w = self._read_image(image).shape[:2]
        elif output_shape:
            h, w = output_shape
        else:
            raise ValueError("Either image or output_shape should be specified")

        if output_type == 0:
            return cv.resize(img_1, dsize=(w, h))
        elif output_type == 1:
            return cv.resize(img_2, dsize=(w, h))
        else:
            raise ValueError("output_type must be 0 or 1")

    def _postprocess(self, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prediction = prediction.transpose(0, 1, 3, 4, 2)
        prediction = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)
        img_1, img_2 = prediction[0][0], prediction[1][0]
        img_1, img_2 = (
            cv.cvtColor(img_1, cv.COLOR_RGB2BGR),
            cv.cvtColor(img_2, cv.COLOR_RGB2BGR),
        )

        return img_1, img_2

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv.resize(image, dsize=self.input_shape)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = img.astype("float32")
        img /= 255.0

        return img

