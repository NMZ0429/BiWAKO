from typing import Literal

import cv2 as cv
import numpy as np
import onnxruntime as rt

from .base_inference import BaseInference
from .utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "super_resolution4864": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/super_resolution4864.onnx",
    "super_resolution6464": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/super_resolution6464.onnx",
}


class RealESRGANInference(BaseInference):
    def __init__(
        self, model: Literal["super_resolution4864", "super_resolution6464"]
    ) -> None:
        """RealESRGAN Inference class.

        Args:
            model (Literal["super_resolution4864", "super_resolution6464"]): Model name.
        """
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.session = rt.InferenceSession(self.model_path)
        _, _, self.w, self.h = self.session.get_inputs()[0].shape
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image: Image) -> np.ndarray:
        image = self._read_image(image)
        input_image = self._preprocess(image)
        result = self.session.run([self.output_name], {self.input_name: input_image})[0]

        return self.__postprocess(result)

    def render(self, prediction: np.ndarray, image: Image) -> np.ndarray:
        return prediction

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image.

        1. Resize to training size.
        2. BGR to RGB
        3. Chnnel swap
        4. Add batch dimension.
        5. Convert to float32.
        6. Normalize.

        Args:
            img (np.ndarray): image in cv2 format

        Returns:
            np.ndarray: preprocessed image
        """
        img = cv.resize(img, dsize=(self.h, self.w))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = img.astype("float32")
        img = img / 255.0

        return img

    def __postprocess(self, img: np.ndarray) -> np.ndarray:
        """Postprocess image.

        1. Remove batch dimension.
        2. Scale to 0-255.
        3. Chnnel swap
        4. RGB to BGR

        Args:
            img (np.ndarray): image in cv2 format

        Returns:
            np.ndarray: postprocessed image
        """
        hr_image = np.squeeze(img)
        hr_image = np.clip((hr_image * 255), 0, 255).astype(np.uint8)
        hr_image = hr_image.transpose(1, 2, 0)
        hr_image = cv.cvtColor(hr_image, cv.COLOR_RGB2BGR)

        return hr_image
