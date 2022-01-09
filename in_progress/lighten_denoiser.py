from typing import Literal

from numpy import ndarray

from onnxruntime import InferenceSession
from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "hep480_640": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/hep480_640.onnx"
}


class HEP(BaseInference):
    def __init__(self, model: Literal["hep480_640"] = "hep480_640"):
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.model = InferenceSession(self.model_path)

    def predict(self, image: Image) -> ndarray:
        img = self._read_image(image)

        return img
