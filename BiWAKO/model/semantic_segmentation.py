import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "fast_scnn384": "https://github.com/NMZ0429/Weights/releases/download/Temp/fast_scnn384.onnx",
    "fast_scnn7681344": "https://github.com/NMZ0429/Weights/releases/download/Temp/fast_scnn7681344.onnx",
}


class FastSCNN(BaseInference):
    def __init__(self, model: str = "fast_scnn384", **kwargs) -> None:
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.model = InferenceSession(self.model_path)
        self.input_shape = (384, 384) if "384" in model else (1344, 768)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])
        self.c_map = self.__get_color_map_list(19)

    def predict(self, image: Image) -> np.ndarray:
        image = self._read_image(image)
        dsize = (image.shape[1], image.shape[0])
        image = self._preprocess(image)
        prediction = self.model.run([self.output_name], {self.input_name: image})[0]
        prediction = np.squeeze(prediction).transpose(1, 2, 0)
        prediction = cv.resize(prediction, dsize)

        return np.argmax(prediction, axis=2)

    def render(self, prediction: np.ndarray, image: Image, **kwargs) -> np.ndarray:
        input_img = self._read_image(image)
        for i in range(0, 19):
            mask = np.where(prediction == i, 0, 1)

            bg_image = np.zeros(input_img.shape, dtype=np.uint8)
            bg_image[:] = (
                self.c_map[i * 3 + 0],
                self.c_map[i * 3 + 1],
                self.c_map[i * 3 + 2],
            )

            # Overlay
            mask = np.stack((mask,) * 3, axis=-1).astype("uint8")
            mask_image = np.where(mask, input_img, bg_image)
            input_img = cv.addWeighted(input_img, 0.5, mask_image, 0.5, 1.0)

        return input_img

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, self.input_shape)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return np.expand_dims(image, axis=0)

    def __get_color_map_list(self, num_classes, custom_color=None):
        num_classes += 1
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3 + 2] |= ((lab >> 0) & 1) << (7 - j)
                color_map[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j)
                color_map[i * 3] |= ((lab >> 2) & 1) << (7 - j)
                j += 1
                lab >>= 3
        color_map = color_map[3:]

        if custom_color:
            color_map[: len(custom_color)] = custom_color

        return color_map
