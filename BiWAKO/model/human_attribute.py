from typing import Literal

import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "human_attribute": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/human_attribute.onnx"
}


class HumanParsing(BaseInference):
    """Basic ResNet50 model for parsing attributes of pedestrians

    Attributes:
        model (onnxruntime.InferenceSession): ONNXRuntime instance.
        conf_thresh (float): confidence threshold for prediction.
        input_name (str): name of input node.
        output_name (str): name of output node.
        input_shape (tuple): shape of input node.
        labels (np.ndarray): mapping of label index to label name.
    """

    def __init__(
        self,
        model: Literal["human_attribute"] = "human_attribute",
        conf_thresh: float = 0.4,
    ) -> None:
        model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.conf_thresh = conf_thresh
        self.model = InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        self.labels = np.array(
            [
                "is_male",
                "has_bag",
                "has_hat",
                "has_longsleeves",
                "has_longpants",
                "has_longhair",
                "has_coat_jacket",
            ]
        )

    def predict(self, image: Image) -> np.ndarray:
        """Return the prediction on the image.
        
        Args:
            image (Image): image to be processed in str or cv2 format.

        Returns:
            np.ndarray: processed image
        """
        x = self._read_image(image)
        x = self._preprocess(x)
        onnx_result = self.model.run([self.output_name], {self.input_name: x})

        return onnx_result[0].squeeze()

    def render(self, prediction: np.ndarray, image: Image) -> np.ndarray:
        """Return the original image with the prediction at the top left corner.

        Args:
            prediction (np.ndarray): prediction of the model.
            image (Image): image to be rendered in str or cv2 format.

        Returns:
            np.ndarray: rendered image
        """
        # filter array with confidence threshold
        label = self.labels[prediction > self.conf_thresh]
        img = self._read_image(image)
        # add text to image
        for i, l in enumerate(label):
            cv.putText(
                img, l, (10, 30 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
            )

        return img

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, (80, 160))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        return image
