import copy

import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "basic": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/basic.onnx",
    "mobile": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/mobile.onnx",
    "human_seg": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/human_seg.onnx",
    "portrait": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/portrait.onnx",
}


__all__ = ["U2Net"]


class U2Net(BaseInference):
    """Salient object segmentation model.

    Attributes:
        IS (onnxruntime.InferenceSession): Inference session.
        input_name (str): Input node name.
        output_name (str): Output node name.
        input_size (int): Input size. Set to 320.
        mean (List[float]): Mean. Set to [0.485, 0.456, 0.406].
        std (List[float]): Standard deviation. Set to [0.229, 0.224, 0.225].
    """

    def __init__(self, model: str = "mobile") -> None:
        """U2Net Inference class.

        Args:
            model (str, optional): Model name or downloaded onnx file. Accept one of `["basic", "mobile", "human_seg", "portrait"]`.
                                    If model has not been downloaded, it will be downloaded automatically.
        """
        model_path = maybe_download_weight(WEIGHT_PATH, model)

        self.IS = InferenceSession(model_path)
        self.input_name = self.IS.get_inputs()[0].name
        self.output_name = self.IS.get_outputs()[0].name
        self.input_size = 320
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def predict(self, image: Image) -> np.ndarray:
        """Return the predicted mask of the image.

        Args:
            image (Union[str, np.ndarray]): Image in cv2 format or path to image.

        Returns:
            np.ndarray: Predicted mask in cv2 format.
        """

        x = self._read_image(image)

        x = self._preprocess(x)

        # Predict
        onnx_result = self.IS.run(
            [self.output_name], {self.input_name: x}
        )  # return list of np.ndarray

        # Postprocess
        onnx_result = np.array(onnx_result)
        onnx_result = self.__postprocess(onnx_result)

        return onnx_result

    def render(self, prediction: np.ndarray, image: Image) -> np.ndarray:
        """Apply the predicted mask to the original image.

        Args:
            prediction (np.ndarray): Predicted mask in cv2 format.
            image (Union[str, np.ndarray]): Image in cv2 format or path to image.

        Returns:
            np.ndarray: Rendered original image in cv2 format.
        """
        img = cv.imread(image) if isinstance(image, str) else image
        mask = copy.deepcopy(prediction)
        mask = cv.resize(mask, dsize=(img.shape[1], img.shape[0])) / 255

        return (img * mask.reshape(img.shape[0], img.shape[1], 1)).astype(np.uint8)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image.

        0. BGR to RGB.
        1. Resize to training size.
        2. Convert to float32.
        3. Normalize.
        4. Channel swap.
        5. Add batch dimension.

        Args:
            img (np.ndarray): image in cv2 format.

        Returns:
            np.ndarray: Preprocessed image in cv2 format.
        """
        x = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x = cv.resize(img, dsize=(self.input_size, self.input_size))
        x = np.array(x, dtype=np.float32)
        x = (x / 255 - self.mean) / self.std
        x = x.transpose(2, 0, 1).astype("float32")
        x = x.reshape(-1, 3, self.input_size, self.input_size)

        return x

    def __postprocess(self, onnx_result: np.ndarray) -> np.ndarray:
        """Postprocess image.

        1. Remove batch dimension.
        2. Channel swap.
        3. Denormalize.
        4. Resize to original size.
        5. Convert to cv2 format.

        Args:
            img (np.ndarray): image in cv2 format
        """
        onnx_result = onnx_result.squeeze()
        min_value = np.min(onnx_result)
        max_value = np.max(onnx_result)
        onnx_result = (onnx_result - min_value) / (max_value - min_value)
        onnx_result *= 255
        onnx_result = onnx_result.astype("uint8")

        return onnx_result
