from typing import Literal, Optional, Tuple
from onnxruntime import InferenceSession
import numpy as np
import cv2 as cv
from .base_inference import BaseInference
from .util.utils import maybe_download_weight, Image

WEIGHT_PATH = {
    "denoise_320_480": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/denoise_320_480.onnx"
}


class HINet(BaseInference):
    """HINet for denoising image.

    Attributes:
        model (InferenceSession): ONNX model.
        input_name (str): Name of input node.
        input_shape (tuple): Shape of input node.
    """

    def __init__(self, model: str = "denoise_320_480") -> None:
        """Initialize HINet.

        Args:
            model (str, optional): Choise of model. Weight file is automatically downloaded to the current directory at the first time. Defaults to "denoise_320_480.onnx".
        """
        model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.model = InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = (480, 320)

    def predict(self, image: Image) -> np.ndarray:
        """Return denoised image.

        Args:
            image (Image): Image to be denoised in str or cv2 format.

        Returns:
            np.ndarray: Denoised image array containing two images for different denoising methods.
        """
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
        """Return the denoised image in original image size.

        Args:
            prediction (np.ndarray): Return value of predict().
            image (Image, optional): Image to be processed in str or cv2 format. Defaults to None.
            output_type (Literal[0, 1], optional): Choice of denoising method either 0 or 1. Defaults to 0.
            output_shape (Optional[Tuple[int, int]], optional): Optional tuple of int to resize the return image. Defaults to None.

        Raises:
            ValueError: If none of the original image or image size is given.
            ValueError: If output_type is not 0 or 1.

        Returns:
            np.ndarray: Denoised image in cv2 format.
        """
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
