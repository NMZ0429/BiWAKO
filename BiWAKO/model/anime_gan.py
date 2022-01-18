from typing import Literal, Tuple

import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "animeGAN512": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/animeGAN512.onnx"
}


class AnimeGAN(BaseInference):
    """Style Transfer GAN trained for Anime.

    Attributes:
        model_path (str): Path to ONNX model file. If the file is automatically downloaded, the destination path is saved to this.
        model (InferenceSession): ONNX model.
        input_name (str): Name of input node.
        output_name (str): Name of output node.
        input_size (int): Size of input image. Set to 512.
    """

    def __init__(self, model: Literal["animeGAN512"] = "animeGAN512") -> None:
        """Initialize AnimeGAN model.

        Args:
            model (Literal[, optional): Either path to the downloaded model or name of the model to trigger automatic download. Defaults to "animeGAN512".
        """
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.model = InferenceSession(self.model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.ouput_name = self.model.get_outputs()[0].name
        self.input_size = 512

    def predict(self, image: Image) -> np.ndarray:
        """Return the predicted image from the AnimeGAN model.

        Args:
            image (Image): Image to predict in str or cv2 image format.

        Returns:
            np.ndarray: Predicted image of size 512*512 in cv2 image format
        """
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
        """Return the predicted image in original size.

        Args:
            prediction (np.ndarray): Predicted image of size 512*512 in cv2 image format.
            image (Image, optional): Original image passed to predict(). Defaults to None.
            input_size (Tuple[int, int], optional): Optional tuple of int to resize. Defaults to None.

        Raises:
            ValueError: If none of image or input_size is provided.

        Returns:
            np.ndarray: Predicted image in original size.
        """
        if not (image is None):
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
