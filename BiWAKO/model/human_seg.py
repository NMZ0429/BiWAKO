from typing import Optional

import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "modnet_256": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/modnet_256.onnx"
}


class MODNet(BaseInference):
    """Segmentation model trained on human portrait image.

    Attributes:
        model (onnxruntime.InferenceSession): Inference session.
        input_name (str): Name of input node.
        output_name (str): Name of output node.
        input_shape (tuple): Size of input image.
        score_th (float): Threshold for mask.
    """

    def __init__(self, model: str = "modnet_256", score_th: float = 0.5) -> None:
        """Initialize MODNet.

        Args:
            model (str, optional): Choice of model or path to the onnx file. Defaults to "modnet_256". If chosen model has not been downloaded, it will be downloaded automatically.
            score_th (float, optional): Optional threshold for mask used in self.render(). Any pixels in the mask with confidence score lower than this value will be set to 0. Defaults to 0.5.
        """
        model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.model = InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape[2:]
        self.score_th = score_th

    def predict(self, image: Image) -> np.ndarray:
        """Return mask of given image.

        Args:
            image (Image): Image to be segmented. Accept path to image or cv2 image.

        Returns:
            np.ndarray: Predicted mask in original size.
        """
        image = self._read_image(image)
        w, h = image.shape[1], image.shape[0]
        image = self._preprocess(image)
        pred = self.model.run([self.output_name], {self.input_name: image})[0]
        mask = cv.resize(pred[0], dsize=(w, h))

        return mask

    def render(
        self,
        prediction: np.ndarray,
        image: Image,
        black_out: bool = False,
        score_th: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """Apply the mask to the input image.

        Args:
            prediction (np.ndarray): Mask returned by predict().
            image (Image): Image to be segmented. Accept path to image or cv2 image.
            black_out (bool, optional): Whether to use black background. Defaults to False.
            score_th (float, optional): Optional threshold for mask. Defaults to use the value set in the constructor.

        Returns:
            np.ndarray: Segmented image in cv2 format.
        """
        score_th = score_th or self.score_th
        image = self._read_image(image)
        rendered = np.zeros_like(image, dtype=np.uint8)
        rendered[:] = (230, 216, 173)
        if black_out:
            mask = np.where(prediction > score_th, 1, 0)
            mask = np.stack((mask,) * 3, axis=-1).astype(np.uint8)
            image = (image * mask).astype(np.uint8)
        else:
            mask = np.where(prediction > score_th, 0, 1)
            mask = np.stack((mask,) * 3, axis=-1).astype(np.uint8)
            mask_image = np.where(mask, image, rendered)
            image = cv.addWeighted(image, 0.5, mask_image, 0.5, 1.0)

        return image  # type: ignore

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference. This is automatically called by predict().

        Preprocess:
            1. Resize image to the same size as the model input.
            2. Normalize image to `[-1, 1]` with mean and std of 0.5.
            3. Convert image to float32 and reshape to (1, C, H, W).

        Args:
            image (np.ndarray): Image in cv2 format.

        Returns:
            np.ndarray: Preprocessed image in numpy format.
        """
        image = cv.resize(image, dsize=self.input_shape)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = (image - 127.5) / 127.5
        image = image.transpose((2, 0, 1)).astype(np.float32)
        image = np.expand_dims(image, axis=0)

        return image
