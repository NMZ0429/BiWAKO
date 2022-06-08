import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight, get_color_map_list

WEIGHT_PATH = {
    "fast_scnn384": "https://github.com/NMZ0429/Weights/releases/download/Temp/fast_scnn384.onnx",
    "fast_scnn7681344": "https://github.com/NMZ0429/Weights/releases/download/Temp/fast_scnn7681344.onnx",
}


class FastSCNN(BaseInference):
    """Semantic Segmentation

    Attributes:
        model_path (str): Path to the model file.
        model (onnxruntime.InferenceSession): Inference session.
        input_shape (tuple): Input shape of the model. Set to (384, 384) for fast_scnn384 and (1344, 768) for fast_scnn7681344.
        input_name (str): Name of the input node.
        output_name (str): Name of the output node.
        mean (np.ndarray): Mean value of the dataset.
        std (np.ndarray): Standard deviation of the dataset.
        c_map: Color map for the semantic segmentation 19 object classes.
    """

    def __init__(self, model: str = "fast_scnn384", **kwargs) -> None:
        """Initialize FastSCNN model.

        Args:
            model (str, optional): Choice of model. Accept model name or path to the downloaded onnx file. If onnx file has not been downloaded,
                                    it will be downloaded automatically. Currently avaiable models are `[fast_scnn384, fast_scnn7681344]`.
                                    Defaults to "fast_scnn384".
        """
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.model = InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        self.input_shape = (384, 384) if "384" in model else (1344, 768)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])
        self.c_map = get_color_map_list(19)

    def predict(self, image: Image) -> np.ndarray:
        """Return the prediction map. The last channel has 19 classes.

        Args:
            image (Image): Image to be predicted. Accept path to the image or cv2 image.

        Returns:
            np.ndarray: Prediction map in the shape of (height, width, 19).
        """
        image = self._read_image(image)
        dsize = (image.shape[1], image.shape[0])
        image = self._preprocess(image)
        prediction = self.model.run([self.output_name], {self.input_name: image})[0]
        prediction = np.squeeze(prediction).transpose(1, 2, 0)
        prediction = cv.resize(prediction, dsize)

        return np.argmax(prediction, axis=2)

    def render(self, prediction: np.ndarray, image: Image, **kwargs) -> np.ndarray:
        """Apply the prediction map to the image.

        Args:
            prediction (np.ndarray): Prediction map retuned by `predict`.
            image (Image): Image to be rendered. Accept path to the image or cv2 image.

        Returns:
            np.ndarray: Rendered image.
        """
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
        """Preprocess the image. Automatically called.

        Preprocess:
            1. Resize to the input shape.
            2. To RGB and normalize.
            3. Add the mean and divide by the standard deviation.
            4. Channel swap and float32.
            5. Add batch dimension.

        Args:
            image (np.ndarray): Image to be preprocessed.

        Returns:
            np.ndarray: Preprocessed image.
        """
        image = cv.resize(image, self.input_shape)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image /= 255.0  # type: ignore
        image = (image - self.mean) / self.std  # type: ignore
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return np.expand_dims(image, axis=0)
