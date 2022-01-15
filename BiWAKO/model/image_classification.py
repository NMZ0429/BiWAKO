import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

from .base_inference import BaseInference
from .util.imagenet import get_imagenet_label
from .util.utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "resnet152v2": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/resnet152v2.onnx",
    "resnet101v2": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/resnet101v2.onnx",
    "resnet50v2": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/resnet50v2.onnx",
    "resnet18v2": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/resnet18v2.onnx",
}


class ResNet(BaseInference):
    """Basic ResNet V2 trained on ImageNet

    Attributes:
        model_path (str): Path to the model file. If automatic download is not enabled, this path is used to save the file.
        model (onnxruntime.InferenceSession): Inference session for the model.
        input_name (str): Name of the input node.
        output_name (str): Name of the output node.
        input_shape (tuple): Shape of the input node.
        label (dict): Dictionary of the label. The key is the class index and the value is the class name.
        mean (np.ndarray): Mean of the normalization.
        var (np.ndarray): Variance of the normalization.
    """

    def __init__(self, model: str = "resnet18v2") -> None:
        """Initialize ResNet
        
        Available models:
                "resnet152v2"
                "resnet101v2"
                "resnet50v2" or
                "resnet18v2"

        Args:
            model (str, optional): Choice of the model from the table above or path to the downloaded onnx file. If the file has not been downloaded, the automatic download is triggered. Defaults to "resnet18v2".
        """
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self.var = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.model = InferenceSession(self.model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        self.output_name = self.model.get_outputs()[0].name
        self.label = get_imagenet_label()

    def predict(self, image: Image) -> np.ndarray:
        """Return the prediction of the model

        Args:
            image (Image): Image to be predicted. Accept path or cv2 image.

        Returns:
            np.ndarray: 1 by 1000 array of the prediction. Softmax is not applied.
        """
        image = self._read_image(image)
        image = self._preprocess(image)
        pred = self.model.run([self.output_name], {self.input_name: image})[0]

        return pred

    def render(
        self, prediction: np.ndarray, image: Image, topk: int = 5, **kwargs
    ) -> np.ndarray:
        """Return the original image with the predicted class names

        Args:
            prediction (np.ndarray): Prediction returned by predict().
            image (Image): Image to be predicted. Accept path or cv2 image.
            topk (int, optional): Number of classes to display with higher probability. Defaults to 5.

        Returns:
            np.ndarray: Image with the predicted class names.
        """
        image = self._read_image(image)
        score = self._postprocess(prediction)
        for i in range(topk):
            cls = self.label[score[i]]
            cv.putText(
                image,
                cls,
                (10, 30 + i * 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        return image

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (256, 256))
        image = image[128 - 112 : 128 + 112, 128 - 112 : 128 + 112]
        # normalize
        image = (image / 255 - self.mean) / self.var
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        # channel first to channel last
        image = np.expand_dims(image, axis=0)

        return image

    def _postprocess(self, prediction: np.ndarray) -> np.ndarray:
        prediction = np.squeeze(prediction)
        y = np.exp(prediction - np.max(prediction))
        f_x = y / np.sum(np.exp(prediction))

        return np.argsort(f_x)[::-1]
