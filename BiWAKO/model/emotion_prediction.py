import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "ferplus8": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/ferplus8.onnx"
}


class FerPlus(BaseInference):
    """Emotion prediction model.

    The model requires the input image to be trimmed around the face.
    Use YuNet to detect the face and crop the image around it.

    Attributes:
        model_path (str): Path to the model weights. If automatic download is triggered, this path is used to save the model.
        session (onnxruntime.InferenceSession): The inference session.
        input_name (str): The name of the input node.
        output_name (str): The name of the output node.
        emotion_table (list): A list of emotions trained.
    """

    def __init__(self, model: str = "ferplus8"):
        """Initialize the model.

        Args:
            model (str): The name of the model. Also accept the path to the onnx file. If not found, the model will be downloaded. Currently only support "ferplus8".
        """
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.session = InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.emotion_table = np.array(
            [
                "neutral",
                "happiness",
                "surprise",
                "sadness",
                "anger",
                "disgust",
                "fear",
                "contempt",
            ]
        )

    def predict(self, image: Image) -> np.ndarray:
        """Return the array of the confidences of each predction.

        Args:
            image (Image): Image to be processed. Accept the path to the image or cv2 image.

        Returns:
            np.ndarray: The array of the confidences of each predction.
        """

        image = self._read_image(image)
        image = self._preprocess(image)
        prediction = self.session.run([self.output_name], {self.input_name: image})
        prediction = self._postprocess(prediction)

        return prediction

    def render(
        self, prediction: np.ndarray, image: Image, *args, **kwargs
    ) -> np.ndarray:
        """Return the list of emotions and their confidences in string over the image.

        Args:
            prediction (np.ndarray): The array of the confidences of each predction.
            image (Image): Image to be processed. Accept the path to the image or cv2 image. Not actually required.

        Returns:
            np.ndarray:  Image with the list of emotions and their confidences in string.
        """
        img = self._read_image(image)
        lw = max(round(sum(img.shape) / 2 * 0.003), 2)
        # convert prediction to emotion_table
        classes = np.argsort(prediction)[::-1]
        cv.putText(
            img,
            self.emotion_table[classes[0]],
            (10, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            lw,
            (180, 233, 84),
            thickness=2,
        )

        return img

    def _preprocess(self, image: np.ndarray) -> np.ndarray:

        # bgr to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = self.resize_and_gray(image)

        return image

    def _postprocess(self, prediction: np.ndarray) -> np.ndarray:
        prob = self.softmax(prediction)
        prob = np.squeeze(prob)

        return prob

    def softmax(self, x):

        y = np.exp(x - np.max(x))
        f_x = y / np.sum(np.exp(x))
        return f_x

    def rgb2gray(self, rgb):
        """Convert the input image into grayscale"""
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def resize_img(self, img_to_resize):
        """Resize image to FER+ model input dimensions"""
        r_img = cv.resize(img_to_resize, dsize=(64, 64), interpolation=cv.INTER_AREA)
        r_img.resize((1, 1, 64, 64))
        return r_img

    def resize_and_gray(self, img_to_preprocess):
        """Resize input images and convert them to grayscale."""
        """if img_to_preprocess.shape == (64, 64):
            img_to_preprocess.resize((1, 1, 64, 64))
            return img_to_preprocess"""

        grayscale = self.rgb2gray(img_to_preprocess)
        processed_img = self.resize_img(grayscale)
        return processed_img.astype(np.float32)
