import cv2 as cv
import numpy as np
import onnxruntime as rt


from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight


WEIGHT_PATH = {
    "suim_net_3248": "https://github.com/NMZ0429/Weights/releases/download/Temp/suim_net_3248.onnx",
    "suim_rsb_72128": "https://github.com/NMZ0429/Weights/releases/download/Temp/suim_rsb_72128.onnx",
    "suim_vgg_25632": "https://github.com/NMZ0429/Weights/releases/download/Temp/suim_vgg_25632.onnx",
    "suim_vgg_72128": "https://github.com/NMZ0429/Weights/releases/download/Temp/suim_vgg_72128.onnx",
}


class SUIMNet(BaseInference):
    """Semantic segmentation model for diver's view dataset.

    Attributes:
        model_path (str): Path to the onnx file. If automatic download is triggered, it is downloaded to this path.
        session (onnxruntime.InferenceSession): Inference session.
        h (int): Height of the input image.
        w (int): Width of the input image.
        input_name (str): Name of the input node.
        output_name (str): Name of the output node.
        c_map: Color map for the segmentation.
    """

    def __init__(
        self, model: str = "suim_rsb_72128", num_classes: int = 5, **kwargs
    ) -> None:
        """Initialize the SUIMNet model.

        Args:
            model (str, optional): Choice of the model or path to the downloaded onnx file. If the model hasn't been downloaded, it is automatically downloaded to this path. Defaults to "suim_rsb_72128".
            num_classes (int, optional): Number of classes to segmentate. Defaults to 5.
        """
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.session = rt.InferenceSession(self.model_path)
        self.h, self.w = self.session.get_inputs()[0].shape[2:4]
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.c_map = self._get_color_map_list(num_classes)

    def predict(self, image: Image) -> np.ndarray:
        """Return the segmentation map of the image.

        Args:
            image (Image): Image to segmentate. Accept path to the image or cv2 image.

        Returns:
            np.ndarray: Segmentation map of shape (h, w, num_classes). Each element is a confidence of the class.
        """
        image = self._read_image(image)
        h, w = image.shape[:2]
        image = self._preprocess(image)
        pred = self.session.run([self.output_name], {self.input_name: image})[0]
        pred = np.squeeze(pred)
        pred = cv.resize(pred, (w, h))
        pred = pred.transpose(2, 0, 1)

        return pred

    def render(self, prediction: np.ndarray, image: Image, **kwargs) -> np.ndarray:
        """Apply the segmentation map to the image.

        Args:
            prediction (np.ndarray): Segmentation map of shape (h, w, num_classes) returned by predict().
            image (Image): Input image. Accept path to the image or cv2 image.

        Returns:
            np.ndarray: Rendered image in cv2 format.
        """
        img = self._read_image(image)

        for i, m in enumerate(prediction):
            bg_image = np.zeros(img.shape, dtype=np.uint8)
            bg_image[:] = (
                self.c_map[i * 3 + 0],
                self.c_map[i * 3 + 1],
                self.c_map[i * 3 + 2],
            )
            mask = np.where(m > 0.5, 0, 1)
            mask = np.stack((mask,) * 3, axis=-1).astype("uint8")
            mask_image = np.where(mask, img, bg_image)
            img = cv.addWeighted(img, 0.25, mask_image, 0.75, 1.0)

        return img

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, (self.w, self.h))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image / 255.0
        image = image.transpose((2, 0, 1)).astype(np.float32)

        return np.expand_dims(image, axis=0)

    def _get_color_map_list(self, num_classes, custom_color=None):
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
