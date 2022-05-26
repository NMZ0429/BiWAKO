import cv2
import numpy as np
import onnxruntime as rt

from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "mono_depth_small": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/mono_depth_small.onnx",
    "mono_depth_large": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/mono_depth_large.onnx",
}

__all__ = ["MiDAS"]


class MiDAS(BaseInference):
    """MonoDepth prediction model.

    Model:
        This is a pretrained MiDASv1 model. Currently available models are:
            - mono_depth_small
            - mono_depth_large

    Attributes:
        model_path (str): Path to model file. If automatic download is triggered, this path is used to save the model.
        session (onnxruntime.InferenceSession): Inference session.
        input_name (str): Input node name.
        output_name (str): Output node name.
        input_shape (tuple): Input shape.
        h (int): Alias of input_shape[2].
        w (int): Alias of input_shape[3].
    """

    def __init__(
        self,
        model: str = "mono_depth_small",
        show_exp: bool = False,
    ) -> None:
        """Initialize model.

        Args:
            model (str, optional): Model name or path to the downloaded onnx file. Defaults to "mono_depth_small".
                Onnx file is downloaded automatically.
            show_exp (bool, optional): True to display expected input size. Defaults to False.

        Examples:
            >>> model = MiDAS("mono_depth_large")
            downloading mono_depth_large.onnx to mono_depth_large.onnx
            100%|██████████| 416M/416M [02:11<00:00, 3.47MB/s]

            >>> model = MiDAS("weights/midas/mono_depth_small.onnx") # download to a specific directory
            downloading mono_depth_small.onnx to weights/midas/mono_depth_small.onnx
            100%|██████████| 66.8M/66.8M [03:26<00:00, 323kB/s]
        """
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.session = rt.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        self.__collect_setting(show_exp)

    def predict(self, img: Image) -> np.ndarray:
        """Predict.

        Args:
            img (Union[str, np.ndarray]): image path or numpy array in cv2 format

        Returns:
            np.ndarray: predicted depthmap
        """
        img = self._read_image(img)
        img = self._preprocess(img)
        output = self.session.run([self.output_name], {self.input_name: img})[0]
        return output

    def render(self, prediction: np.ndarray, query: Image) -> np.ndarray:
        """Return the resized depth map in cv2 foramt.

        Args:
            prediction (np.ndarray): predicted depthmap
            query (Union[str, np.ndarray]): query image path or numpy array in cv2 format used for resizing.

        Returns:
            np.ndarray: Resized depthmap
        """
        orig_img_size = self._read_image(query).shape[:2][::-1]

        prediction = self.normalize_depth(prediction).transpose((1, 2, 0))
        prediction = cv2.resize(
            prediction, orig_img_size, interpolation=cv2.INTER_CUBIC
        )
        prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

        return prediction

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image.

        0. BGR to RGB
        1. Scale to [0, 1]
        1. Resize to training size.
        2. Normalize.
        3. Add batch dimension.
        4. Convert to float32.

        Args:
            img (np.ndarray): image in cv2 format
        """
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = cv2.resize(img, (self.w, self.h))
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def __collect_setting(self, verbose: bool = True) -> None:
        """Collect setting from onnx file.

        Args:
            verbose (bool, optional): True to display setting. Defaults to True.
        """
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.h, self.w = self.input_shape[2], self.input_shape[3]

        if verbose:
            print(f"Input shape: {self.input_shape}")
            print("Normalization expected.")

    def normalize_depth(self, depth: np.ndarray, bits=1) -> np.ndarray:
        """Nomralize depthmap by given bits.

        Args:
            depth (np.ndarray): Depth map predicted by model.
            bits (int, optional): Number of bits used to normalize the depth map. Defaults to 1.

        Raises:
            ValueError: If bits is not in [1, 8].

        Returns:
            np.ndarray: Normalized depth map.
        """
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2 ** (8 * bits)) - 1
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)
        if bits == 1:
            return out.astype("uint8")
        elif bits == 2:
            return out.astype("uint16")
        else:
            raise ValueError("bits must be 1 or 2")
