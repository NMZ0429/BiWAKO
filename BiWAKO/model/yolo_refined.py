from typing import List

import cv2 as cv
import numpy as np
import onnxruntime as rt

from .base_inference import BaseInference
from .util.utils import Colors, Image, maybe_download_weight

WEIGHT_PATH = {
    "yolo_nano": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_nano.onnx",
    "yolo_nano6": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_nano6.onnx",
    "yolo_nano_simp": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_nano_simp.onnx",
    "yolo_nano6_simp": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_nano6_simp.onnx",
    "yolo_s": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_s.onnx",
    "yolo_s6": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_s6.onnx",
    "yolo_s_simp": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_s_simp.onnx",
    "yolo_s6_simp": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_s6_simp.onnx",
    "yolo_m": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_m.onnx",
    "yolo_m6": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_m6.onnx",
    "yolo_m_simp": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_m_simp.onnx",
    "yolo_m6_simp": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_m6_simp.onnx",
    "yolo_l": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_l.onnx",
    "yolo_l_simp": "https://github.com/NMZ0429/Weights/releases/download/Temp/yolo_l_simp.onnx",
}


class YOLO2(BaseInference):
    def __init__(self, model: str = "yolo_nano_simp") -> None:
        self.model_path = maybe_download_weight(url_dict=WEIGHT_PATH, key=model)
        self.session = rt.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.coco_label = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        self.colors = Colors()

    def predict(self, image: Image) -> List[np.ndarray]:
        """Return the prediction of the model.

        Args:
            image (Image): The image to be predicted. Accept both path and array in cv2 format.

        Returns:
            List[np.ndarray]: The prediction of the model in the format of `[scores, labels, boxes]`.
        """
        image = self._preprocess(self._read_image(image))
        return self.session.run(None, {self.input_name: image})

    def render(
        self, prediction: List[np.ndarray], image: Image, **kwargs
    ) -> np.ndarray:
        image = self._read_image(image)
        prediction[2] = prediction[2].astype(int, copy=False)

        for i in range(len(prediction[0])):
            label = prediction[1][i]
            xyxy = prediction[2][i]
            c = self.colors(label, True)

            cv.rectangle(
                img=image, pt1=(xyxy[0], xyxy[1]), pt2=(xyxy[2], xyxy[3]), color=c,
            )
            cv.putText(
                image,
                self.coco_label[label],
                (xyxy[0], xyxy[1] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                c,
                2,
            )

        return image

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return np.transpose((image.astype(np.float32, copy=True) / 255.0), (2, 0, 1))

