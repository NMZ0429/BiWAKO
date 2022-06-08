import numpy as np
import cv2 as cv
from typing import List, Union
from onnxruntime import InferenceSession
from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight, get_color_map_list


WEIGHT_PATH = {
    "sparseinst_r50_m": "https://github.com/NMZ0429/Weights/releases/download/Temp/sparseinst_r50_m.onnx",
    "sparseinst_r50_l": "https://github.com/NMZ0429/Weights/releases/download/Temp/sparseinst_r50_l.onnx",
}

__all__ = ["SparseInst"]


class SparseInst(BaseInference):
    def __init__(
        self,
        model: str = "sparseinst_r50_l",
        threshold: float = 0.5,
        show_exp: bool = False,
    ) -> None:
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.IS = InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.__collect_setting(show_exp)
        self.threshold = threshold

    def __collect_setting(self, show_exp: bool = False) -> None:
        self.input_name = self.IS.get_inputs()[0].name
        self.output_names = [output.name for output in self.IS.get_outputs()]
        self.input_shape = self.IS.get_inputs()[0].shape
        self.h = self.input_shape[2]
        self.w = self.input_shape[3]
        if show_exp:
            print(f"Input shape: {self.input_shape}")
            print(f"Input name: {self.input_name}")
            print(f"Output name: {self.output_names}")

    def predict(self, img: Image) -> List[np.ndarray]:
        """Predict segmentation.

        Args:
            img (Image): Input image.

        Returns:
            List[Union[int, float]]: Predicted segmentation.
        """
        img = self._read_image(img)
        img = self._preprocess(img)
        pred = self.IS.run(self.output_names, {self.input_name: img})
        # Post process
        mask_preds = np.array(pred[0])
        scores = np.array(pred[1])
        labels = np.array(pred[2])

        # Extraction by score threshold
        mask_preds = mask_preds[scores > self.threshold]
        labels = labels[scores > self.threshold]
        scores = scores[scores > self.threshold]

        return [mask_preds, scores, labels]

    def _c_id(self, index):
        temp_index = abs(int(index + 1)) * 3
        color = (
            (37 * temp_index) % 255,
            (17 * temp_index) % 255,
            (29 * temp_index) % 255,
        )
        return color

    def render(
        self, prediction: List[np.ndarray], image: Image, **kwargs
    ) -> np.ndarray:

        img = self._read_image(image)
        image_width, image_height = img.shape[1], img.shape[0]
        masks, scores, labels = prediction
        bboxes_list = self._calc_bbox(masks, image_width, image_height)

        for index, (mask, score, label, bboxes) in enumerate(
            zip(masks, scores, labels, bboxes_list)
        ):
            if score < self.threshold:
                continue

            color = self._c_id(index)

            # Color image
            color_image = np.zeros(img.shape, dtype=np.uint8)
            color_image[:] = color

            # Resized mask image
            mask = np.stack((mask,) * 3, axis=-1).astype("uint8")
            resize_mask = cv.resize(mask, (image_width, image_height))

            # Mask addWeighted
            mask_image = np.where(resize_mask, color_image, img)
            img = cv.addWeighted(img, 0.5, mask_image, 0.5, 1.0)

            # Bounding box & Lable
            for bbox in bboxes:
                cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0))
                cv.putText(
                    img,
                    str(label),
                    (bbox[0], bbox[1] + 15),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                    cv.LINE_AA,
                )

        return img

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed image.
        """
        img = cv.resize(img, (self.w, self.h))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        return img

    def _calc_bbox(
        self, masks: np.ndarray, frame_width: int, frame_height: int
    ) -> List[List]:
        bboxes_list = []
        for mask in masks:
            contours, _ = cv.findContours(
                (mask * 255).astype("uint8"), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            bbox = []
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                x = int((x / self.w) * frame_width)
                y = int((y / self.h) * frame_height)
                w = int((w / self.w) * frame_width)
                h = int((h / self.h) * frame_height)
                bbox.append([x, y, x + w, y + h])
            bboxes_list.append(bbox)

        return bboxes_list
