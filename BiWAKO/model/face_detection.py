from itertools import product
from typing import List, Tuple, Optional

import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "yunet_120_160": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/yunet_120_160.onnx"
}


class YuNet(BaseInference):
    """Face Detection model.

    Attributes:
        model (onnxruntime.InferenceSession): ONNX model.
        input_name (str): name of input node.
        output_names (list): names of three output nodes.
        input_shape (list): shape of input image. Set to [160, 120] by default.
        conf_th (float): confidence threshold. Set to 0.6 by default.
        nms_th (float): non-maximum suppression threshold. Set to 0.3 by default.
        topk (int): keep top-k results. Set to 5000 by default.
        priors (np.ndarray): prior boxes.
    """

    def __init__(
        self,
        model: str = "yunet_120_160",
        input_shape: Optional[Tuple[int, int]] = None,
        conf_th: float = 0.6,
        nms_th: float = 0.3,
        topk: int = 5000,
        keep_topk: int = 750,
    ):
        """Initialize YuNet.

        Args:
            model (str): model name. Set to "yunet_120_160" by default.
            input_shape (tuple, optional): Input image shape. Defaults to (160, 120).
            conf_th (float, optional): Confidence level threshold. Defaults to 0.6.
            nms_th (float, optional): NMS threshold. Defaults to 0.3.
            topk (int, optional): Number of faces to detect. Defaults to 5000.
            keep_topk (int, optional): Number of predictions to save. Defaults to 750.
        """
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.model = InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [self.model.get_outputs()[i].name for i in range(3)]

        if input_shape is None:
            input_shape = (160, 120)
        self.input_shape = input_shape  # [w, h]
        self.w, self.h = input_shape
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.topk = topk
        self.keep_topk = keep_topk

        self.MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self.STEPS = [8, 16, 32, 64]
        self.VARIANCE = [0.1, 0.2]
        self.priors = self._generate_priors()

    def predict(self, image: Image) -> Tuple[list, list, list]:
        """Return the face detection result.

        The prediction result is a tuple of three lists.
        First list is bounding boxes, second list is landmarks, and third list is scores.
        For example, accesing 2nd parson's bounding box is done by prediction`[1][0]`, and prediction`[1][1]` is landmarks of 2nd person.

        Args:
            image (Image): image to be detected. Accept path or cv2 image.

        Returns:
            Tuple[list, list, list]: Tuple of three lists of bounding box, landmark, and score.
        """
        img = self._read_image(image)
        img = self._preprocess(img)
        pred = self.model.run(self.output_names, {self.input_name: img})
        bboxes, landmarks, scores = self._postprocess(pred)

        return bboxes, landmarks, scores

    def render(self, prediction: tuple, image: Image) -> np.ndarray:
        """Render the bounding box and landmarks on the original image.

        Args:
            prediction (tuple): prediction result returned by predict().
            image (Image): original image in str or cv2 image.

        Returns:
            np.ndarray: Original image with bounding box and landmarks.
        """
        bboxes, landmarks, scores = prediction
        image = self._read_image(image)
        image_width, image_height = image.shape[1], image.shape[0]

        for bbox, landmark, score in zip(bboxes, landmarks, scores):
            if self.conf_th > score:
                continue

            x1 = int(image_width * (bbox[0] / self.w))
            y1 = int(image_height * (bbox[1] / self.h))
            x2 = int(image_width * (bbox[2] / self.w)) + x1
            y2 = int(image_height * (bbox[3] / self.h)) + y1

            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(
                image,
                "{:.4f}".format(score),
                (x1 + 40, y1 - 20),
                cv.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 255, 0),
            )

            for _, landmark_point in enumerate(landmark):
                x = int(image_width * (landmark_point[0] / self.w))
                y = int(image_height * (landmark_point[1] / self.h))
                cv.circle(image, (x, y), 2, (0, 255, 0), 2)

        return image

    def _preprocess(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.w, self.h))
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, self.h, self.w)

        return image

    def _postprocess(
        self, result
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        dets = self._decode(result)
        selected = cv.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self.conf_th,
            nms_threshold=self.nms_th,
            top_k=self.topk,
        )
        scores = []
        bboxes = []
        landmarks = []
        if len(selected) > 0:
            dets = dets[selected]
            if len(dets.shape) == 3:
                dets = np.squeeze(dets, axis=1)
            for det in dets[: self.keep_topk]:
                scores.append(det[-1])
                bboxes.append(det[0:4].astype(np.int32))
                landmarks.append(det[4:14].astype(np.int32).reshape((5, 2)))

        return bboxes, landmarks, scores

    def _decode(self, result):
        loc, conf, iou = result

        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]

        _idx = np.where(iou_scores < 0.0)
        iou_scores[_idx] = 0.0
        _idx = np.where(iou_scores > 1.0)
        iou_scores[_idx] = 1.0
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = np.array(self.input_shape)

        bboxes = np.hstack(
            (
                (
                    self.priors[:, 0:2]
                    + loc[:, 0:2] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
                (self.priors[:, 2:4] * np.exp(loc[:, 2:4] * self.VARIANCE)) * scale,
            )
        )
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        landmarks = np.hstack(
            (
                (
                    self.priors[:, 0:2]
                    + loc[:, 4:6] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + loc[:, 6:8] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + loc[:, 8:10] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + loc[:, 10:12] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + loc[:, 12:14] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
            )
        )

        dets = np.hstack((bboxes, landmarks, scores))

        return dets

    def _generate_priors(self) -> np.ndarray:
        """Initialize prior bboxes.

        Returns:
            np.ndarray: Prior bboxes.
        """
        w, h = self.input_shape

        feature_map_2th = [int(int((h + 1) / 2) / 2), int(int((w + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2), int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2), int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2), int(feature_map_4th[1] / 2)]
        feature_map_6th = [int(feature_map_5th[0] / 2), int(feature_map_5th[1] / 2)]

        feature_maps = [
            feature_map_3th,
            feature_map_4th,
            feature_map_5th,
            feature_map_6th,
        ]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.MIN_SIZES[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self.STEPS[k] / w
                    cy = (i + 0.5) * self.STEPS[k] / h

                    priors.append([cx, cy, s_kx, s_ky])

        return np.array(priors, dtype=np.float32)
