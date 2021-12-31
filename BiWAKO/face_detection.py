from typing import Any, ClassVar, Literal, Optional, Tuple
from itertools import product

import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

from .base_inference import BaseInference
from .utils import Image, maybe_download_weight

WEIGHT_PATH = {
    "yunet_120_160": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/yunet_120_160.onnx"
}


class YuNet(BaseInference):

    # Feature map用定義
    MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    STEPS = [8, 16, 32, 64]
    VARIANCE = [0.1, 0.2]

    def __init__(
        self,
        model: Literal["yunet_120_160"] = "yunet_120_160",
        input_shape=[160, 120],
        conf_th=0.6,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
    ):
        model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.onnx_session = InferenceSession(model_path)

        self.input_name = self.onnx_session.get_inputs()[0].name
        output_name_01 = self.onnx_session.get_outputs()[0].name
        output_name_02 = self.onnx_session.get_outputs()[1].name
        output_name_03 = self.onnx_session.get_outputs()[2].name
        self.output_names = [output_name_01, output_name_02, output_name_03]

        # 各種設定
        self.input_shape = input_shape  # [w, h]
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.topk = topk
        self.keep_topk = keep_topk

        # priors生成
        self.priors = self._generate_priors()

    def predict(self, image: Image) -> Any:
        img = self._read_image(image)
        img = self._preprocess(img)

        pred = self.onnx_session.run(self.output_names, {self.input_name: img})

        bboxes, landmarks, scores = self._postprocess(pred)

        return bboxes, landmarks, scores

    def render(
        self,
        prediction: list,
        image: Image,
        score_th: float = 0.6,
        input_shape=(160, 120),
    ) -> np.ndarray:
        bboxes, landmarks, scores = prediction
        image = self._read_image(image)
        image_width, image_height = image.shape[1], image.shape[0]

        for bbox, landmark, score in zip(bboxes, landmarks, scores):
            if score_th > score:
                continue

            # 顔バウンディングボックス
            x1 = int(image_width * (bbox[0] / input_shape[0]))
            y1 = int(image_height * (bbox[1] / input_shape[1]))
            x2 = int(image_width * (bbox[2] / input_shape[0])) + x1
            y2 = int(image_height * (bbox[3] / input_shape[1])) + y1

            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # スコア
            cv.putText(
                image,
                "{:.4f}".format(score),
                (x1 + 40, y1 - 20),
                cv.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 255, 0),
            )

            # 顔キーポイント
            for _, landmark_point in enumerate(landmark):
                x = int(image_width * (landmark_point[0] / input_shape[0]))
                y = int(image_height * (landmark_point[1] / input_shape[1]))
                cv.circle(image, (x, y), 2, (0, 255, 0), 2)

        return image

    def _generate_priors(self) -> np.ndarray:
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

    def _preprocess(self, image):
        # BGR -> RGB 変換
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # リサイズ
        image = cv.resize(
            image,
            (self.input_shape[0], self.input_shape[1]),
            interpolation=cv.INTER_LINEAR,
        )

        # リシェイプ
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, self.input_shape[1], self.input_shape[0])

        return image

    def _postprocess(self, result):
        # 結果デコード
        dets = self._decode(result)

        # NMS
        keepIdx = cv.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self.conf_th,
            nms_threshold=self.nms_th,
            top_k=self.topk,
        )

        # bboxes, landmarks, scores へ成形
        scores = []
        bboxes = []
        landmarks = []
        if len(keepIdx) > 0:
            dets = dets[keepIdx]
            if len(dets.shape) == 3:
                dets = np.squeeze(dets, axis=1)
            for det in dets[: self.keep_topk]:
                scores.append(det[-1])
                bboxes.append(det[0:4].astype(np.int32))
                landmarks.append(det[4:14].astype(np.int32).reshape((5, 2)))

        return bboxes, landmarks, scores

    def _decode(self, result):
        loc, conf, iou = result

        # スコア取得
        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]

        _idx = np.where(iou_scores < 0.0)
        iou_scores[_idx] = 0.0
        _idx = np.where(iou_scores > 1.0)
        iou_scores[_idx] = 1.0
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = np.array(self.input_shape)

        # バウンディングボックス取得
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

        # ランドマーク取得
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