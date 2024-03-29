from typing import List, Tuple

import torch
import torchvision
import numpy as np
import cv2
import onnxruntime as rt

from .base_inference import BaseInference
from .util.utils import Image, maybe_download_weight, Colors

WEIGHT_PATH = {
    "yolo_nano": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/yolo_nano.onnx",
    "yolo_s": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/yolo_s.onnx",
    "yolo_xl": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/yolo_xl.onnx",
    "yolo_extreme": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/yolo_extreme.onnx",
    "yolo_nano_smp": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/yolo_nano_smp.onnx",
    "yolo_s_smp": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/yolo_s_smp.onnx",
    "yolo_xl_smp": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/yolo_lx_smp.onnx",
    "yolo_extreme_smp": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/yolo_extreme_smp.onnx",
}


class YOLO(BaseInference):
    """YOLOv5 onnx model.

    Attributes:
        model_path (str): Path to the onnx file. If auto download is triggered, the file is downloaded to this path.
        session (onnxruntime.InferenceSession): Inference session.
        input_name (str): Name of the input node.
        output_name (str): Name of the output node.
        input_shape (tuple): Shape of the input image. Set accordingly to the model.
        coco_label (list): List of coco 80 labels.
        colors (Colors): Color palette written by Ultralytics at https://github.com/ultralytics/yolov5/blob/a3d5f1d3e36d8e023806da0f0c744eef02591c9b/utils/plots.py
    """

    def __init__(self, model: str = "yolo_nano") -> None:
        """Initialize the model.

        Args:
            model (str): Model type to be used. Also accept path to the onnx file. If the model is not found, it will be downloaded automatically. Currently `[yolo_nano, yolo_s, yolo_xl and yolo_extreme]` are supported. Default is `yolo_nano`. Adding `_smp` to the model name will use the simplified model.

        Examples:
            >>> model = YOLO("yolo_nano_smp")
            downloading yolo_nano_smp.onnx to yolo_nano_smp.onnx
            100%|██████████| 7.57M/7.57M [00:01<00:00, 6.89MB/s]
        """
        self.model_path = maybe_download_weight(WEIGHT_PATH, model)
        self.session = rt.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = (1280, 1280) if ("yolo_extreme" in model) else (640, 640)
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

    def predict(self, image: Image) -> np.ndarray:
        """Return the prediction of the model.

        Args:
            image (Image): Image to be predicted. Accept str or cv2 image.

        Returns:
            np.ndarray: n by 6 array where 2nd dimension is xyxy with label and confidence.
        """
        img = self._read_image(image)
        orig_size = img.shape
        img = self._preprocess(img)
        pred = self.session.run([self.output_name], {self.input_name: img})
        pred = torch.tensor(pred)

        clf = self.non_max_suppression(pred)
        clf = clf[0]  # batch processing is not currently supported
        clf[:, :4] = self._scale_coords(self.input_shape, clf[:, :4], orig_size).round()

        return clf.numpy()

    def render(self, prediction: np.ndarray, image: Image) -> np.ndarray:
        """Return the original image with predicted bounding boxes.

        Args:
            prediction (np.ndarray): Prediction of the model.
            image (Image): Image to be predicted. Accept str or cv2 image.

        Returns:
            np.ndarray: Image with predicted bounding boxes in cv2 format.
        """
        rtn = self._read_image(image)

        lw = max(round(sum(rtn.shape) / 2 * 0.003), 2)
        for pred in prediction:
            x1, y1, x2, y2 = pred[:4].astype("int")
            cls = int(pred[5])
            c = self.colors(cls, True)
            cv2.rectangle(
                rtn,
                (x1, y1),
                (x2, y2),
                color=c,
                thickness=lw,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                rtn,
                self.coco_label[cls],
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                c,
                2,
            )
            # print(f"{self.coco_label[cls]} {x1} {y1} {x2} {y2}")

        return rtn

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # img = cv2.resize(image, dsize=(640, 640))
        img = self.letterbox(image, self.input_shape, stride=64)[0]
        img = img.astype("float32")
        img = img / 255.0
        img = img.transpose(2, 0, 1)[::-1]
        img = np.expand_dims(img, axis=0)

        return np.ascontiguousarray(img)

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=1000,
    ) -> List[torch.Tensor]:
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert (
            0 <= conf_thres <= 1
        ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert (
            0 <= iou_thres <= 1
        ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

        # Settings
        _, max_wh = (
            2,
            4096,
        )  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]  # type: ignore

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = (
                x[:, :4] + c,
                x[:, 4],
            )  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]

        return output

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2

        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (
            (
                torch.min(box1[:, None, 2:], box2[:, 2:])  # type: ignore
                - torch.max(box1[:, None, :2], box2[:, :2])  # type: ignore
            )
            .clamp(0)
            .prod(2)
        )
        return inter / (
            area1[:, None] + area2 - inter
        )  # iou = inter / (area1 + area2 - inter)

    def letterbox(
        self,
        im,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=False,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (
            new_shape[1] - new_unpad[0],
            new_shape[0] - new_unpad[1],
        )  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im, ratio, (dw, dh)

    def _scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(
                img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
            )  # gain  = old / new
            pad = (
                (img1_shape[1] - img0_shape[1] * gain) / 2,
                (img1_shape[0] - img0_shape[0] * gain) / 2,
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self._clip_coords(coords, img0_shape)
        return coords

    def _clip_coords(self, boxes, shape: Tuple[int, int]) -> None:
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
