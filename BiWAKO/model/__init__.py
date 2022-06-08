from .mono_depth import MiDAS
from .segmentation import U2Net
from .super_resolution import RealESRGAN
from .emotion_prediction import FerPlus
from .human_attribute import HumanParsing
from .denoise import HINet
from .face_detection import YuNet
from .anime_gan import AnimeGAN
from .image_classification import ResNet
from .human_seg import MODNet
from .yolo_refined import YOLO2
from .semantic_segmentation import FastSCNN
from .suim_net import SUIMNet
from .instance_seg import SparseInst


# Exclude old yolo implementation if pytorch is not installed
try:
    import torch
    from .object_detection import YOLO

    __all__ = [
        "MiDAS",
        "U2Net",
        "RealESRGAN",
        "YOLO",
        "FerPlus",
        "HumanParsing",
        "HINet",
        "YuNet",
        "AnimeGAN",
        "ResNet",
        "MODNet",
        "YOLO2",
        "FastSCNN",
        "SUIMNet",
        "SparseInst",
    ]

except ModuleNotFoundError:
    __all__ = [
        "MiDAS",
        "U2Net",
        "RealESRGAN",
        "FerPlus",
        "HumanParsing",
        "HINet",
        "YuNet",
        "AnimeGAN",
        "ResNet",
        "MODNet",
        "YOLO2",
        "FastSCNN",
        "SUIMNet",
        "SparseInst",
    ]
