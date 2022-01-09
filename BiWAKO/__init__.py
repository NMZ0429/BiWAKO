__version__ = "0.0.2"
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
]

from .mono_depth import MiDAS
from .segmentation import U2Net
from .super_resolution import RealESRGAN
from .object_detection import YOLO
from .emotion_prediction import FerPlus
from .human_attribute import HumanParsing
from .denoise import HINet
from .face_detection import YuNet
from .anime_gan import AnimeGAN
from .image_classification import ResNet

