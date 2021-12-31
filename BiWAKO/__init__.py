__version__ = "0.0.1"
__all__ = [
    "MiDASInference",
    "U2NetInference",
    "RealESRGANInference",
    "YOLO",
    "FerPlus",
    "HumanParsing",
    "HINet",
    "YuNet",
]

from .mono_depth import MiDASInference
from .segmentation import U2NetInference
from .super_resolution import RealESRGANInference
from .object_detection import YOLO
from .emotion_prediction import FerPlus
from .human_attribute import HumanParsing
from .denoise import HINet
from .face_detection import YuNet

