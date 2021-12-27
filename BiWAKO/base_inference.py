from abc import ABC, abstractmethod

import cv2 as cv
from numpy import ndarray

from .utils import Image


class BaseInference(ABC):
    @abstractmethod
    def predict(self, image: Image) -> ndarray:
        pass

    @abstractmethod
    def render(self, prediction: ndarray, image: Image) -> ndarray:
        pass

    @abstractmethod
    def _preprocess(self, image: ndarray) -> ndarray:
        pass

    def _read_image(self, image=Image) -> ndarray:
        """Return cv2 image object if given path. Otherwise return numpy array.

        Args:
            Union[str, np.ndarray]: image path or numpy array in cv2 format

        Returns:
            np.ndarray: RGB image.
        """
        if isinstance(image, str):
            rtn = cv.imread(image)
        elif isinstance(image, ndarray):
            rtn = image
            # print("Image is pre-loaded. Make sure that the image is in cv2 format.")
        else:
            raise ValueError(f"image must be str or np.ndarray. got {type(image)}")

        return rtn
