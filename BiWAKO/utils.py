import os
from typing import Union, Dict
import requests
from tqdm import tqdm
from pathlib import Path
import numpy as np

Image = Union[str, np.ndarray]


def download_weight(file_url: str, save_path: Union[str, Path] = "") -> str:
    """Download onnx file to save_path and return the path to the saved file.

    Args:
        file_url (str): url to onnx file
        save_path (Union[str, Path], optional): path to store the fie. Defaults to "".

    Returns:
        str: path to the saved files
    """

    if isinstance(save_path, str):
        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True)

    output_file = save_path / file_url.split("/")[-1]
    file_size = int(requests.head(file_url).headers["content-length"])

    res = requests.get(file_url, stream=True)
    pbar = tqdm(
        total=file_size, unit="B", unit_scale=True, desc=file_url.split("/")[-1]
    )
    with open(output_file, "wb") as file:
        for chunk in res.iter_content(chunk_size=1024):
            file.write(chunk)
            pbar.update(len(chunk))
        pbar.close()

    return str(output_file)


def maybe_download_weight(url_dict: Dict[str, str], key: str) -> str:
    """If "key.onnx" is in the current directory, return the path to the file. Otherwise, try to download the file from url_dict[key] and return the path to the saved file.

    Args:
        url_dict (Dict[str, str]): dictionary of url to onnx file
        key (str): key to the url_dict

    Raises:
        ValueError: If the key is not in the url_dict

    Returns:
        str: path to the saved files
    """
    if not (os.path.exists(key) or os.path.exists(key + ".onnx")):
        if key in url_dict:
            model_path = download_weight(url_dict[key])
        else:
            raise ValueError(
                f"Downloadable model not found: Available models are {list(url_dict.keys())}"
            )
    else:
        if os.path.exists(key):
            model_path = key
        else:
            model_path = key + ".onnx"

    return model_path


def print_onnx_information(onnx_path: str) -> None:
    import onnx

    model = onnx.load(onnx_path)
    for info in model.graph.input:  # type: ignore
        print(info.name, end=": ")
        # get type of input tensor
        tensor_type = info.type.tensor_type
        # check if it has a shape:
        if tensor_type.HasField("shape"):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    print(d.dim_value, end=", ")  # known dimension
                elif d.HasField("dim_param"):
                    print(d.dim_param, end=", ")  # unknown dimension with symbolic name
                else:
                    print("?", end=", ")  # unknown dimension with no name
        else:
            print("unknown rank", end="")
        print()

    for info in model.graph.output:  # type: ignore
        print(info.name, end=": ")
        # get type of output tensor
        tensor_type = info.type.tensor_type
        # check if it has a shape:
        if tensor_type.HasField("shape"):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    print(d.dim_value, end=", ")  # known dimension
                elif d.HasField("dim_param"):
                    print(d.dim_param, end=", ")  # unknown dimension with symbolic name
                else:
                    print("?", end=", ")  # unknown dimension with no name
        else:
            print("unknown rank", end="")
        print()


class Colors:
    def __init__(self):
        hex = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
