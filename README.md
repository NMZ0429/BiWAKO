<div align="center">

<img src="docs/img/biwako.png" width="450">

### The home of moduled onnx models

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/NaMAZU)](https://pypi.org/project/NaMAZU/)
[![PyPI version](https://badge.fury.io/py/NaMAZU.svg)](https://badge.fury.io/py/NaMAZU)
[![PyPI Status](https://pepy.tech/badge/NaMAZU)](https://pepy.tech/project/NaMAZU)
[![license](https://img.shields.io/badge/License-GPL--3.0-informational)](https://github.com/NMZ0429/NaMAZU/blob/main/LICENSE)
![onnx](https://img.shields.io/badge/ONNX-1.10-005CED.svg?logo=ONNX&style=popout)

* * *

</div>

# BiWAKO

## Docs

**Refer to the [docs](https://nmz0429.github.io/BiWAKO/)**

This README is the copy of the front page of the docs site and some sections are not properly rendered.

## 0. Introduction

This repository offers

1. **Models**: Trained state-of-the-art models for various vision tasks in ONNXRuntime backend
2. **No-Code Modules**: Easy interface to use those models for both prediction and visualizing output. No coding is needed. The interface is universal among all models in this library.
3. **Extentiability**: Customizable modules to use it for applications such as realtime inference.

## 1. Installation

Install directly from this repository.

```sh
$cd BiWAKO
$pip install -e .
```

!!! warning
    Downloading from pip server is currently suspended in order to protect weight files. We will update it soon.

## 2. Usage

<img src="docs/img/basic_usage.png" width="700">

No matter which model you use, these interface is the same.

```python
import BiWAKO

# 1. Initialize Model
model = BiWAKO.MiDAS(model="mono_depth_small")

# 2. Feed Image (accept cv2 image or path to the image)
prediction = model.predict(image_or_image_path)

# 3. Visiualize result as a cv2 image
result_img = model.render(prediction, image_or_image_path)
```

More specifically...

1. Instantiate model with `BiWAKO.ModelName(weight)`. The `ModelName` and `weight` corresponding to the task you want to work on can be found at the table in the next section. Weight file is automaticaly downloaded.
2. call `predict(image)`. `image` can be either path to the image or cv2 image array.
3. call `render(prediction, image)`. `prediction` is the return value of `predict()` method and `image` is the same as above. Some model takes optional arguments to control details in the output.

### 2-2 High Level APIs

We also provides some APIs to even accelerate productions. See [API page](api/index.md) for further details/

## 4. Models

The following list is the current availability of models with weight variations.  
Click the link at the model column for futher documentation.

|Task| Model| Weights|
|:----|:----|:----|
| Mono Depth Prediction | [MiDAS](models/mono_depth.md) | mono_depth_small<br>mono_depth_large |
| Salient Object Detection | [U2Net](models/salient_det.md) | mobile<br>basic<br>human_seg<br>portrait|
| Super Resolution | [RealESRGAN](models/super_resolution.md) | super_resolution4864<br>super_resolution6464 |
| Object Detection | [YOLO2/YOLO](models/obj_det.md) | Please refer to docs for details |
| Emotion Prediction | [FerPlus](models/emotion.md) | ferplus8 |
| Human Parsing | [HumanParsing](models/human_parsing.md) |human_attribute |
| Denoise | [HINet](models/denoising.md) | denoise_320_480 |
| Face Detection | [YuNet](models/face_det.md) | yunet_120_160 |
| Style Transfer | [AnimeGAN](models/style_transfer.md) | animeGAN512 |
| Image Classification | [ResNetV2](models/image_clf.md) | resnet18v2<br>resnet50v2<br>resnet101v2<br>resnet152v2 |
| Human Portrait Segmentation | [MODNet](models/human_seg.md) | modnet_256 |
| Semantic Segmentation | [FastSCNN](models/semantic_seg.md) | fast_scnn384<br>fast_scnn7681344 |
| Diver's View Segmentation | [SUIMNet](models/suim_net.md) | suim_net_3248<br>suim_rsb_72128<br>suim_vgg_25632<br>suim_vgg_72128 |

## 5. Deployment

It is extremely easy to use BiWAKO at application layer.

### 1. Real Time Prediction

Any model can be used in the same way to run real-time inference.

<img src="docs/img/live_demo.png" width="450">

### 2. FastAPI Implementation

Like the above example, you can build simple Backend API for inference on web server.
We have prepared sample deployment of the library with FastAPI.[Read this for details](demo/index.md).

<img src="docs/img/fastapi_demo.png" width="450">

### 3. Video Prediction

We also provides pre-defined video prediction API. [Read this for details](api/video_predictor.md)

<img src="docs/img/video_demo.png" width="450">
