<div align="center">

<img src="utils/biwako.png" width="450">

### The home of moduled onnx models

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/NaMAZU)](https://pypi.org/project/NaMAZU/)
[![PyPI version](https://badge.fury.io/py/NaMAZU.svg)](https://badge.fury.io/py/NaMAZU)
[![PyPI Status](https://pepy.tech/badge/NaMAZU)](https://pepy.tech/project/NaMAZU)
[![license](https://img.shields.io/badge/License-GPL--3.0-informational)](https://github.com/NMZ0429/NaMAZU/blob/main/LICENSE)
![onnx](https://img.shields.io/badge/ONNX-1.10-005CED.svg?logo=ONNX&style=popout)

* * *

</div>

# BiWAKO

## Usage

No matter which model you use, these interface is the same.

```python
import BiWAKO

# 1. Initialize Model
model = BiWAKO.MiDASInference(model_type="small")

# 2. Feed Image (accept cv2 image or path to the image)
prediction = model.predict(image_or_image_path)

# 3. Show result as a cv2 image
result_img = model.render(prediction, image_or_image_path)
```

<img src="utils/biwako_api.png" width="800">

## Models

|Task| Model| Weights|
|:----|:----|:----|
| Mono Depth Prediction | MiDAS | Large-Small |
| Salient Object Detection | U2Net | Basic-Mobile-Human |
| Super Resolution | RealESRGAN | Large-Small |
| Object Detection | YOLOv5 | nano-s-large-extreme |
| Emotion Prediction | FerPlus | ferplus8 |
| Human Parsing | HumanParsing |human_attribute |
| Denoise | HINet | denoise_320_480 |
| Face Detection | YuNet | yunet_120_160 |
| Style Transfer | GAN | animeGAN512 |
