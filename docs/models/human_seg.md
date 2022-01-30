# Human Portrait Segmentation

<figure markdown>
  ![Image title](img/modnet.jpg){ width="700" }
  <figcaption>Query image, predicted segmentation map, and visualization</figcaption>
</figure>

## `BiWAKO.MODNet`

::: BiWAKO.MODNet
    handler: python
    selection:
        members:
            - __init__
            - predict
            - render
            - _preprocess
    rendering:
        show_root_heading: false
        show_source: false

## Reference

[https://github.com/ZHKKKe/MODNet/blob/master/onnx/inference_onnx.py](https://github.com/ZHKKKe/MODNet/blob/master/onnx/inference_onnx.py)