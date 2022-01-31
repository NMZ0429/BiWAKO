# Object Detection

<figure markdown>
  ![Image title](img/yolo.jpg){ width="700" }
  <figcaption>Query image and prediction</figcaption>
</figure>

Currently, there are two YOLOv5 implementations avaialables for object detection task. Old implementation will be removed in future version.

Those models contains a large number of pretrained weights. Please refer to the following table for available options.  
The first letter `nano, s, ...` is the scaling and suffix `simp` means the model is simplified by onnx simplifier. With the number `6`, the model is updated to the upstream version of YOLOv5.

| Model| Weights| Model | Weights |
|:----|:----|:----|:----|
| YOLO2 | yolo_nano<br>yolo_nano6<br>yolo_nano_simp<br>yolo_nano6_simp<br>yolo_s<br>yolo_s6<br>yolo_s_simp<br>yolo_s6_simp<br>yolo_m<br>yolo_m6<br>yolo_m_simp<br>yolo_m6_simp<br>yolo_l<br>yolo_l_simp<br>yolo_x<br>yolo_x6<br>yolo_x_simp<br>yolo_x6_simp | YOLO | yolo_nano<br>yolo_s<br>yolo_xl<br>yolo_extreme<br>yolo_nano_smp<br>yolo_s_smp<br>yolo_xl_smp<br>yolo_extreme_smp |

---

## `BiWAKO.YOLO2`

!!! note
    It is recommended to use this model rather than previous YOLO. This model optimizes pre/post-processing operations with new ort opsets. Runtime is 3~4 times faster than the previous model. If you want to use the raw output of the YOLO or customize post-processing with your choice of parameters, use the previous model below.

::: BiWAKO.YOLO2
    handler: python
    selection:
        members:
            - __init__
            - predict
            - render
    rendering:
        show_root_heading: false
        show_source: false

---

## `BiWAKO.YOLO`

::: BiWAKO.YOLO
    handler: python
    selection:
        members:
            - __init__
            - predict
            - render
    rendering:
        show_root_heading: false
        show_source: false
