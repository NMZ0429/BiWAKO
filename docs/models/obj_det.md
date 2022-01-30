# Object Detection

There are two models available now.

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
