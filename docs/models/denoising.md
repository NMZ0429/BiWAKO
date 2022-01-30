# HINet

!!! warning
    This is very large model and the weight file takes approximately 400MB on the disk.

<figure markdown>
  ![Image title](img/hinet.png){ width="700" }
  <figcaption>Original image and denoised image</figcaption>
</figure>

## `BiWAKO.HINet`

::: BiWAKO.HINet
    handler: python
    selection:
        members:
            - __init__
            - predict
            - render
    rendering:
        show_root_heading: false
        show_source: false
