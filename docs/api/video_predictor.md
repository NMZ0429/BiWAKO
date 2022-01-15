# Video Predictor

Easy way to run models on video inputs and export visualized results as videos.

## Usage

```python
from BiWAKO import MODNet
from BiWAKO.api import VideoPredictor


model = MODNet()
video_predictor = VideoPredictor(model)
video_predictor.run("some/video.mp4", title="modnet_prediction.mp4")
```

## `BiWAKO.api.VideoPredictor`

::: BiWAKO.api.VideoPredictor
    handler: python
    selection:
        members:
            - __init__
            - run
    rendering:
        show_root_heading: false
        show_source: false
