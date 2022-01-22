import sys

sys.path.append("../")


import BiWAKO

video = "path/to/your_video.mp4"

model = BiWAKO.U2Net("basic.onnx")  # type: ignore
video_pred = BiWAKO.api.VideoPredictor(model)

video_pred.run(video, "prediction.mp4")
