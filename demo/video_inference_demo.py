import sys

sys.path.append("../")


import BiWAKO

video = "/Users/gen/Downloads/horse_2.mp4"

model = BiWAKO.U2Net("weights/basic.onnx")  # type: ignore
video_pred = BiWAKO.api.VideoPredictor(model)

video_pred.run(video, "horse_pred2.mp4")
