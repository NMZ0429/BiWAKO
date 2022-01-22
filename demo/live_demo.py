import sys

sys.path.append("../")

import BiWAKO
import cv2

# Uncomment the model you want to use and comment out the other models

model = BiWAKO.MiDAS("weights/mono_depth_small")
# model = BiWAKO.U2Net("mobile")
# model = BiWAKO.HumanParsing("human_attribute")
# model = BiWAKO.YuNet()
# model = BiWAKO.ResNet("resnet18v2")
# model = BiWAKO.MODNet("modnet_256")
# model = BiWAKO.FastSCNN("weights/fast_scnn7681344")

# model = BiWAKO.YOLO2("yolo_s_simp")
# model = BiWAKO.YOLO("weights/yolo_s")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    pred = model.predict(frame)
    cv2.imshow("Live Prediction", model.render(pred, frame))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
