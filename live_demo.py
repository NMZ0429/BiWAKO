from warnings import resetwarnings
import cv2

from BiWAKO import YOLO, HumanParsing, MiDAS, U2Net, YuNet, ResNet


# Uncomment the model you want to use and comment out the other models

# model = MiDASInference("mono_depth_small")
# model = YOLO("yolo_nano")
model = U2Net("mobile")
# model = HumanParsing("human_attribute")
# model = YuNet()
# model = ResNet("resnet18v2")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Live Prediction", model.render(model.predict(frame), frame))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
