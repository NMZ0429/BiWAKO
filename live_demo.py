from BiWAKO import MiDAS, YOLO, U2Net, HumanParsing, YuNet
import cv2

cap = cv2.VideoCapture(0)

# Uncomment the model you want to use and comment out the other models

# model = MiDASInference("mono_depth_small")
model = YOLO("yolo_nano")
# model = U2Net("mobile")
# model = HumanParsing("human_attribute")
# model = YuNet()

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow("frame", model.render(model.predict(frame), frame))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
