from BiWAKO import MiDASInference, YOLO, U2NetInference, HumanParsing, YuNet
import cv2

cap = cv2.VideoCapture(0)

# setup
# model = MiDASInference("mono_depth_small")
# model = YOLO("yolo_nano")
# model = U2NetInference("mobile")
# model = HumanParsing("human_attribute")
model = YuNet()

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow("frame", model.render(model.predict(frame), frame, score_th=0.3))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
