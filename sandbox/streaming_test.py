# Test Script
import uuid
import cv2
from uuid import uuid4
# raspivid -n -w 1024 -h 768 -t 0 -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=10 pt=96 ! udpsink host=192.168.0.100 port=9000
cap = cv2.VideoCapture(
    'udpsrc port=9000 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264"'
    '! rtph264depay'
    '! video/x-h264,width=1024,height=768,framerate=30/1'
    '! h264parse'
    '! avdec_h264'
    '! videoconvert'
    '! appsink ',
    cv2.CAP_GSTREAMER,
)


while True:

    ret, frame = cap.read()

    if ret is True:
        cv2.imshow("receive", frame)
    else:
        print("empty frame")
        continue

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if cv2.waitKey(1) & 0xFF == ord("1"):
        cv2.imwrite(f"{uuid4()}_1.jpg", frame)
cap.release()
cap.receive()
