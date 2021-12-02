# Test Script
import uuid
import cv2
from uuid import uuid4
# raspivid -n -w 1024 -h 768 -t 0 -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=10 pt=96 ! udpsink host=192.168.0.100 port=9000

cap = cv2.VideoCapture(
    'udpsrc port=9000 caps="application/x-rtp, media=(string)video, payload=(int)96, clock-rate=(int)90000, encoding-name=(string)H264"'
    "! rtph264depay"
    "! video/x-h264,width=1024,height=768,framerate=25/1"
    "! h264parse"
    "! avdec_h264"
    #"! nvv4l2decoder"
    #"! nvvidconv"
    "! videoconvert"
    #"! video/x-raw, format=(string)I420"
    "! appsink ",
    cv2.CAP_GSTREAMER
)


writer = cv2.VideoWriter('output.mp4', 0x7634706d, 20.0, (1024,768))

while True:

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            writer.write(frame)
        else:
            print("empty frame")
            continue

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cap.receive()
