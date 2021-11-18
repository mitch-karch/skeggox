import cv2
import time
from base_camera import BaseCamera

class Camera(BaseCamera):
    def __init__(self):
        super(Camera, self).__init__()

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(
            'udpsrc port=9000 caps="application/x-rtp, media=(string)video, payload=(int)96, clock-rate=(int)90000, encoding-name=(string)H264"'
            "! rtpjitterbuffer"
            "! rtph264depay"
            "! video/x-h264,width=1024,height=768,framerate=30/1"
            "! h264parse"
            "! nvv4l2decoder"
            "! nvvidconv"
            "! video/x-raw, format=BGRx "
            "! videoconvert"
            "! video/x-raw, format=BGR"
            "! appsink ",
            cv2.CAP_GSTREAMER,
        )

        while True:
            _,img = camera.read()
            yield cv2.imencode('.jpg',img)[1].tobytes()
