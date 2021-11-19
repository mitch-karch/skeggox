import cv2
from turbojpeg import TurboJPEG
import time
from base_camera import BaseCamera


class Camera(BaseCamera):
    def __init__(self):
        super(Camera, self).__init__()

    @staticmethod
    def frames():
        jpeg = TurboJPEG()

        camera = cv2.VideoCapture(
            'udpsrc port=9000 caps="application/x-rtp, media=(string)video, payload=(int)96, clock-rate=(int)90000, encoding-name=(string)H264"'
            "! rtph264depay"
            "! video/x-h264,width=1024,height=768,framerate=25/1"
            "! h264parse"
            "! nvv4l2decoder"
            "! nvvidconv"
            "! video/x-raw, format=(string)I420"
            "! appsink ",
        )

        while True:
            _, img = camera.read()
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)
            # yield cv2.imencode(".jpg", img)[1].tobytes()
            yield jpeg.encode(img)
