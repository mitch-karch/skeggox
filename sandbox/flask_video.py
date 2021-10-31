# Test Script
import flask
import cv2
from uuid import uuid4
# raspivid -n -w 1024 -h 768 -t 0 -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=10 pt=96 ! udpsink host=192.168.0.100 port=9000

class VideoCamera(object):

    def __init__(self):
        #capturing video
        self.ds_factor = 1
        self.video = cv2.VideoCapture(
            'udpsrc port=9000 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264"'
            '! rtph264depay'
            '! video/x-h264,width=1024,height=768,framerate=30/1'
            '! h264parse'
            '! avdec_h264'
            '! videoconvert'
            '! appsink ',
            cv2.CAP_GSTREAMER,
        )

    def __del__(self):
    #releasing camera
        self.video.release()

    def get_frame(self):
    #extracting frames
        ret, frame = self.video.read()
        frame=cv2.resize(frame,None,fx=self.ds_factor,fy=self.ds_factor,interpolation=cv2.INTER_AREA)                    

        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
