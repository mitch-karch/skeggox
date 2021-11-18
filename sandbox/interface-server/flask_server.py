# Skeggox Interface Server
from uuid import uuid4
from flask import Flask, render_template, Response

from opencv_camera import Camera

frame = None

app = Flask(__name__)


@app.route("/")
def index():
    # rendering webpage
    return render_template("index.html", commands=AVAILABLE_COMMANDS)


def gen(camera):
    global frame
    while True:
        # get camera frame
        frame = camera.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen(Camera()), mimetype="multipart/x-mixed-replace; boundary=frame")


ONE, THREE, FIVE, LC, RC, ZERO, EMPTY = "1", "3", "5", "LC", "RC", "0", "E"
AVAILABLE_COMMANDS = {
    "1": ONE,
    "3": THREE,
    "5": FIVE,
    "L Clutch": LC,
    "R Clutch": RC,
    "0": ZERO,
    "Empty": EMPTY,
}


@app.route("/<cmd>")
def command(cmd=None):
    global frame

    file_string = f"static/caps/{uuid4()}_{cmd}.jpg"
    with open(file_string, "wb") as f:
        f.write(frame)

    response = file_string
    return response, 200, {"Content-Type": "text/plain"}


if __name__ == "__main__":
    # defining server ip address and port
    app.run(host="0.0.0.0", port="8000", debug=True, threaded=True)