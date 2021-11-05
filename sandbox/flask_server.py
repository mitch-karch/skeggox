# main.py
# import the necessary packages
from uuid import uuid4
from flask import Flask, render_template, Response

from flask_video import VideoCamera

frame = None

app = Flask(__name__)
@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html' , commands=AVAILABLE_COMMANDS)

def gen(camera):
    global frame
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


ONE, THREE, FIVE, LC, RC, ZERO, EMPTY = "1", "3", "5", "LC", "RC", "0", "E"
AVAILABLE_COMMANDS = {
    'One': ONE,
    'Three': THREE,
    'Five': FIVE,
    'Left Clutch': LC,
    'Right Clutch': RC,
    'Zero': ZERO,
    'Empty': EMPTY
}

@app.route('/<cmd>')
def command(cmd=None):
    global frame

    file_string = f"caps/{uuid4()}_{cmd}.jpg"

    response = f"Capturing {file_string}"

    with open(file_string, 'wb') as f:
        f.write(frame)

    # ser.write(camera_command)
    return response, 200, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='8000', debug=True)
