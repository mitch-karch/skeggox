raspivid -n -w 1024 -h 768 -fps 25 -g 50 -t 0 -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=10 pt=96 ! udpsink host=192.168.1.157 port=9000
