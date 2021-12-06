#!/bin/bash
gst-launch-1.0 filesrc -v location=$1 ! qtdemux ! video/x-h264 ! rtph264pay config-interval=10 pt=96 ! udpsink host=192.168.1.150 port=9000