#!/bin/bash
filen=$1
filename="${filen%%.*}-conv"
ffmpeg -i $1 -c:v libx264 -crf 18 -preset slow -c:a copy $filename.mp4