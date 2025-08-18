#!/bin/bash
# start.sh

export TOKEN_SAPIEN_STR="your_token_here"
pip install flask sapien pillow numpy opencv-python

if [ "$1" = "prod" ]; then
    pip install gunicorn
    gunicorn -w 2 -b 0.0.0.0:5000 --timeout 300 anno_tool:app
else
    python anno_tool.py
fi