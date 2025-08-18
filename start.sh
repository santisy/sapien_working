#!/bin/bash
# start.sh

if [ "$1" = "prod" ]; then
    pip install gunicorn
    gunicorn -w 2 -b 0.0.0.0:5000 --timeout 300 anno_tool:app
else
    python anno_tool.py
fi