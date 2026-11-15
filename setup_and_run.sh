#!/usr/bin/env bash
set -e

apt-get update -y && apt-get install -y ffmpeg espeak-ng fonts-dejavu-core wget curl
pip install --upgrade pip
pip install flask requests pillow numpy soundfile kokoro>=0.3.4 whisperx transformers torch torchvision

mkdir -p assets/fonts workspace
wget -q -O assets/fonts/Inter-Bold.ttf "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf" || true

python3 app.py
