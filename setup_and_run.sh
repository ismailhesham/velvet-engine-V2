#!/usr/bin/env bash
set -e

echo "=========================================="
echo "  KINETIC ESSAY STUDIO - SETUP & LAUNCH"
echo "=========================================="

apt-get update -y && apt-get install -y ffmpeg espeak-ng fonts-dejavu-core fonts-liberation wget curl

pip install --upgrade pip
pip install gradio==4.44.0 requests pillow numpy soundfile kokoro>=0.3.4 whisperx transformers torch torchvision

mkdir -p assets/fonts workspace

wget -q -O assets/fonts/Inter-Regular.ttf "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf" || true
wget -q -O assets/fonts/Inter-Bold.ttf "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf" || true

python3 app.py