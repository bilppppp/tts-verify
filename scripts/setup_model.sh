#!/bin/bash
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY all_proxy
cd ~/tts-verify
for i in $(seq 1 60); do
  curl -fSL -s --speed-limit 30000 --speed-time 20 -C - -o vosk-model-cn-0.22.zip https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip && break
  echo "retry $i $(stat -c %s vosk-model-cn-0.22.zip)"; sleep 2
done
unzip -q -o vosk-model-cn-0.22.zip && echo MODEL_DONE
