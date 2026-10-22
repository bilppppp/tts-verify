#!/bin/bash
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
cd ~/tts-verify
tts-venv/bin/modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local_dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice && echo QWEN_DONE
