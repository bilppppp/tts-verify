#!/bin/bash
set -ex
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY all_proxy ALL_PROXY
cd ~/tts-verify
export UV_HTTP_TIMEOUT=300
UVI="$HOME/.local/bin/uv pip install --python tts-venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple"
$UVI torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --find-links https://mirrors.aliyun.com/pytorch-wheels/cu118/ --index-strategy unsafe-best-match
$UVI qwen-tts pypinyin soundfile fastapi uvicorn python-multipart huggingface_hub modelscope
tts-venv/bin/python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_arch_list());x=torch.randn(512,512,device='cuda');print((x@x).sum().item(), torch.cuda.get_device_name(0))"
echo TTS_DONE
