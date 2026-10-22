#!/bin/bash
set -ex
# 可选：如需代理下载，运行前设置 PROXY，例如 PROXY=http://127.0.0.1:7891
[ -n "${PROXY:-}" ] && export https_proxy=$PROXY http_proxy=$PROXY HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY
cd ~/tts-verify
[ -d miniforge3 ] || { curl -fsSL -o miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh; bash miniforge.sh -b -p ~/tts-verify/miniforge3; }
export PATH=~/tts-verify/miniforge3/bin:$PATH
conda create -y -n kaldi -c conda-forge "kaldi=5.5.1172=cpu*" python=3.11
echo KALDI_DONE
