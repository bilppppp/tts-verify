# tts-verify：TTS 多音字读音校验

用 Kaldi 音素识别检查 TTS 有没有把多音字读错。比如"攒竹穴"的"攒"应读 cuán，TTS 常读成 zǎn。

方法来自 goldengrape 的文章[《用 Kaldi 音素识别验证 TTS 发音》](https://quaily.com/goldengrape/p/using-kaldi-phoneme-recognition-validate-tts-pronunciation)。本仓库把它做成了一个能直接用的命令行和网页工具，并接上了 TTS 合成。

## 能做什么

- **合成并校验**：给出文本、目标字、正确读音和可能的错读，工具会用 [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) 合成语音，再判断目标字实际读成了什么。
- **校验已有音频**：`--wav` 传入任意 TTS 生成的 wav，不重新合成。
- **同音字对照实验**：`--synth-text` 让 TTS 读另一段文本（例如把"银行"换成"银星"），但仍按原文分析，用来检验判定器本身准不准。
- **三种结论**：正确、错误（读成了什么）、不判定。分辨不了时不硬下结论。
- 网页界面（FastAPI），可以用 systemd 用户服务常驻。

## 原理

1. **不看识别出的汉字，看音素路径。** 语言模型会把读错的词"纠正"成常见词，所以直接读 Kaldi（vosk-model-cn-0.22）最优解码路径上的音素编号（base = (id-11)//4）。
2. **标定句当锚点。** 模型里没有音素符号表。工具自动挑一些常用、只有一个读音、并且覆盖正确读音和错读声母/韵母的词，拼成标定句，一起合成、解码，从中反推出编号对应的音（`app/anchors.py`）。另外用 `data/bootstrap_corpus.txt` 自动学了一张"音节 → base 序列"表（`app/data/phone_table.json`），作为辅助。
3. **N-best 交叉验证。** 再取 10 条候选路径，看目标位置的结论是否一致：同结论占比 ≥0.8 记 high；≥0.6 且竞争读音 <0.3 记 medium；其余改判"不判定"。
4. **判定规则。** 完全匹配某读音即判该读音；部分匹配时，区别项全对上正确读音判正确，对上错读而正确读音解释不了判错误，其余不判定；多个候选读音预期相同（如 穴/血）也不判定。
5. FunASR（可选，127.0.0.1:8765）的转写只作旁证，不参与判定。

细节见 [`app/README.md`](app/README.md)。

## 目录

```
app/            核心代码：kaldi_decode.py、phonetable.py、anchors.py、pinyin_util.py、tts_engine.py、verify.py（CLI）、web.py（网页）
app/tests/      run_tests.py（多音字用例）、run_controls.py（同音字对照）、rescore.py
app/data/       phone_table.json、bootstrap_corpus.txt、anchor_pool.json
bin/tts-verify  命令行入口
scripts/        安装和辅助脚本
deploy/         systemd 用户服务模板
```

## 安装

测试环境：Ubuntu 24.04 x86_64，i7-6700HQ，GTX 970M 3GB（驱动 535）。不需要 sudo，全部装在 `~/tts-verify`，约占 12G 磁盘。

代码默认安装目录是 `~/tts-verify`。可用环境变量 `TTSV_ROOT` 改 Kaldi/模型根目录、`TTSV_QWEN` 改 TTS 模型路径；`bin/tts-verify`、`scripts/*.sh`、`deploy/*.service` 里也写的是 `~/tts-verify`，换目录时要一起改。

```bash
git clone git@github.com:bilppppp/tts-verify.git ~/tts-verify-src
mkdir -p ~/tts-verify && cp -r ~/tts-verify-src/{app,bin} ~/tts-verify/ && cp ~/tts-verify-src/scripts/*.sh ~/tts-verify/
cd ~/tts-verify
mkdir -p logs runs out

# 1. Kaldi：Miniforge + conda-forge 的 kaldi=5.5.1172（CPU 版），env 名 kaldi
#    需要代理时：PROXY=http://127.0.0.1:7891 bash setup_kaldi.sh
bash setup_kaldi.sh

# 2. 中文识别模型 vosk-model-cn-0.22（约 1.3G，支持断点续传）
bash setup_model.sh

# 3. TTS Python 环境（先装 uv，再建 Python 3.12 venv）
uv venv --python 3.12 tts-venv
bash setup_tts.sh        # torch 2.7.1+cu118、qwen-tts、pypinyin、fastapi 等（脚本里用的是阿里云镜像，可自行换源）
tts-venv/bin/pip show jieba >/dev/null 2>&1 || ~/.local/bin/uv pip install --python tts-venv/bin/python jieba

# 4. 下载 TTS 模型（脚本用 modelscope；也可以用 huggingface-cli 从 HF 下载到同一路径）
bash dl_qwen.sh
```

关于老显卡：

- GTX 9xx 这类 Maxwell 卡（sm_50/52）新版 PyTorch 已不支持，所以用 **torch 2.7.1+cu118**。
- GPU 默认用 **bf16**（不是 fp16），可用环境变量 `TTSV_GPU_DTYPE` 修改。显存不够时自动分段合成，再不行退回 CPU fp32。
- 没有 GPU 也能跑，只是慢一些。Kaldi 解码本来就只用 CPU。

代理（如 `127.0.0.1:7891`）是可选的，只在下载时用。运行时脚本会清掉 `ALL_PROXY`，并把 `127.0.0.1,localhost` 加进 `NO_PROXY`。

## 命令行

```bash
~/tts-verify/bin/tts-verify check --text "攒竹穴在眉头内侧，按压攒竹穴可以缓解头痛。" --char 攒 --correct cuan2 --wrong zan3
# 可选：
#   --occurrence 2        只看第 2 处
#   --wrong zhe5 zhuo2    多个错读
#   --wav my.wav          校验已有音频
#   --synth-text "..."    换一段文本送给 TTS（对照实验），分析仍按 --text
#   --device auto|cuda|cpu
#   --speaker Vivian|Serena|Uncle_Fu|Dylan|Eric
#   --json
~/tts-verify/bin/tts-verify calib cuan2 zan3   # 只看会生成什么标定句
~/tts-verify/bin/tts-verify bootstrap          # 重新学音素表
```

每次运行的结果存在 `~/tts-verify/runs/<时间>/`，包括 report.json、音频和 kaldi.log。

## 网页界面

```bash
cp deploy/tts-verify-web.service ~/.config/systemd/user/
# 编辑文件，把 <server-ip> 换成监听地址（如 Tailscale IP 或 127.0.0.1）
systemctl --user daemon-reload
systemctl --user enable --now tts-verify-web
# 访问 http://<server-ip>:8770
```

模型在第一次请求时才加载，约占 2.1–2.3G 显存。网页服务占着显卡时，命令行的 `--device auto` 会自动改用 CPU。

## 实测结果（Qwen3-TTS 0.6B，GTX 970M）

16 个多音字用例：

| 用例 | 结果 |
|---|---|
| 攒竹穴（应读 cuán） | ❌ 读成 zǎn（high） |
| 银行、步行、银行行长（行×2、长）、重庆、重量 | ✅ |
| 长大、音乐、快乐、着急、看着、角色、会计 | ✅ |
| 长度 | ⚠️ 被识别成"程度"，不判定 |

也就是 1 个读错，14 个读对，1 个不判定。

同音字对照（故意让 TTS 读别的字，检验判定器）：声母或韵母不同的 6 例都抓到了，3 例老实给出"不判定"。但有 2 例只差声调的没识别出来：窜（cuàn）被判成 cuán，招（zhāo）被判成 zháo。

## 耗时

| 步骤 | 耗时 |
|---|---|
| TTS 合成 4 秒音频，GPU（970M，bf16） | 约 9 秒（RTF 2.35） |
| 同上，CPU（i7-6700HQ，fp32） | 约 31 秒（RTF 8.0） |
| Kaldi 解码 + N-best | 约 6 秒 |
| 一次完整校验（含标定句） | 35–56 秒 |

## 局限

- **声调不灵。** 声调只能靠不同韵母编号间接区分，只差声调的错读可能漏掉。可以考虑加基频（F0）分析补上。
- **受词典和语言模型影响。** Kaldi 的音素来自识别出那个词的词典读音。词典里没有的读音只能通过别的词体现，最优路径也会被拉向常见词，所以需要 N-best 和"不判定"。
- **只能事后校验。** 发现问题后，修正还要在 TTS 前端做拼音标注或用 SSML 指定读音。
- 音素表是自动学的，有噪声，所以判定优先用标定句锚点。
- 资源占用大：约 12G 磁盘，GPU 显存 2G 以上。

## 致谢

- 方法：goldengrape，[用 Kaldi 音素识别验证 TTS 发音](https://quaily.com/goldengrape/p/using-kaldi-phoneme-recognition-validate-tts-pronunciation)
- TTS 模型：[Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- 识别模型：[vosk-model-cn-0.22](https://alphacephei.com/vosk/models)；Kaldi 来自 conda-forge
