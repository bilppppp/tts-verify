# TTS 多音字读音校验（Kaldi 音素路径法）

方法来自 goldengrape《用 Kaldi 音素识别验证 TTS 发音》。流程：先用 TTS 合成测试句和一句自动生成的标定句，再用 Kaldi（vosk-model-cn-0.22）解码，读出最优路径上的音素编号（base=(id-11)//4）。用标定句里的无歧义词标定声母和韵母的编号，对比目标字的实测音素，再用 N-best（10 条）交叉验证。最后给出三种结论之一：正确、错误（读成了什么）、不判定。

## 组成
- `~/tts-verify/miniforge3/envs/kaldi`：Kaldi 5.5.1172 CPU 版（conda-forge）
- `~/tts-verify/vosk-model-cn-0.22`：中文大模型
- `~/tts-verify/models/Qwen3-TTS-12Hz-0.6B-CustomVoice`：TTS 模型。GTX 970M 上用 bf16 跑，显存不够时自动分段合成，再不行就退回 CPU fp32
- `~/tts-verify/tts-venv`：Python 3.12 环境，含 torch 2.7.1+cu118（支持 sm_50）、qwen-tts、pypinyin、jieba、fastapi
- `app/kaldi_decode.py`：Kaldi 解码，输出最优路径的词和音素，以及 N-best
- `app/phonetable.py`：模型包里没有音素符号表，这里从解码数据学出"音节 → base 序列"表，存在 `data/phone_table.json`
- `app/anchors.py`：自动生成标定句。选词条件：常用（jieba 词频）、所有字都只有一个读音、在 vosk 词表里，并且覆盖正确读音和错读各自的声母、韵母
- `app/verify.py`：命令行入口和判定逻辑；`app/web.py`：网页界面
- `app/tests/run_tests.py`：多音字测试；`app/tests/run_controls.py`：同音字对照测试

## 命令行
```
~/tts-verify/bin/tts-verify check --text "攒竹穴在眉头内侧，按压攒竹穴可以缓解头痛。" --char 攒 --correct cuan2 --wrong zan3
# 可选参数：
#   --occurrence 2        只看第 2 处
#   --wrong zhe5 zhuo2    可以给多个错读
#   --wav my.wav          校验已有音频（不重新合成）
#   --synth-text "..."    换一段文本送给 TTS（如同音字替换），分析仍按 --text
#   --device cpu|cuda|auto
#   --speaker Vivian|Serena|Uncle_Fu|Dylan|Eric
#   --json                输出 JSON
~/tts-verify/bin/tts-verify calib cuan2 zan3     # 只看会生成什么标定句
~/tts-verify/bin/tts-verify bootstrap            # 用 data/bootstrap_corpus.txt 重新学音素表
```
每次运行的结果都在 `~/tts-verify/runs/<时间>/`，包括 report.json、音频和 kaldi.log。

## 网页界面
http://<server-ip>:8770 （只在 Tailscale 内网可访问）
```
systemctl --user status|start|stop|restart tts-verify-web
journalctl --user -u tts-verify-web -f
```
模型在第一次请求时才加载到 GPU，约占 2.3 GB 显存。服务在跑的时候，命令行用 `--device auto` 发现显存不够会自动改用 CPU。

## 判定规则
1. 实测 base 序列和某个候选读音的预期完全一致，就判为这个读音。
2. 如果只是部分匹配：区别项全部对上正确读音，判正确；有区别项对上错读、而正确读音解释不了，判错误；其余情况一律不判定。
3. 多个候选读音预期相同（例如 穴/血 走的是同一条音素路径），判不判定。
4. 最优路径的结论还要看 N-best（10 条）：同结论占比 ≥0.8 记 high；占比 ≥0.6 且竞争读音占比 <0.3 记 medium；其余改成不判定。（实测：TTS 读成"众庆"时，最优路径被语言模型拉回"重庆"，但 N-best 是 5:5，所以这条规则必须有。）
5. FunASR（:8765）的转写只作为旁证列出。它受语言模型影响，不参与判定。

## 局限
- Kaldi 的音素来自所识别出那个词的词典读音。词典里没有的读音（比如"攒竹"读 cuán），只能通过别的词或字的音素体现出来，最优路径也会被语言模型拉向常见词。
- 声调只能通过不同的韵母编号间接区分。
- 音素表是自动学出来的，pypinyin 和 ASR 词典读音不一致的地方可能带噪声，所以标定句的锚点优先。
- 标签（如 "156(zh)"）只是辅助阅读，判定比较的是编号本身。
