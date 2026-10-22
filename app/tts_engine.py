"""TTS backends. Default: Qwen3-TTS-12Hz-0.6B-CustomVoice (GPU bf16 on GTX 970M, CPU fp32 fallback)."""
import os, re, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import soundfile as sf

QWEN_ID = os.environ.get("TTSV_QWEN", os.path.expanduser("~/tts-verify/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"))
_cache = {}


def pick_device(pref="auto"):
    import torch
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        try:
            free = torch.cuda.mem_get_info()[0]
            if pref == "auto" and free < float(os.environ.get("TTSV_MIN_VRAM", "2.0e9")):
                print(f"[tts] GPU free VRAM {free/1e9:.2f} GB too low (web service running?) -> CPU", flush=True)
                return "cpu"
            x = torch.randn(64, 64, device="cuda"); (x @ x).sum().item()
            return "cuda"
        except Exception:
            pass
    if pref == "cuda":
        raise RuntimeError("CUDA requested but not usable")
    return "cpu"


def load_qwen(device="auto"):
    import torch
    from qwen_tts import Qwen3TTSModel
    if device in ("auto", "cuda") and ("qwen", "cuda") in _cache:
        return _cache[("qwen", "cuda")]
    if device == "cpu" and ("qwen", "cpu") in _cache:
        return _cache[("qwen", "cpu")]
    dev = pick_device(device)
    key = ("qwen", dev)
    if key in _cache:
        return _cache[key]
    dtype = getattr(torch, os.environ.get("TTSV_GPU_DTYPE", "bfloat16")) if dev == "cuda" else torch.float32
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("TTSV_THREADS", "6")))
    t = time.time()
    m = Qwen3TTSModel.from_pretrained(QWEN_ID, device_map=("cuda:0" if dev == "cuda" else "cpu"),
                                      dtype=dtype, attn_implementation="sdpa")
    _cache[key] = (m, dev)
    print(f"[tts] loaded {QWEN_ID} on {dev} ({dtype}) in {time.time()-t:.1f}s", flush=True)
    return _cache[key]


def _gen(m, text, speaker, seed):
    import torch
    torch.manual_seed(seed)
    with torch.inference_mode():
        wavs, sr = m.generate_custom_voice(text=text, language="Chinese", speaker=speaker)
    return np.asarray(wavs[0]), sr


def _chunks(text, maxlen=14):
    parts = [p for p in re.split(r"(?<=[，。！？；、,.!?;])", text) if p.strip()]
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > maxlen:
            out.append(cur); cur = p
        else:
            cur += p
    if cur: out.append(cur)
    return out


def synth(text, out_wav, device="auto", speaker="Vivian", seed=1234):
    """Synthesize; on GTX 970M (3 GB) long inputs can OOM -> retry in punctuation chunks, then CPU."""
    import torch
    m, dev = load_qwen(device)
    t = time.time(); mode = "full"
    try:
        wav, sr = _gen(m, text, speaker, seed)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); mode = "chunked"
        try:
            pieces = []
            for c in _chunks(text):
                w, sr = _gen(m, c, speaker, seed); pieces += [w, np.zeros(int(0.25 * sr), dtype=w.dtype)]
                torch.cuda.empty_cache()
            wav = np.concatenate(pieces)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); mode = "cpu-fallback"
            m, dev = load_qwen("cpu")
            wav, sr = _gen(m, text, speaker, seed)
    if dev == "cuda":
        torch.cuda.empty_cache()
    el = time.time() - t
    sf.write(out_wav, wav, sr)
    dur = len(wav) / sr
    return {"device": dev, "mode": mode, "seconds": round(el, 2), "audio_sec": round(dur, 2), "rtf": round(el / max(dur, 1e-6), 2)}
