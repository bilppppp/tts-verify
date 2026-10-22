"""Small web UI for verify.py. Run: tts-venv/bin/python web.py --host <server-ip> --port 8770"""
import argparse, html, json, threading, time, shutil, sys
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import verify as V

app = FastAPI(title="TTS 多音字校验")
LOCK = threading.Lock()

FORM = """<!doctype html><html><head><meta charset="utf-8"><title>TTS 多音字校验</title>
<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 12px}
input,select{font-size:15px;padding:4px;margin:3px 0}label{display:block;margin-top:8px}
pre{background:#f5f5f5;padding:10px;white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}
.ok{color:#080;font-weight:bold}.bad{color:#c00;font-weight:bold}.und{color:#a60;font-weight:bold}</style></head><body>
<h2>TTS 多音字读音校验（Kaldi 音素路径 + 锚点标定 + N-best）</h2>
<form method="post" action="/check" enctype="multipart/form-data">
<label>测试句 <input name="text" size="70" value="{text}"></label>
<label>目标字 <input name="char" size="3" value="{char}"> 第几处(空=全部) <input name="occurrence" size="3" value=""></label>
<label>正确读音 <input name="correct" size="10" value="{correct}"> 可能错读(空格分隔) <input name="wrong" size="20" value="{wrong}"></label>
<label>音色 <select name="speaker">{speakers}</select> 设备 <select name="device"><option>auto</option><option>cuda</option><option>cpu</option></select></label>
<label>或上传测试音频（可选，不上传则用 Qwen3-TTS 合成） <input type="file" name="wav"></label>
<p><button type="submit">开始校验</button> （GPU 约 30–60 秒，CPU 约 2 分钟）</p></form>{result}</body></html>"""
SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"]


def page(result="", text="攒竹穴在眉头内侧，按压攒竹穴可以缓解头痛。", char="攒", correct="cuan2", wrong="zan3"):
    sp = "".join(f"<option>{s}</option>" for s in SPEAKERS)
    return FORM.replace("{text}", html.escape(text)).replace("{char}", html.escape(char)).replace("{correct}", html.escape(correct)) \
        .replace("{wrong}", html.escape(wrong)).replace("{speakers}", sp).replace("{result}", result)


def render_html(rep):
    rid = Path(rep["run_dir"]).name
    rows = ""
    for x in rep["results"]:
        cls = "ok" if x["verdict"].startswith("正确") else "bad" if x["verdict"].startswith("错误") else "und"
        rows += (f"<tr><td>{x['occurrence']}</td><td class='{cls}'>{html.escape(x['verdict'])}</td><td>{x['confidence']} ({x['nbest_agreement']})</td>"
                 f"<td>{html.escape(str(x['decoded_word']))}</td><td>{html.escape(str(x['observed']))}</td><td>{html.escape(str(x['reason']))}</td>"
                 f"<td>{html.escape(json.dumps(x['nbest_tally'], ensure_ascii=False))}</td></tr>")
    aud = ""
    for n in ("test_raw.wav", "calib_raw.wav", "test_16k.wav", "cal_16k.wav"):
        if (Path(rep["run_dir"]) / n).exists():
            aud += f"<div>{n}: <audio controls src='/runs/{rid}/{n}'></audio></div>"
    return (f"<h3>结果</h3><table><tr><th>处</th><th>结论</th><th>置信度(N-best一致率)</th><th>识别词</th><th>实测音素</th><th>理由</th><th>N-best 分布</th></tr>{rows}</table>"
            f"{aud}<pre>{html.escape(V.render(rep))}</pre><p><a href='/runs/{rid}/report.json'>report.json</a></p>")


@app.get("/", response_class=HTMLResponse)
def index():
    return page()


@app.post("/check", response_class=HTMLResponse)
def check(text: str = Form(...), char: str = Form(...), correct: str = Form(...), wrong: str = Form(...),
          occurrence: str = Form(""), speaker: str = Form("Vivian"), device: str = Form("auto"), wav: UploadFile = File(None)):
    run_dir = V.OUT / time.strftime("%Y%m%d-%H%M%S-web")
    run_dir.mkdir(parents=True, exist_ok=True)
    wavp = None
    if wav is not None and wav.filename:
        wavp = str(run_dir / ("upload_" + Path(wav.filename).name))
        with open(wavp, "wb") as f: shutil.copyfileobj(wav.file, f)
    try:
        with LOCK:
            rep = V.run_check(text, char.strip(), correct, wrong.split(), int(occurrence) if occurrence.strip() else None,
                              wav=wavp, device=device, speaker=speaker, outdir=run_dir)
        res = render_html(rep)
    except BaseException as e:
        res = f"<pre class='bad'>出错: {html.escape(repr(e))}</pre>"
    return page(res, text, char, correct, wrong)


@app.post("/api/check")
def api_check(payload: dict):
    with LOCK:
        rep = V.run_check(payload["text"], payload["char"], payload["correct"], payload["wrong"], payload.get("occurrence"),
                          device=payload.get("device", "auto"), speaker=payload.get("speaker", "Vivian"))
    return JSONResponse(rep)


@app.get("/runs/{rid}/{name}")
def runfile(rid: str, name: str):
    p = (V.OUT / rid / name).resolve()
    if not str(p).startswith(str(V.OUT.resolve())) or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--preload", action="store_true")
    a = ap.parse_args()
    if a.preload:
        import tts_engine as T; T.load_qwen("auto")
    uvicorn.run(app, host=a.host, port=a.port)
