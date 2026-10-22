#!/bin/bash
cd ~/tts-verify/app; unset ALL_PROXY all_proxy
../tts-venv/bin/python - <<'PY'
import tts_engine as T, json
out={}
for dev in ["cuda","cpu"]:
    r=T.synth("他在银行工作，每天步行去上班。", f"/tmp/timing_{dev}.wav", device=dev); out[dev]=r; print(dev, r, flush=True)
json.dump(out, open("../runs/timing.json","w"))
PY
