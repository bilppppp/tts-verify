import json, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verify import final_verdict
rows = []
for d in sorted(glob.glob(os.path.expanduser("~/tts-verify/runs/test-auto-*")) + sorted(glob.glob(os.path.expanduser("~/tts-verify/runs/ctl-*")))):
    p = os.path.join(d, "report.json")
    if not os.path.exists(p): continue
    r = json.load(open(p)); rd = [r["readings"]["correct"]] + r["readings"]["wrong"]
    for x in r["results"]:
        lab, conf, ag = final_verdict(x["path_verdict"], x["nbest_tally"], rd)
        t = r["timings"]; tt = r["tts"].get("test", {})
        print(" | ".join(map(str, [os.path.basename(d), r.get("synth_text") or r["text"], f"{r['char']}#{x['occurrence']}", "/".join(rd), x["decoded_word"],
              x["observed"], lab, conf, ag, x["nbest_tally"], (r.get("funasr") or {}).get("text"), tt.get("seconds"), tt.get("audio_sec"), t.get("kaldi_s"), t.get("total_s")])))
