"""Controls: feed TTS a homophone so we KNOW what it says, then check the app's verdict (analysis uses original text)."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import verify as V
C = [  # name, analysis text, synth text, char, correct, wrong, occ, expected-truth
 ("ctl-银航(hang2)", "他在银行工作。", "他在银航工作。", "行", "hang2", ["xing2"], None, "hang2"),
 ("ctl-银星(xing2)", "他在银行工作。", "他在银星工作。", "行", "hang2", ["xing2"], None, "xing2"),
 ("ctl-虫庆(chong2)", "我下周要去重庆出差。", "我下周要去虫庆出差。", "重", "chong2", ["zhong4"], None, "chong2"),
 ("ctl-众庆(zhong4)", "我下周要去重庆出差。", "我下周要去众庆出差。", "重", "chong2", ["zhong4"], None, "zhong4"),
 ("ctl-掌大(zhang3)", "孩子长大了。", "孩子掌大了。", "长", "zhang3", ["chang2"], None, "zhang3"),
 ("ctl-常大(chang2)", "孩子长大了。", "孩子常大了。", "长", "zhang3", ["chang2"], None, "chang2"),
 ("ctl-音月(yue4)", "他喜欢音乐。", "他喜欢音月。", "乐", "yue4", ["le4"], None, "yue4"),
 ("ctl-音勒(le4)", "他喜欢音乐。", "他喜欢音勒。", "乐", "yue4", ["le4"], None, "le4"),
 ("ctl-窜竹(cuan4)", "按压攒竹穴可以缓解头痛。", "按压窜竹穴可以缓解头痛。", "攒", "cuan2", ["zan3"], None, "cuan4(声调不同)"),
 ("ctl-着急→招急(zhao1)", "他着急地看着窗外。", "他招急地看着窗外。", "着", "zhao2", ["zhe5", "zhuo2"], 1, "zhao1(声调不同)"),
]
device = sys.argv[1] if len(sys.argv) > 1 else "auto"
out = []
for name, text, st, ch, c, w, occ, truth in C:
    rep = V.run_check(text, ch, c, w, occ, device=device, outdir=str(V.OUT / f"ctl-{name}"), log=lambda *a: None, synth_text=st)
    for r in rep["results"]:
        row = {"name": name, "synth_text": st, "truth": truth, "verdict": r["verdict"], "path_verdict": r["path_verdict"], "conf": r["confidence"],
               "agree": r["nbest_agreement"], "tally": r["nbest_tally"], "decoded_word": r["decoded_word"], "observed": r["observed"],
               "reason": r["reason"], "decoded": rep["decoded_test"], "funasr": rep["funasr"].get("text"), "timings": rep["timings"]}
        out.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
json.dump(out, open(V.OUT / f"controls-{device}.json", "w"), ensure_ascii=False, indent=1)
