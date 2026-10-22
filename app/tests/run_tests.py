import sys, json, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import verify as V
CASES = [
 ("攒竹穴", "攒竹穴在眉头内侧，按压攒竹穴可以缓解头痛。", "攒", "cuan2", ["zan3"], None),
 ("银行", "他在银行工作，每天步行去上班。", "行", "hang2", ["xing2"], 1),
 ("步行", "他在银行工作，每天步行去上班。", "行", "xing2", ["hang2"], 2),
 ("银行行长-行1", "银行行长今天来视察工作。", "行", "hang2", ["xing2"], 1),
 ("银行行长-行2", "银行行长今天来视察工作。", "行", "hang2", ["xing2"], 2),
 ("银行行长-长", "银行行长今天来视察工作。", "长", "zhang3", ["chang2"], None),
 ("重庆", "我下周要去重庆出差。", "重", "chong2", ["zhong4"], None),
 ("重量", "这个箱子的重量是二十公斤。", "重", "zhong4", ["chong2"], None),
 ("长大", "孩子长大了，个子也长高了。", "长", "zhang3", ["chang2"], None),
 ("长度", "这条河的长度是五公里。", "长", "chang2", ["zhang3"], None),
 ("音乐", "他喜欢音乐，每天都很快乐。", "乐", "yue4", ["le4"], 1),
 ("快乐", "他喜欢音乐，每天都很快乐。", "乐", "le4", ["yue4"], 2),
 ("着急", "他着急地看着窗外。", "着", "zhao2", ["zhe5", "zhuo2"], 1),
 ("看着", "他着急地看着窗外。", "着", "zhe5", ["zhao2", "zhuo2"], 2),
 ("角色", "她在电影里扮演了一个重要的角色。", "角", "jue2", ["jiao3"], None),
 ("会计", "我姐姐是一名会计，在公司负责财务。", "会", "kuai4", ["hui4"], None),
]
device = sys.argv[1] if len(sys.argv) > 1 else "auto"
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
out = []
for name, text, ch, c, w, occ in CASES:
    if only and name not in only: continue
    t = time.time()
    rep = V.run_check(text, ch, c, w, occ, device=device, outdir=str(V.OUT / f"test-{device}-{name}"), log=lambda *a: None)
    rows = [{"name": name, "text": text, "char": ch, "correct": c, "wrong": w, "verdict": r["verdict"], "path_verdict": r["path_verdict"],
             "conf": r["confidence"], "agree": r["nbest_agreement"], "tally": r["nbest_tally"], "decoded_word": r["decoded_word"],
             "observed": r["observed"], "reason": r["reason"], "funasr": rep["funasr"].get("text"), "decoded": rep["decoded_test"],
             "calib": rep["calibration"]["text"], "calib_decoded": rep["calibration"]["decoded"], "expected": rep["expected"],
             "tts": rep["tts"], "timings": rep["timings"]} for r in rep["results"]]
    out += rows
    for r in rows:
        print(json.dumps(r, ensure_ascii=False), flush=True)
json.dump(out, open(V.OUT / f"tests-{device}.json", "w"), ensure_ascii=False, indent=1)
