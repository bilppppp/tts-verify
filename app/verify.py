#!/usr/bin/env python3
"""TTS polyphone pronunciation verifier (Kaldi phone-path + anchor calibration + N-best).
Method: https://quaily.com/goldengrape/p/using-kaldi-phoneme-recognition-validate-tts-pronunciation
"""
import argparse, json, os, sys, time, difflib, shutil
from collections import Counter, defaultdict
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import kaldi_decode as K
from pinyin_util import norm_syl, split_syl, word_pinyin, is_hanzi, is_monophone_char
from phonetable import PhoneTable
import anchors as A

OUT = APP.parent / "runs"


def hz(s):
    return "".join(c for c in s if is_hanzi(c))


def words_chunks(dec, wb):
    words = [w for w in dec["words"] if w not in ("<UNK>", "<unk>", "!SIL")]
    chunks = [[K.base_of(p) for p in c] for c in K.segment(dec["phones"], wb)]
    return words, chunks


def locate(orig_hz, words, target_idx):
    """Map index in original hanzi string -> (word_index, char_index_in_word) in decoded words."""
    dec_hz, owner = "", []
    for wi, w in enumerate(words):
        for ci, c in enumerate(w):
            if is_hanzi(c):
                dec_hz += c; owner.append((wi, ci))
    sm = difflib.SequenceMatcher(None, orig_hz, dec_hz, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if i1 <= target_idx < i2:
            if tag == "equal" or (tag == "replace" and (i2 - i1) == (j2 - j1)):
                return owner[j1 + (target_idx - i1)], tag
            return None, tag
    return None, "none"


def observe(table, orig_hz, dec, wb, tidx, readings):
    """Return dict with observed target chunk from one decoding path."""
    words, chunks = words_chunks(dec, wb)
    if len(words) != len(chunks):
        return {"ok": False, "why": f"词数({len(words)})与音素切分数({len(chunks)})不一致"}
    loc, tag = locate(orig_hz, words, tidx)
    if loc is None:
        return {"ok": False, "why": f"目标字在识别结果中对不上（{tag}）", "words": words}
    wi, ci = loc
    w = words[wi]
    syls = word_pinyin(w)
    if len(syls) != len(w):
        return {"ok": False, "why": f"词「{w}」拼音数与字数不符", "words": words}
    alt = {ci: list(dict.fromkeys(readings + [syls[ci]]))}
    seg = table.segment(syls, chunks[wi], alt=alt)
    if seg is None:
        return {"ok": False, "why": f"词「{w}」的 {len(chunks[wi])} 个音素无法切成 {len(syls)} 个音节", "words": words}
    return {"ok": True, "word": w, "char": w[ci], "char_match": tag == "equal", "word_phones": chunks[wi],
            "obs": list(seg[1][ci]), "words": words}


def judge(obs, exp):
    """exp: {reading: tuple|None}; readings[0] is correct. Returns (label, detail)."""
    obs = tuple(obs)
    readings = list(exp)
    correct = readings[0]
    exact = [r for r in readings if exp[r] is not None and tuple(exp[r]) == obs]
    if len(exact) == 1:
        return exact[0], "音素序列与该读音预期完全一致"
    if len(exact) > 1:
        return "undecidable", "多个候选读音的预期音素序列相同，本方法分不开"

    def comps(seq, r):
        ini = split_syl(r)[0]
        if ini:
            return {"声母": seq[:1], "韵母(含介母/声调)": seq[1:], "音素个数": len(seq)}
        return {"声母": (), "韵母(含介母/声调)": seq, "音素个数": len(seq)}

    votes = {}
    ec = exp.get(correct)
    for r in readings[1:]:
        er = exp.get(r)
        if ec is None or er is None:
            continue
        cc, cr = comps(tuple(ec), correct), comps(tuple(er), r)
        co_c, co_r = comps(obs, correct), comps(obs, r)
        diff = [k for k in cc if cc[k] != cr[k]]
        sup_c = [k for k in diff if co_c[k] == cc[k]]
        sup_r = [k for k in diff if co_r[k] == cr[k]]
        votes[r] = (sup_c, sup_r, diff)
    if not votes:
        return "undecidable", "缺少候选读音的预期音素（锚点/表中没有）"
    wrong_hits = [r for r, (sc, sr, d) in votes.items() if sr and not sc]
    # 'correct' only if EVERY distinguishing component matches the correct reading (otherwise 不判定)
    correct_all = all(sc == d and not sr for sc, sr, d in votes.values())
    if len(wrong_hits) == 1:
        sc, sr, d = votes[wrong_hits[0]]
        return wrong_hits[0], f"区别项 {sr} 对上错读、正确读音无法解释（部分匹配）"
    if correct_all:
        return correct, "所有区别项都支持正确读音（非区别项有偏差）"
    return "other", "与各候选读音都对不上或证据矛盾"


def final_verdict(verdict, tally, readings):
    """Combine best-path verdict with N-best distribution. Returns (label, confidence, agreement)."""
    mapped = sum(n for k, n in tally.items() if k != "unmapped")
    agree = tally.get(verdict, 0) / mapped if mapped else 0.0
    rival = max([tally.get(r, 0) for r in readings if r != verdict] + [0]) / mapped if mapped else 0.0
    conf = "high" if agree >= 0.8 else "medium" if agree >= 0.6 and rival < 0.3 else "low"
    if verdict in ("undecidable", "other") or conf == "low":
        return "不判定", conf, round(agree, 2)
    if verdict == readings[0]:
        return f"正确（读 {verdict}）", conf, round(agree, 2)
    return f"错误（读成 {verdict}）", conf, round(agree, 2)


def run_check(text, char, correct, wrongs, occurrence=None, wav=None, calib_text=None, calib_wav=None,
              device="auto", speaker="Vivian", outdir=None, nbest=10, log=print, synth_text=None):
    t0 = time.time()
    readings = [norm_syl(correct)] + [norm_syl(w) for w in wrongs]
    run_dir = Path(outdir or OUT / time.strftime("%Y%m%d-%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    orig_hz = hz(text)
    idxs = [i for i, c in enumerate(orig_hz) if c == char]
    if not idxs:
        raise SystemExit(f"句子里没有目标字「{char}」")
    if occurrence:
        idxs = [idxs[occurrence - 1]]
    timings = {}
    # calibration sentence
    cover, missing, chosen = {}, [], None
    if not calib_text:
        calib_text, chosen, cover, missing = A.build_calibration(readings, exclude_chars=char)
    log(f"[calib] {calib_text}")
    # synthesis
    tts_info = {}
    if wav is None or calib_wav is None:
        import tts_engine as T
        if wav is None:
            wav = str(run_dir / "test_raw.wav")
            tts_info["test"] = T.synth(synth_text or text, wav, device=device, speaker=speaker)
            log(f"[tts] test {tts_info['test']}")
        if calib_wav is None:
            calib_wav = str(run_dir / "calib_raw.wav")
            tts_info["calib"] = T.synth(calib_text, calib_wav, device=device, speaker=speaker)
            log(f"[tts] calib {tts_info['calib']}")
    timings["tts_s"] = round(sum(v["seconds"] for v in tts_info.values()), 2)
    # decode
    t = time.time()
    dec = K.decode({"test": wav, "cal": calib_wav}, run_dir, nbest=nbest)
    timings["kaldi_s"] = round(time.time() - t, 2)
    wb = K.load_word_boundary()
    table = PhoneTable.load()
    # calibration: decoded anchor words -> component bases
    cwords, cchunks = words_chunks(dec["cal"], wb)
    anchor_obs = defaultdict(Counter)
    anchor_src = defaultdict(list)
    if len(cwords) == len(cchunks):
        for w, ch in zip(cwords, cchunks):
            syls = word_pinyin(w)
            if len(syls) != len(w) or not all(is_hanzi(c) for c in w):
                continue
            # only trust planned anchor words (or, for a user-given calib text, words made of single-reading chars):
            # a mis-recognised polyphone like 毫米→"着 你" would otherwise poison the anchors
            if (chosen is not None and w not in chosen) or (chosen is None and not all(is_monophone_char(c) for c in w)):
                continue
            seg = table.segment(syls, ch)
            if not seg:
                continue
            table.add(syls, ch, w)
            for s, c in zip(syls, seg[1]):
                anchor_obs["S:" + s][tuple(c)] += 1; anchor_src["S:" + s].append(f"{w}:{list(c)}")
                ini, fin = split_syl(s)
                if ini and len(c) >= 2:
                    anchor_obs["I:" + ini][c[0]] += 1; anchor_src["I:" + ini].append(f"{w}:{c[0]}")
                    anchor_obs["F:" + fin][tuple(c[1:])] += 1; anchor_src["F:" + fin].append(f"{w}:{list(c[1:])}")
                elif not ini:
                    anchor_obs["F:" + fin][tuple(c)] += 1; anchor_src["F:" + fin].append(f"{w}:{list(c)}")
        table.fit(); table.save()
    anchors = {k: v.most_common(1)[0][0] for k, v in anchor_obs.items()}
    labels = table.base_labels()
    exp, exp_src = {}, {}
    for r in readings:
        e, src = table.expected(r, anchors)
        ini, fin = split_syl(r)
        used = []
        if ini and ("I:" + ini) in anchors: used.append("I:" + ini)
        if ("F:" + fin) in anchors: used.append("F:" + fin)
        if ("S:" + r) in anchors: used = ["S:" + r]
        exp[r] = list(e) if e else None
        exp_src[r] = {"how": src, "calibrated": used}

    def lab(seq):
        return [f"{b}({labels.get(b, '?')})" for b in seq] if seq else None

    results = []
    for k, ti in enumerate(idxs):
        best = observe(table, orig_hz, dec["test"], wb, ti, readings)
        verdict, why = ("undecidable", best.get("why")) if not best["ok"] else judge(best["obs"], exp)
        tally = Counter(); nb_detail = []
        for h in dec["test"]["nbest"]:
            o = observe(table, orig_hz, h, wb, ti, readings)
            v = judge(o["obs"], exp)[0] if o["ok"] else "unmapped"
            tally[v] += 1
            nb_detail.append({"words": "".join(o.get("words", h["words"])), "target_word": o.get("word"), "obs": o.get("obs"), "verdict": v})
        final_label, conf, agree = final_verdict(verdict, tally, readings)
        results.append({
            "occurrence": k + 1 if not occurrence else occurrence, "char_index": ti,
            "verdict": final_label, "path_verdict": verdict, "reason": why, "confidence": conf,
            "nbest_agreement": agree, "nbest_tally": dict(tally),
            "decoded_word": best.get("word"), "decoded_char": best.get("char"),
            "observed": lab(best.get("obs")), "word_phones": best.get("word_phones"),
            "nbest": nb_detail})
    asr = funasr_check(wav, orig_hz, idxs, char)
    timings["total_s"] = round(time.time() - t0, 2)
    report = {
        "text": text, "synth_text": synth_text, "char": char, "readings": {"correct": readings[0], "wrong": readings[1:]},
        "calibration": {"text": calib_text, "decoded": " ".join(dec["cal"]["words"]), "cover": cover,
                        "missing": missing, "anchors": {k: (list(v) if isinstance(v, tuple) else v) for k, v in anchors.items()},
                        "anchor_sources": dict(anchor_src)},
        "decoded_test": " ".join(dec["test"]["words"]),
        "expected": {r: lab(exp[r]) for r in readings}, "expected_source": exp_src,
        "results": results, "funasr": asr, "tts": tts_info, "timings": timings, "run_dir": str(run_dir),
        "test_wav": wav, "calib_wav": calib_wav,
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


def funasr_check(wav, orig_hz, idxs, char, url="http://127.0.0.1:8765/asr/offline/transcribe"):
    """Second opinion from the existing FunASR service (text only; LM-biased, so weak evidence)."""
    try:
        import requests
        with open(wav, "rb") as f:
            r = requests.post(url, files={"file": f}, data={"language": "zh"}, timeout=120, proxies={"http": None, "https": None})
        txt = r.json().get("full_text", "")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    a = hz(txt)
    sm = difflib.SequenceMatcher(None, orig_hz, a, autojunk=False)
    got = []
    for ti in idxs:
        c = None
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if i1 <= ti < i2 and (tag == "equal" or (i2 - i1) == (j2 - j1)):
                c = a[j1 + ti - i1]
        ctx = a[max(0, (j1 + ti - i1) - 2): (j1 + ti - i1) + 3] if c else None
        got.append({"char": c, "context": ctx, "pinyin_of_context": word_pinyin(ctx) if ctx else None})
    return {"ok": True, "text": txt, "targets": got}


def render(rep):
    L = []
    L.append(f"句子: {rep['text']}   目标字: {rep['char']}   正确: {rep['readings']['correct']}   错读候选: {', '.join(rep['readings']['wrong'])}")
    L.append(f"识别(测试句): {rep['decoded_test']}")
    L.append(f"标定句: {rep['calibration']['text']}")
    L.append(f"识别(标定句): {rep['calibration']['decoded']}")
    for r, e in rep["expected"].items():
        L.append(f"  预期 {r:8s}: {e}   来源: {rep['expected_source'][r]}")
    for x in rep["results"]:
        L.append(f"— 第{x['occurrence']}处 → 【{x['verdict']}】 置信度 {x['confidence']} (N-best 一致率 {x['nbest_agreement']}, {x['nbest_tally']})")
        L.append(f"    识别为词「{x['decoded_word']}」中的「{x['decoded_char']}」，实测音素 {x['observed']}；理由: {x['reason']}")
    fa = rep.get("funasr") or {}
    if fa.get("ok"):
        L.append(f"FunASR 旁证: {fa['text']}  目标位置: {[(g['char'], g['context']) for g in fa['targets']]}")
    t = rep["timings"]
    L.append(f"耗时: TTS {t.get('tts_s')}s, Kaldi {t.get('kaldi_s')}s, 总计 {t.get('total_s')}s   结果目录: {rep['run_dir']}")
    return "\n".join(L)


def bootstrap(corpus, wavdir):
    """Learn phone table from corpus sentences + their wavs (b00.wav ...)."""
    lines = [l.strip() for l in open(corpus) if l.strip()]
    utts = {}
    for i, l in enumerate(lines):
        p = Path(wavdir) / f"b{i:02d}.wav"
        if p.exists(): utts[f"b{i:02d}"] = str(p)
    dec = K.decode(utts, OUT / "bootstrap", nbest=1)
    wb = K.load_word_boundary()
    table = PhoneTable.load()
    n = 0
    for u, d in dec.items():
        words, chunks = words_chunks(d, wb)
        if len(words) != len(chunks): continue
        for w, ch in zip(words, chunks):
            syls = word_pinyin(w)
            if len(syls) == len(w) and all(is_hanzi(c) for c in w):
                table.add(syls, ch, w); n += 1
    table.fit(); table.save()
    print(f"pairs added {n}, syllables known {len(table.syl)}")
    return table


def main():
    ap = argparse.ArgumentParser(description="用 Kaldi 音素路径校验 TTS 多音字读音")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("check")
    c.add_argument("--text", required=True); c.add_argument("--char", required=True)
    c.add_argument("--correct", required=True, help="正确读音, 如 cuan2 或 cuán")
    c.add_argument("--wrong", required=True, nargs="+", help="可能的错读, 可多个")
    c.add_argument("--occurrence", type=int, help="第几处（默认全部）")
    c.add_argument("--wav", help="已有测试音频（不合成）"); c.add_argument("--calib-wav"); c.add_argument("--calib-text")
    c.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    c.add_argument("--synth-text", help="送给 TTS 的文本（如用同音字替换做对照），分析仍按 --text");
    c.add_argument("--speaker", default="Vivian"); c.add_argument("--out"); c.add_argument("--json", action="store_true")
    b = sub.add_parser("bootstrap"); b.add_argument("--corpus", default=str(APP / "data/bootstrap_corpus.txt")); b.add_argument("--wavdir", default=str(APP / "data/boot_wav"))
    sub.add_parser("calib").add_argument("readings", nargs="+")
    a = ap.parse_args()
    if a.cmd == "check":
        rep = run_check(a.text, a.char, a.correct, a.wrong, a.occurrence, a.wav, a.calib_text, a.calib_wav, a.device, a.speaker, a.out,
                        log=(lambda *x: print(*x, file=sys.stderr)), synth_text=a.synth_text)
        print(json.dumps(rep, ensure_ascii=False, indent=1) if a.json else render(rep))
    elif a.cmd == "bootstrap":
        bootstrap(a.corpus, a.wavdir)
    elif a.cmd == "calib":
        print(A.build_calibration([norm_syl(r) for r in a.readings]))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
