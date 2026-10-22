"""Calibration-sentence builder: pick unambiguous common words that cover the components of each reading."""
import json, hashlib
from collections import defaultdict
from pathlib import Path
from pinyin_util import split_syl, norm_syl, is_monophone_char, is_gb2312_level1, is_hanzi
import kaldi_decode as K

POOL = Path(__file__).parent / "data/anchor_pool.json"
# hand-picked fallbacks known to decode cleanly (from the article + bootstrap corpus)
PREFERRED = ["全体", "村民", "咱们", "侧身", "勇敢", "泉水", "团结", "游泳", "学习", "工作", "城市", "孩子",
             "朋友", "花朵", "天气", "苹果", "翻译", "帮助", "老师", "医生", "动物", "故事", "历史", "飞机"]


def _vocab():
    v = set()
    for line in open(K.M / "graph/words.txt", encoding="utf-8"):
        v.add(line.split()[0])
    return v


def _jieba_freq():
    import jieba, os
    f = os.path.join(os.path.dirname(jieba.__file__), "dict.txt")
    d = {}
    for line in open(f, encoding="utf-8"):
        p = line.split()
        if len(p) >= 2: d[p[0]] = int(p[1])
    return d


def build_pool():
    vocab = _vocab()
    freq = _jieba_freq()
    pool = defaultdict(list)
    words = set(PREFERRED)
    for w, f in freq.items():
        if len(w) != 2 or w not in vocab or f < 50:
            continue
        if not all(is_hanzi(c) and is_gb2312_level1(c) for c in w):
            continue
        if all(is_monophone_char(c) for c in w):
            words.add(w)
        elif is_monophone_char(w[0]) and w[1] in "子们的了么么":  # neutral-tone anchors (孩子/我们/好了...)
            words.add(w)
    for w in sorted(words, key=lambda w: -freq.get(w, 0)):
        from pinyin_util import word_pinyin
        syls = word_pinyin(w)
        if len(syls) != len(w):
            continue
        for i, s in enumerate(syls):
            ini, fin = split_syl(s)
            if ini: pool["I:" + ini].append([w, i])
            pool["F:" + fin].append([w, i])
            pool["S:" + s].append([w, i])
    POOL.write_text(json.dumps(pool, ensure_ascii=False))
    return pool


def load_pool():
    if POOL.exists():
        return json.loads(POOL.read_text())
    return build_pool()


_FREQ = None


def _rank(w, key):
    global _FREQ
    if _FREQ is None: _FREQ = _jieba_freq()
    pref = 0 if w in PREFERRED else 1
    return (pref, -_FREQ.get(w, 0), w)


def components(readings):
    keys = []
    for r in readings:
        ini, fin = split_syl(r)
        for k in (["I:" + ini] if ini else []) + ["F:" + fin, "S:" + norm_syl(r)]:
            if k not in keys: keys.append(k)
    return keys


def build_calibration(readings, exclude_chars="", per_key=2, max_words=16):
    pool = load_pool()
    chosen, cover = [], defaultdict(list)
    keys = components(readings)
    for k in keys:
        cands = [w for w, i in pool.get(k, []) if not any(c in w for c in exclude_chars)]
        cands = sorted(set(cands), key=lambda w: _rank(w, k))
        n = per_key if not k.startswith("S:") else 1
        # reuse already chosen words first
        have = [w for w in chosen if w in cands]
        for w in have[:n]: cover[k].append(w)
        for w in cands:
            if len(cover[k]) >= n or len(chosen) >= max_words: break
            if w not in cover[k]:
                cover[k].append(w)
                if w not in chosen: chosen.append(w)
    text = "，".join(chosen) + "。"
    missing = [k for k in keys if not cover[k] and not k.startswith("S:")]
    return text, chosen, dict(cover), missing
