"""Learn syllable -> base-phone-sequence table for this Kaldi model from decoded (word, phones) pairs.

Vosk-model-cn-0.22 ships no phone symbol table, only numeric ids. word_boundary.int shows
ids 1-10 = SIL/SPN (+ position variants), then groups of 4 (B/E/I/S): base=(id-11)//4.
A word's phone sequence comes from the lexicon, so a decoded word + its pinyin gives
(syllables, bases). Syllables have 1-3 phones (zero-initial / initial+final / initial+medial+final);
we resolve segmentation by hard-EM.
"""
import json, itertools
from collections import Counter, defaultdict
from pathlib import Path
from pinyin_util import split_syl

TABLE = Path(__file__).parent / "data/phone_table.json"


class PhoneTable:
    def __init__(self):
        self.syl = defaultdict(Counter)   # 'zan3' -> Counter{(155,16): n}
        self.pairs = []                   # raw (syllables, bases) observations

    # ---------- persistence ----------
    def save(self, path=TABLE):
        d = {"pairs": self.pairs}
        Path(path).write_text(json.dumps(d, ensure_ascii=False))

    @classmethod
    def load(cls, path=TABLE):
        t = cls()
        if Path(path).exists():
            d = json.loads(Path(path).read_text())
            t.pairs = [(list(x[0]), list(x[1]), x[2] if len(x) > 2 else "") for x in d.get("pairs", [])]
            t.fit()
        return t

    def add(self, syls, bases, word=None):
        if syls and bases and 1 <= len(bases) / len(syls) <= 3:
            self.pairs.append((list(syls), list(bases), word or ""))

    # ---------- component views ----------
    def initial_base(self):
        c = defaultdict(Counter)
        for s, cnt in self.syl.items():
            ini, fin = split_syl(s)
            for seq, n in cnt.items():
                if ini and len(seq) >= 2:
                    c[ini][seq[0]] += n
        return c

    def final_tail(self):
        c = defaultdict(Counter)
        for s, cnt in self.syl.items():
            ini, fin = split_syl(s)
            for seq, n in cnt.items():
                tail = seq[1:] if (ini and len(seq) >= 2) else seq
                c[fin][tuple(tail)] += n
        return c

    def best_syl(self, s):
        c = self.syl.get(s)
        return c.most_common(1)[0][0] if c else None

    def expected(self, s, anchors=None):
        """Expected base sequence for syllable s. anchors: {'I:z': base, 'F:an3': (tail,)} override."""
        anchors = anchors or {}
        if anchors.get("S:" + s) is not None:
            return tuple(anchors["S:" + s]), "anchor-syllable"
        ini, fin = split_syl(s)
        ib, ft = self.initial_base(), self.final_tail()
        i_b = anchors.get("I:" + ini) if ini else None
        if ini and i_b is None and ib.get(ini):
            i_b = ib[ini].most_common(1)[0][0]
        f_t = anchors.get("F:" + fin)
        if f_t is None and ft.get(fin):
            f_t = ft[fin].most_common(1)[0][0]
        if ini and i_b is not None and f_t is not None:
            return (i_b,) + tuple(f_t), "composed"
        if not ini and f_t is not None:
            return tuple(f_t), "composed"
        b = self.best_syl(s)
        if b:
            return b, "table"
        return None, "unknown"

    # ---------- segmentation ----------
    def _score_chunk(self, s, chunk, ib, ft):
        if self.syl.get(s) and chunk in self.syl[s]:
            return 4
        ini, fin = split_syl(s)
        sc = 0
        if ini:
            if len(chunk) < 2:
                return -3
            if ib.get(ini):
                sc += 1.5 if ib[ini].most_common(1)[0][0] == chunk[0] else -2
            tail = chunk[1:]
        else:
            tail = chunk
        if ft.get(fin):
            sc += 1.5 if ft[fin].most_common(1)[0][0] == tuple(tail) else -2
        # prior: 2 phones for syllables with initial, 1 without
        prior = 2 if ini else (2 if fin[0] in "yw" else 1)
        sc += 0.2 if len(chunk) == prior else 0
        return sc

    def segment(self, syls, bases, alt=None):
        """Best split of bases into len(syls) chunks. alt: {idx: [candidate syllables]} to try at idx.
        Returns (score, chunks, chosen_syls) or None."""
        n = len(syls); ib, ft = self.initial_base(), self.final_tail()
        best = None
        for lens in itertools.product((1, 2, 3), repeat=n):
            if sum(lens) != len(bases):
                continue
            chunks, p = [], 0
            for L in lens:
                chunks.append(tuple(bases[p:p + L])); p += L
            tot, chosen = 0, []
            for i, (s, ch) in enumerate(zip(syls, chunks)):
                cands = (alt or {}).get(i) or [s]
                scs = [(self._score_chunk(c, ch, ib, ft), c) for c in cands]
                sc, c = max(scs)
                tot += sc; chosen.append(c)
            if best is None or tot > best[0]:
                best = (tot, chunks, chosen)
        return best

    @staticmethod
    def _w(word, i):
        """Polyphonic chars: pypinyin's guess may disagree with the ASR lexicon -> low weight."""
        if not word or i >= len(word):
            return 1.0
        from pinyin_util import is_monophone_char
        return 1.0 if is_monophone_char(word[i]) else 0.2

    def fit(self, iters=4):
        self.syl = defaultdict(Counter)
        # seed: words where phone count == 2 per syllable-with-initial, 1 otherwise
        for syls, bases, word in self.pairs:
            exp = [2 if (split_syl(s)[0] or s[0] in "yw") else 1 for s in syls]
            if sum(exp) == len(bases):
                p = 0
                for i, (s, L) in enumerate(zip(syls, exp)):
                    self.syl[s][tuple(bases[p:p + L])] += self._w(word, i); p += L
        for _ in range(iters):
            new = defaultdict(Counter)
            for syls, bases, word in self.pairs:
                if len(syls) > 6:
                    continue
                r = self.segment(syls, bases)
                if r is None:
                    continue
                for i, (s, ch) in enumerate(zip(syls, r[1])):
                    new[s][ch] += self._w(word, i)
            self.syl = new
        return self

    def base_labels(self):
        """Human-readable guess per base id: initials and final tails."""
        lab = defaultdict(Counter)
        for ini, c in self.initial_base().items():
            for b, n in c.items(): lab[b][ini] += n
        for fin, c in self.final_tail().items():
            for t, n in c.items():
                if len(t) == 1: lab[t[0]][fin] += n
                elif len(t) == 2:
                    lab[t[0]]["介" + fin[0]] += n
                    lab[t[1]][fin] += n * 0.5
        return {b: c.most_common(1)[0][0] for b, c in lab.items()}
