"""Kaldi decoding helpers for vosk-model-cn-0.22 (CPU)."""
import os, subprocess, re
from pathlib import Path

ROOT = Path(os.environ.get("TTSV_ROOT", Path.home() / "tts-verify"))
M = ROOT / "vosk-model-cn-0.22"
KENV = ROOT / "miniforge3/envs/kaldi"


def kaldi_env():
    env = os.environ.copy()
    kb, el = KENV / "bin", KENV / "lib"
    env["PATH"] = f"{kb}:{env.get('PATH','')}"
    env["LD_LIBRARY_PATH"] = f"{el}/fst:{el}:{env.get('LD_LIBRARY_PATH','')}"
    return env


def run(cmd, cwd, log):
    with open(log, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n"); f.flush()
        r = subprocess.run(list(map(str, cmd)), cwd=cwd, env=kaldi_env(), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"kaldi cmd failed ({r.returncode}): {cmd[0]}; see {log}")


def to16k(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(dst)], check=True)


def load_word_boundary():
    wb = {}
    for line in open(M / "graph/phones/word_boundary.int"):
        i, t = line.split(); wb[int(i)] = t
    return wb


def base_of(pid):
    return (pid - 11) // 4 if pid > 10 else -1


def read_ark_t(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if p: out[p[0]] = p[1:]
    return out


def segment(phone_ids, wb):
    """Split phone id sequence into word chunks via position markers. Silence/nonword dropped."""
    words, cur = [], []
    for pid in phone_ids:
        t = wb.get(pid, "nonword") if pid > 10 else "nonword"
        if t == "nonword":
            if cur: words.append(cur); cur = []
            continue
        if t in ("begin", "singleton") and cur:
            words.append(cur); cur = []
        cur.append(pid)
        if t in ("end", "singleton"):
            words.append(cur); cur = []
    if cur: words.append(cur)
    return words


def decode(utts, workdir, nbest=10):
    """utts: {uttid: wavpath(any format)}. Returns {uttid: {words, phones, nbest:[{words,phones}]}}"""
    wd = Path(workdir); wd.mkdir(parents=True, exist_ok=True)
    log = wd / "kaldi.log"
    scp, s2u = [], []
    for u, p in utts.items():
        dst = wd / f"{u}_16k.wav"; to16k(p, dst)
        scp.append(f"{u} {dst}"); s2u.append(f"{u} {u}")
    (wd / "wav.scp").write_text("\n".join(scp) + "\n")
    (wd / "spk2utt.txt").write_text("\n".join(s2u) + "\n")
    (wd / "ivector.conf").write_text(f"""--cmvn-config={M}/ivector/online_cmvn.conf
--ivector-period=10
--splice-config={M}/ivector/splice.conf
--lda-matrix={M}/ivector/final.mat
--global-cmvn-stats={M}/ivector/global_cmvn.stats
--diag-ubm={M}/ivector/final.dubm
--ivector-extractor={M}/ivector/final.ie
--num-gselect=5
--min-post=0.025
--posterior-scale=0.1
""")
    n = sum(1 for _ in open(M / "graph/phones/word_boundary.int"))
    (wd / "phones_dummy.txt").write_text("\n".join(f"P{i} {i}" for i in range(0, n + 1)) + "\n")
    W = M / "graph/words.txt"
    run(["online2-wav-nnet3-latgen-faster", f"--word-symbol-table={W}", f"--config={M}/conf/model.conf",
         f"--mfcc-config={M}/conf/mfcc.conf", "--ivector-extraction-config=ivector.conf", "--do-endpointing=false",
         M / "am/final.mdl", M / "graph/HCLG.fst", "ark,t:spk2utt.txt", "scp:wav.scp", "ark:lat.ark"], wd, log)
    run(["lattice-best-path", f"--word-symbol-table={W}", "ark:lat.ark", "ark,t:best_words.ark", "ark,t:ali_words.txt"], wd, log)
    run(["lattice-to-phone-lattice", M / "am/final.mdl", "ark:lat.ark", "ark:phonelat.ark"], wd, log)
    run(["lattice-best-path", "--word-symbol-table=phones_dummy.txt", "ark:phonelat.ark", "ark,t:best_phones.ark", "ark,t:ali_phones.txt"], wd, log)
    run(["lattice-to-nbest", f"--n={nbest}", "ark:lat.ark", "ark:nb.ark"], wd, log)
    run(["lattice-best-path", f"--word-symbol-table={W}", "ark:nb.ark", "ark,t:nb_words.ark", "ark,t:/dev/null"], wd, log)
    run(["lattice-to-phone-lattice", M / "am/final.mdl", "ark:nb.ark", "ark:nbphone.ark"], wd, log)
    run(["lattice-best-path", "--word-symbol-table=phones_dummy.txt", "ark:nbphone.ark", "ark,t:nb_phones.ark", "ark,t:/dev/null"], wd, log)
    id2w = {}
    for line in open(W, encoding="utf-8"):
        w, i = line.split(); id2w[i] = w
    def words_of(toks):  # best_words.ark is ark,t of ids? lattice-best-path writes ids
        return [id2w.get(t, t) for t in toks]
    bw, bp = read_ark_t(wd / "best_words.ark"), read_ark_t(wd / "best_phones.ark")
    nw, npn = read_ark_t(wd / "nb_words.ark"), read_ark_t(wd / "nb_phones.ark")
    res = {}
    for u in utts:
        nb = []
        for k in range(1, nbest + 1):
            key = f"{u}-{k}"
            if key in npn:
                nb.append({"words": words_of(nw.get(key, [])), "phones": [int(x) for x in npn[key]]})
        res[u] = {"words": words_of(bw.get(u, [])), "phones": [int(x) for x in bp.get(u, [])], "nbest": nb}
    return res
