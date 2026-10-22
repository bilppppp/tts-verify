"""Pinyin helpers: syllable keys like 'cuan2', components (initial, final+tone)."""
import re
from pypinyin import pinyin, lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_tone3, to_initials, to_finals_tone3

INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]


def norm_syl(s):
    """Accept 'cuán', 'cuan2', 'lv4', 'lü4', 'ma' -> tone3 key with v for ü and 5 for neutral."""
    s = s.strip().lower().replace("u:", "v")
    if not re.search(r"[1-5]$", s):
        t = to_tone3(s, neutral_tone_with_five=True)
        s = t if re.search(r"[1-5]$", t) else s + "5"
    s = s.replace("ü", "v")
    return s


def split_syl(syl):
    """'quan2' -> ('q', 'van2'); 'an4' -> ('', 'an4'). Finals normalized (ü after j/q/x/y written v)."""
    syl = norm_syl(syl)
    tone = syl[-1]; body = syl[:-1]
    ini = ""
    for i in INITIALS:
        if body.startswith(i):
            ini = i; break
    fin = body[len(ini):]
    if ini in ("j", "q", "x") and fin.startswith("u"):
        fin = "v" + fin[1:]
    elif ini in ("z", "c", "s") and fin == "i":
        fin = "ii"      # apical vowel after z/c/s is a different phone in this model
    elif ini in ("zh", "ch", "sh", "r") and fin == "i":
        fin = "iii"     # retroflex apical vowel
    # y-/w- syllables and zero-initial syllables: no consonant initial; the whole body is the 'final' key
    return ini, fin + tone


def chars_pinyin(text, heteronym=False):
    """List per char of tone3 readings (list), non-hanzi -> None."""
    out = []
    for ch in text:
        if re.match(r"[\u4e00-\u9fff]", ch):
            r = pinyin(ch, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
            out.append([norm_syl(x) for x in r])
        else:
            out.append(None)
    return out


def word_pinyin(word):
    """Context reading of a word via pypinyin phrase dict."""
    return [norm_syl(x) for x in lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)]


def is_hanzi(ch):
    return bool(re.match(r"[\u4e00-\u9fff]", ch))


def is_monophone_char(ch):
    """Char has exactly one reading in pypinyin's single-char dict (ignoring tone-sandhi)."""
    r = pinyin(ch, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
    return len(set(norm_syl(x) for x in r)) == 1


def is_gb2312_level1(ch):
    try:
        b = ch.encode("gb2312")
    except Exception:
        return False
    return len(b) == 2 and 0xB0 <= b[0] <= 0xD7
