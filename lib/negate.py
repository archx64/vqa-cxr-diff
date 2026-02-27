# negate.py
import re

_REPL = [
    # left/right
    (r"\bleft\b", "<<<RIGHT>>>"),
    (r"\bright\b", "left"),
    (r"<<<RIGHT>>>", "right"),
    # increase/decrease / improved/worsened / new/resolved
    (r"\bincrease(d)?\b", "decrease\\1"),
    (r"\bdecrease(d)?\b", "increase\\1"),
    (r"\bworsen(ed|ing)?\b", "improv\\1"),
    (r"\bimprov(ed|ement|ing)?\b", "worsen\\1"),
    (r"\bnew\b", "resolved"),
    (r"\bresolved\b", "new"),
    # presence/absence
    (r"\bis there\b", "is there no"),
    (r"\bno\b", "yes"),
    (r"\byes\b", "no"),
    # higher/lower
    (r"\bhigher\b", "lower"),
    (r"\blower\b", "higher"),
]


def negate_question(q: str) -> str:
    s = q.strip().lower()
    for pat, rep in _REPL:
        s = re.sub(pat, rep, s)
    return s
