"""Response-preservation and refusal helpers used for benign utility audits."""
from collections import Counter
import re
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def token_f1(left, right):
    a, b = TOKEN_RE.findall(str(left).casefold()), TOKEN_RE.findall(str(right).casefold())
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return 2 * sum((Counter(a) & Counter(b)).values()) / (len(a) + len(b))

def refusal(text):
    value = str(text).casefold()
    return any(x in value for x in ("i don't know", "i do not know", "cannot determine",
                                     "insufficient context", "not enough information"))
