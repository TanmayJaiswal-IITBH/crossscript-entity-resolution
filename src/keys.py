"""Blocking-key generation.

Every key is a string that already embeds the country, so blocks never cross
countries.  Keys are designed to be individually selective (small buckets) and
jointly high-recall: a true pair only has to agree on ONE key to be retrieved.

Families
  NS   sorted core-name tokens                     word-order / legal-suffix proof
  NS4  sorted 4-char prefixes of core tokens       typo-at-the-end proof
  NSK  sorted consonant skeletons of core tokens   vowel/transliteration proof
  NT   a single long core token                    truncated-name / DBA proof
  DGS  address digit signature + state             street-number proof
  DGT  address digit signature + one address token
  AT2  two address tokens (the two longest)        street + city proof
  NTA  one core name token + address digit sig     cross-script proof

`gen_pool_keys` is what the 5M-record pools use; `gen_query_keys` is the same
list (queries and pool records must agree on the key alphabet).
"""
import re

_VOWELS = re.compile(r"[aeiou]")
_MIN_TOKEN_LEN = 5
_MAX_TOKENS = 6


def _skel(t):
    s = _VOWELS.sub("", t)
    return (s or t)[:5]


def gen_keys(country, core, addr, digits, state):
    """core/addr are pre-normalised space-joined strings."""
    out = []
    c = country[:2]
    ct = core.split()
    if ct:
        u = sorted(set(ct))[:_MAX_TOKENS]
        j = "".join(u)
        if len(j) >= 4:
            out.append("NS" + c + j)
            out.append("NS4" + c + "".join(sorted(set(t[:4] for t in u))))
            out.append("NSK" + c + "".join(sorted(set(_skel(t) for t in u))))
        # single long tokens: rescues truncated names, DBA prefixes, extra words
        longs = sorted((t for t in set(ct) if len(t) >= _MIN_TOKEN_LEN),
                       key=len, reverse=True)[:3]
        for t in longs:
            out.append("NT" + c + t)
    at = [t for t in addr.split() if len(t) >= 4 and not t.isdigit()]
    at_long = sorted(set(at), key=len, reverse=True)[:3]
    if digits and len(digits) >= 3:
        d = digits[:12]
        if state:
            out.append("DGS" + c + d + state)
        for t in at_long[:2]:
            out.append("DGT" + c + d + t)
        for t in (ct[:1] if ct else []):
            out.append("NTA" + c + t + d)
    if len(at_long) >= 2:
        a, b = sorted(at_long[:2])
        out.append("AT2" + c + a + b)
        if len(at_long) >= 3:
            a, b = sorted(at_long[1:3])
            out.append("AT2" + c + a + b)
    return out


gen_pool_keys = gen_keys
gen_query_keys = gen_keys
