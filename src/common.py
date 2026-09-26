"""Shared IO + text normalisation for the business entity resolution pipeline."""
import os, re
import pyarrow as pa, pyarrow.csv as pv
from unidecode import unidecode

# In the submission package the source lives at
# code/business_entity_resolution/src/, while dataset/ and work/ stay at the
# challenge root -- so the root cannot simply be inferred from __file__ there.
# ER_ROOT overrides it; the default keeps the in-place layout working.
ROOT = os.environ.get("ER_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "output")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

_STR = pa.string()
_TYPES = {c: _STR for c in ["entity_id", "business_name", "business_address", "country",
                            "source1_entity_id", "matched_entity_ids", "candidate_entity_ids"]}
_OPTS = dict(parse_options=pv.ParseOptions(delimiter="\t", newlines_in_values=False),
             read_options=pv.ReadOptions(block_size=1 << 26),
             convert_options=pv.ConvertOptions(column_types=_TYPES, strings_can_be_null=True,
                                               null_values=[]))


def read_tsv(path):
    return pv.read_csv(path, **_OPTS)


def source_path(split, src):
    return os.path.join(DATA, split, "%s_source%d.tsv" % (split, src))


def load_source(split, src):
    """-> dict of python lists: ids, names, addrs, countries"""
    t = read_tsv(source_path(split, src))
    return dict(ids=t["entity_id"].to_pylist(),
                names=t["business_name"].fill_null("").to_pylist(),
                addrs=t["business_address"].fill_null("").to_pylist(),
                countries=t["country"].fill_null("").to_pylist())


def load_gt(path=None):
    t = read_tsv(path or os.path.join(DATA, "train", "train_ground_truth.tsv"))
    return t["source1_entity_id"].to_pylist(), t["matched_entity_ids"].fill_null("").to_pylist()


# ---------------------------------------------------------------- normalisation
_NONALNUM = re.compile(r"[^a-z0-9]+")
_REPEAT = re.compile(r"(.)\1+")
_WS = re.compile(r"\s+")

# Legal-form tokens collapsed to one canonical symbol: they carry almost no
# discriminative signal and appear in wildly inconsistent forms across sources.
LEGAL = {
    "pvt": "pvt", "private": "pvt", "pvtltd": "pvt", "prvt": "pvt",
    "praivet": "pvt", "praivett": "pvt", "piraivet": "pvt", "prv": "pvt", "pra": "pvt",
    "ltd": "ltd", "limited": "ltd", "limitd": "ltd", "limitet": "ltd", "lmtd": "ltd",
    "limittedd": "ltd", "li": "ltd",
    "llp": "llp", "llc": "llc", "lc": "llc", "lp": "llp",
    "inc": "inc", "incorporated": "inc", "incorp": "inc",
    "corp": "corp", "corporation": "corp", "corporated": "corp",
    "co": "co", "company": "co", "cie": "co",
    "sa": "sa", "sas": "sas", "sarl": "sarl", "sasu": "sas", "eurl": "sarl", "sci": "sci",
    "gmbh": "gmbh", "bv": "bv", "nv": "nv", "plc": "plc", "ag": "ag",
}
# purely decorative prefixes / honorifics seen in the data
STOPNAME = {"the", "ms", "mr", "mrs", "smt", "shri", "sri", "sh", "la", "le", "les",
            "de", "du", "des", "et", "and", "of", "for"}


def norm_text(s):
    """Script-folding normaliser: unidecode -> lower -> strip punct -> collapse repeats.

    Collapsing runs of repeated characters is what makes Indic transliterations
    ('limittedd') line up with their Latin counterparts ('limited')."""
    if not s:
        return ""
    s = unidecode(s).lower()
    s = _NONALNUM.sub(" ", s)
    s = _REPEAT.sub(r"\1", s)
    return _WS.sub(" ", s).strip()


def norm_name(s):
    """-> (canonical string, core tokens without legal forms, legal-form token set)"""
    n = norm_text(s)
    if not n:
        return "", [], set()
    toks = n.split()
    legal, core = set(), []
    for w in toks:
        lw = LEGAL.get(w)
        if lw is not None:
            legal.add(lw)
        elif w in STOPNAME:
            continue
        else:
            core.append(w)
    if not core:                      # name was nothing but legal forms
        core = [w for w in toks if w not in STOPNAME] or toks
    return " ".join(toks), core, legal


# address component canonicalisation --------------------------------------------
ADDR_ABBR = {
    "road": "rd", "rd": "rd", "street": "st", "st": "st", "avenue": "ave", "av": "ave",
    "ave": "ave", "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct",
    "ct": "ct", "circle": "cir", "cir": "cir", "boulevard": "blvd", "blvd": "blvd",
    "highway": "hwy", "hwy": "hwy", "place": "pl", "pl": "pl", "parkway": "pkwy",
    "pkwy": "pkwy", "terrace": "ter", "ter": "ter", "square": "sq", "sq": "sq",
    "trail": "trl", "trl": "trl", "way": "way", "suite": "ste", "ste": "ste",
    "apartment": "apt", "apt": "apt", "unit": "unit", "floor": "flr", "flr": "flr",
    "building": "bldg", "bldg": "bldg", "block": "blk", "blk": "blk",
    "sector": "sec", "sec": "sec", "village": "vil", "vil": "vil", "vill": "vil",
    "township": "twp", "twp": "twp", "north": "n", "south": "s", "east": "e", "west": "w",
}

# state / region synonyms -> one code, covering the forms that actually occur
# (US postal codes, Indian states incl. transliterated spellings).
STATE = {
    "maharashtra": "mh", "mhaarastr": "mh", "mh": "mh",
    "karnataka": "ka", "krnaatk": "ka", "ka": "ka", "krnatk": "ka",
    "tamilnadu": "tn", "tmilnaadu": "tn", "tn": "tn", "tamiznaatu": "tn",
    "telangana": "tg", "telnganaa": "tg", "tg": "tg", "telngaana": "tg",
    "andhrapradesh": "ap", "ap": "ap", "aandhrprdes": "ap",
    "gujarat": "gj", "gujraat": "gj", "gj": "gj",
    "westbengal": "wb", "wb": "wb", "pshcimbng": "wb", "pscimbng": "wb",
    "delhi": "dl", "dl": "dl", "newdelhi": "dl", "dili": "dl",
    "uttarpradesh": "up", "up": "up", "utrprdes": "up",
    "rajasthan": "rj", "rj": "rj", "raajsthaan": "rj",
    "madhyapradesh": "mp", "mp": "mp", "mdhyprdes": "mp",
    "kerala": "kl", "kl": "kl", "kerl": "kl",
    "punjab": "pb", "pb": "pb", "haryana": "hr", "hr": "hr",
    "bihar": "br", "br": "br", "odisha": "od", "od": "od", "orissa": "od",
    "assam": "as", "as": "as", "jharkhand": "jh", "jh": "jh",
    "chhattisgarh": "cg", "cg": "cg", "uttarakhand": "uk", "uk": "uk",
    "himachalpradesh": "hp", "hp": "hp", "jammu": "jk", "jk": "jk",
    "chandigarh": "ch", "puducherry": "py", "py": "py",
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct_us", "delaware": "de", "florida": "fl",
    "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in",
    "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "newhampshire": "nh", "newjersey": "nj", "newmexico": "nm",
    "newyork": "ny", "northcarolina": "nc", "northdakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhodeisland": "ri",
    "southcarolina": "sc", "southdakota": "sd", "tennessee": "tn_us", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "westvirginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}

_HASNUM = re.compile(r"\d")


def norm_addr(s):
    """-> (canonical string, token list, numeric-token list)"""
    n = norm_text(s)
    if not n:
        return "", [], []
    toks = [ADDR_ABBR.get(w, w) for w in n.split()]
    nums = [w for w in toks if _HASNUM.search(w)]
    return " ".join(toks), toks, nums
