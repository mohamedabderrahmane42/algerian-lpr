import re
from src.config import CHARSET, BLANK_IDX

WILAYAS = {str(i).zfill(2) for i in range(1, 59)}

def decode_crnn(lp_probs) -> str:
    """
    Decodes the log-probabilities from CRNN to a raw string.
    lp_probs : numpy array of shape (T, num_classes)
    """
    ids = lp_probs.argmax(axis=1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK_IDX: 
            out.append(int(i))
        prev = i
    return ''.join(CHARSET[i] for i in out if i < len(CHARSET))

def _try_format(d: str):
    """Return formatted plate if last 2 digits are a valid wilaya, else None."""
    if len(d) < 6: return None
    wil = d[-2:]
    if wil not in WILAYAS: return None
    mid = d[:-2]
    ml = len(mid)
    # support middle groups of 5, 4, or 3 digits (in priority order for 11-digit plates)
    for rg in (5, 4, 3):
        if ml > rg:
            return f"{mid[:-rg]} {mid[-rg:]} {wil}"
    return f"{mid} {wil}"

def format_algerian(digits: str) -> str:
    """
    Smart Algerian plate formatter with wilaya validation.
    Handles: 10-digit, 11-digit, and OCR error-correction.
    """
    d = re.sub(r'[^0-9]', '', digits)

    # 1. Try original string first (handles valid 10 and 11 digit plates)
    result = _try_format(d)
    if result:
        return result

    # 2. Error-correction: try removing one digit if original failed 
    # (very useful for 11-digit OCR outputs caused by screws/edges)
    if len(d) == 11:
        for i in range(len(d)):
            result = _try_format(d[:i] + d[i+1:])
            if result:
                return result

    # 3. Hard fallback
    n = len(d)
    if n >= 10: return f"{d[:5]} {d[5:8]} {d[8:10]}"
    if n == 9:  return f"{d[:4]} {d[4:7]} {d[7:]}"
    return d
