"""Character set and CTC label encoding.

CTC needs one extra class, the "blank", which the network emits between
characters and to separate repeated letters (e.g. the two l's in "hello").
We reserve index 0 for blank so every real character maps to 1..N.
"""

# Printable characters found in English handwriting datasets (IAM uses this set
# plus a few rare symbols). Keeping the set small helps a small model converge.
CHARS = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)
BLANK = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for blank


def encode(text: str) -> list[int]:
    # Unknown characters are dropped silently.
    # Weakness: a label with many unsupported symbols becomes shorter than the
    # ink in the image, which teaches the model to skip ink. Fix: filter such
    # samples out of the dataset instead (data.py does this for IAM).
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


def ctc_min_len(text: str) -> int:
    """Fewest output frames CTC needs for this label.

    One frame per character plus one blank between each pair of identical
    neighbors ("ll" needs l, blank, l). If the model outputs fewer frames the
    sample cannot be aligned and its loss is infinite.
    """
    ids = encode(text)
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


def greedy_decode(indices) -> str:
    """Best-path CTC decoding: collapse repeats, then remove blanks.

    Greedy decoding is fast and needs no extra dependencies. Weakness: it
    ignores language context, so "c1ass" can beat "class". Fix: beam search
    with a word-level or character n-gram language model (e.g. pyctcdecode +
    KenLM), which typically lowers word error rate by a few points.
    """
    out, prev = [], BLANK
    for i in indices:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(IDX2CHAR[i])
        prev = i
    return "".join(out)
