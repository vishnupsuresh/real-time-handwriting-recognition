"""CTC prefix beam search with a character bigram language model.

Greedy decoding picks the single best character per frame and ignores
spelling, so "headguarters" can win over "headquarters". Beam search keeps the
`beam` most likely partial transcriptions and adds a language-model score for
each new character, so English-looking text is preferred when the image is
ambiguous.

Score of a candidate prefix y:

    score(y) = log P_ctc(y | x) + α · log P_lm(y) + β · |y|

α (alpha) sets how much to trust the language model; β (beta) is a per-character
bonus that offsets the LM's bias toward short outputs.

Weaknesses:
  - A bigram only sees one previous character, so it knows "qu" is common but
    not whole words. Fix: a word-level or 5-gram model (KenLM + pyctcdecode).
  - Pure Python: about 10-50 ms per line, slower than greedy. Fix: pyctcdecode
    or a C++ decoder for production.
  - It can "correct" names or codes that are not English. Lower alpha for those.
"""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .charset import BLANK, CHARS, IDX2CHAR

NEG_INF = -1e30
WORDS_FILE = Path(__file__).resolve().parent.parent / "data" / "words.txt"


class CharBigramLM:
    """P(c | previous char), estimated from the English word list.

    Case-insensitive (counts are shared between 'a' and 'A') so capitals do
    not get penalized for being rare in the word list. Words are weighted by
    1 / rank (Zipf's law) because the list is sorted by frequency: "the"
    should count far more than "verzeichnis".
    """

    def __init__(self, words_file: Path = WORDS_FILE, k: float = 0.5):
        vocab = sorted({c.lower() for c in CHARS})
        self.index = {c: i for i, c in enumerate(vocab)}
        counts = np.full((len(vocab), len(vocab)), k)  # add-k smoothing
        sp = self.index[" "]
        words = [w.strip().lower() for w in words_file.read_text().split() if w.strip()]
        for rank, w in enumerate(words, 1):
            weight = 1.0 / rank
            seq = [sp] + [self.index[c] for c in w if c in self.index] + [sp]
            for a, b in zip(seq, seq[1:]):
                counts[a, b] += weight * 1000
        self.logp_table = np.log(counts / counts.sum(1, keepdims=True))

    def logp(self, prev: str | None, c: str) -> float:
        a = self.index.get((prev or " ").lower(), self.index[" "])
        b = self.index.get(c.lower(), self.index[" "])
        return float(self.logp_table[a, b])


@lru_cache(maxsize=1)
def default_lm() -> CharBigramLM:
    return CharBigramLM()


def beam_decode(logprobs: np.ndarray, lm: CharBigramLM | None = None, beam: int = 10,
                alpha: float = 0.1, beta: float = 0.2, prune: float = -10.0) -> str:
    """logprobs: (T, C) log-softmax output for one line."""
    # Each prefix keeps two probabilities: ending in blank (pb) or in its last
    # character (pnb). They must be separate because "a", blank, "a" means
    # "aa", while "a", "a" collapses to a single "a".
    beams = {(): (0.0, NEG_INF)}
    lm_score = {(): 0.0}

    for t in range(logprobs.shape[0]):
        lp = logprobs[t]
        # Pruning: skip characters with negligible probability this frame.
        cands = np.nonzero(lp > prune)[0]
        nxt = defaultdict(lambda: [NEG_INF, NEG_INF])
        for prefix, (pb, pnb) in beams.items():
            total = np.logaddexp(pb, pnb)
            for c in cands:
                p = lp[c]
                if c == BLANK:
                    e = nxt[prefix]
                    e[0] = np.logaddexp(e[0], total + p)
                    continue
                last = prefix[-1] if prefix else None
                new = prefix + (int(c),)
                if new not in lm_score:
                    bonus = 0.0
                    if lm is not None:
                        prev_ch = IDX2CHAR[last] if last is not None else None
                        bonus = alpha * lm.logp(prev_ch, IDX2CHAR[int(c)])
                    lm_score[new] = lm_score[prefix] + bonus + beta
                e = nxt[new]
                if c == last:
                    # Repeat after blank starts a new character; repeat
                    # without blank stays the same character.
                    e[1] = np.logaddexp(e[1], pb + p)
                    same = nxt[prefix]
                    same[1] = np.logaddexp(same[1], pnb + p)
                else:
                    e[1] = np.logaddexp(e[1], total + p)
        ranked = sorted(nxt.items(),
                        key=lambda kv: np.logaddexp(*kv[1]) + lm_score[kv[0]],
                        reverse=True)
        beams = {k: tuple(v) for k, v in ranked[:beam]}

    best = max(beams, key=lambda k: np.logaddexp(*beams[k]) + lm_score[k])
    return "".join(IDX2CHAR[i] for i in best)
