"""Fast checks for the parts most likely to break silently.

Run:  pytest -q
"""
import math

import numpy as np
import torch

from hwr.charset import BLANK, CHAR2IDX, NUM_CLASSES, ctc_min_len, encode, greedy_decode
from hwr.data import IMG_H, binarize, collate, compress_gaps, preprocess
from hwr.decode import beam_decode
from hwr.model import CRNN, model_config
from live import merge_same_line


def ids(s):
    return [CHAR2IDX[c] for c in s]


def test_encode_roundtrip_drops_unknown():
    assert encode("a€b") == ids("ab")


def test_greedy_collapses_repeats_and_blanks():
    h, e, l, o = ids("helo")
    seq = [h, h, BLANK, e, l, l, BLANK, l, o, BLANK]
    assert greedy_decode(seq) == "hello"


def test_ctc_min_len_counts_repeats():
    assert ctc_min_len("abc") == 3
    assert ctc_min_len("hello") == 6  # "ll" needs a blank between


def test_beam_matches_greedy_without_lm():
    # Sharp distribution: beam search must give the same answer as greedy.
    seq = ids("cat")
    frames = [seq[0], BLANK, seq[1], seq[2], seq[2]]
    lp = np.full((len(frames), NUM_CLASSES), math.log(1e-4))
    for t, c in enumerate(frames):
        lp[t, c] = math.log(0.99)
    assert beam_decode(lp, lm=None) == greedy_decode(frames) == "cat"


def test_model_output_shape():
    model = CRNN().eval()
    x = torch.zeros(2, 1, IMG_H, 200)
    out = model(x)
    assert out.shape == (2, 200 // CRNN.DOWNSAMPLE, NUM_CLASSES)
    assert CRNN.out_lengths(torch.tensor([200, 201])).tolist() == [50, 50]


def test_preprocess_height_and_range():
    img = np.full((60, 300), 255, np.uint8)
    img[20:40, 50:250] = 0
    out = preprocess(img)
    assert out.shape[0] == IMG_H
    assert out.min() >= 0 and out.max() <= 1
    assert out.max() == 1.0  # ink is bright after inversion


def test_binarize_is_black_and_white():
    img = np.random.randint(0, 256, (50, 120), np.uint8)
    assert set(np.unique(binarize(img))) <= {0, 255}


def test_collate_pads_to_widest():
    a, b = np.ones((IMG_H, 40), np.float32), np.ones((IMG_H, 90), np.float32)
    x, targets, tlen, widths, *_ = collate([(a, "hi", "print"), (b, "there", "cursive")])
    assert x.shape == (2, 1, IMG_H, 90)
    assert tlen.tolist() == [2, 5] and widths.tolist() == [40, 90]
    assert len(targets) == 7


def test_ensure_min_width_makes_label_alignable():
    narrow = np.ones((IMG_H, 20), np.float32)
    x, _, _, widths, *_ = collate([(narrow, "hello world", "print")])
    assert CRNN.out_lengths(widths)[0] >= ctc_min_len("hello world")
    # collate assumes 4 px per frame; must match the model.
    assert CRNN.DOWNSAMPLE == 4


def test_compress_gaps_shrinks_only_wide_gaps():
    img = np.full((10, 100), 255, np.uint8)
    img[:, 0:5] = 0      # word 1
    img[:, 7:12] = 0     # small gap of 2 px: kept
    img[:, 80:85] = 0    # gap of 68 px: shrunk to 8 (0.8 * height)
    out = compress_gaps(img)
    assert out.shape[1] == 5 + 2 + 5 + 8 + 5 + 8


def test_merge_same_line_joins_words():
    words_line1 = [(10, 10, 50, 20), (100, 12, 60, 18)]
    line2 = [(10, 60, 120, 20)]
    merged = merge_same_line(words_line1 + line2)
    assert merged == [(10, 10, 150, 20), (10, 60, 120, 20)]


def test_config_contains_charset():
    cfg = model_config()
    assert cfg["img_h"] == IMG_H and len(cfg["chars"]) == NUM_CLASSES - 1
