"""Datasets and batching.

All images are resized to a fixed height and variable width (aspect ratio
kept). Squashing every line to one fixed width would distort long lines and
make character widths inconsistent, which hurts CTC alignment.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .charset import CHAR2IDX, ctc_min_len, encode
from .synth import list_fonts, random_text, render, vary_spacing

IMG_H = 32       # small height keeps CPU training feasible; 64 is better on GPU
MAX_W = 1024     # fits 8-word lines without squeezing; was 512 for 4-word lines


def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold: dark ink -> 0, paper -> 255.

    Shared by training augmentation, live.py and predict.py so the model sees
    the same kind of black-and-white image at inference as in training.
    Adaptive (local) thresholding copes with shadows and uneven room light,
    which a single global threshold cannot.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)


def compress_gaps(img: np.ndarray, max_gap: float = 0.8) -> np.ndarray:
    """Shrink blank column runs wider than max_gap * height (dark ink on light).

    When live.py merges words far apart on one line, the crop contains a long
    empty stretch the model never saw in training, and it starts merging
    words ("meet me at noon" -> "meet meat non"). Cutting every gap down to
    a normal word space fixes that without changing the text.
    """
    h = img.shape[0]
    ink = (img < 128).any(axis=0)
    limit = max(int(max_gap * h), 1)
    keep, run = np.ones(img.shape[1], bool), 0
    for x, has_ink in enumerate(ink):
        run = 0 if has_ink else run + 1
        if run > limit:
            keep[x] = False
    return img[:, keep]


def preprocess(img: np.ndarray, train: bool = False) -> np.ndarray:
    """Grayscale uint8 -> float32 in [0,1], ink = 1 (inverted), height IMG_H."""
    h, w = img.shape
    new_w = max(8, int(round(w * IMG_H / h)))
    if train:
        # Random horizontal stretch: handwriting widths vary a lot per writer.
        new_w = int(new_w * random.uniform(0.8, 1.2))
    new_w = min(new_w, MAX_W)
    img = cv2.resize(img, (new_w, IMG_H), interpolation=cv2.INTER_AREA)
    # Normalize per image to remove paper/ink contrast differences.
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    img = (img - lo) / max(hi - lo, 1.0)
    return 1.0 - img  # ink bright, background 0, so zero padding = blank paper


class SynthLines(Dataset):
    """Infinite-style synthetic dataset; `length` only defines an epoch."""

    def __init__(self, split: str = "train", length: int = 20000, seed: int | None = None):
        self.fonts = list_fonts(split)
        self.length = length
        self.train = split == "train"
        self.seed = seed  # fixed seed -> identical val set every evaluation

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.seed is not None:
            random.seed(self.seed + i)
            np.random.seed((self.seed + i) % 2**32)
        font, style = random.choice(self.fonts)
        text = random_text()
        img = render(vary_spacing(text), font, augment=True)
        # 30% binarized so the model also learns the black-and-white input
        # that live.py and predict.py feed it. Validation stays grayscale so
        # scores are comparable with earlier runs.
        if self.train and random.random() < 0.3:
            img = binarize(img)
        return preprocess(img, train=self.train), text, style


class IAMLines(Dataset):
    """IAM Handwriting Database, line level.

    Download (free registration) from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
    and extract so that you have:
        <root>/lines/a01/a01-000u/a01-000u-00.png
        <root>/ascii/lines.txt
    IAM contains mixed print and cursive writing from 657 writers.
    `ids` lets you pass a writer-independent split (e.g. the Aachen split)
    so test writers are never seen in training.
    """

    def __init__(self, root: str, ids: set[str] | None = None, train: bool = False):
        self.root, self.train, self.items = Path(root), train, []
        for line in (self.root / "ascii" / "lines.txt").read_text().splitlines():
            if line.startswith("#"):
                continue
            parts = line.split(" ")
            lid, status, text = parts[0], parts[1], " ".join(parts[8:]).replace("|", " ")
            if status != "ok" or (ids and lid not in ids):
                continue  # 'err' lines have segmentation errors
            if any(c not in CHAR2IDX for c in text):
                continue  # see charset.encode weakness note
            a, b = lid.split("-")[:2]
            self.items.append((self.root / "lines" / a / f"{a}-{b}" / f"{lid}.png", text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, text = self.items[i]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return preprocess(img, train=self.train), text, "iam"


def ensure_min_width(img: np.ndarray, text: str, frame_px: int = 4) -> np.ndarray:
    """Widen img so the model outputs at least ctc_min_len(text) frames.

    frame_px matches CRNN.DOWNSAMPLE (not imported, to avoid a circular
    import; tests/test_core.py checks the two stay equal). One spare frame per
    5 characters leaves room for blanks between letters.
    """
    need = ctc_min_len(text)
    min_w = (need + need // 5 + 1) * frame_px
    if img.shape[1] >= min_w:
        return img
    return cv2.resize(img, (min_w, img.shape[0]), interpolation=cv2.INTER_LINEAR)


def collate(batch):
    """Pad images to the widest in the batch and flatten labels for CTCLoss."""
    imgs, texts, styles = zip(*batch)
    # Stretch images too narrow for their label. Very condensed writing can
    # make letters thinner than one output frame (4 px), which CTC cannot
    # align; without this ~0.7% of synthetic samples were silently skipped.
    imgs = [ensure_min_width(im, t) for im, t in zip(imgs, texts)]
    max_w = max(im.shape[1] for im in imgs)
    x = torch.zeros(len(imgs), 1, IMG_H, max_w)
    widths = []
    for k, im in enumerate(imgs):
        x[k, 0, :, : im.shape[1]] = torch.from_numpy(im)
        widths.append(im.shape[1])
    labels = [torch.tensor(encode(t), dtype=torch.long) for t in texts]
    targets = torch.cat(labels)
    target_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return x, targets, target_lens, torch.tensor(widths), list(texts), list(styles)
