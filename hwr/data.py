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

from .charset import CHAR2IDX, encode
from .synth import list_fonts, random_text, render

IMG_H = 32       # small height keeps CPU training feasible; 64 is better on GPU
MAX_W = 512      # caps memory per batch; longer lines are downscaled


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
        img = render(text, font, augment=True)
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


def collate(batch):
    """Pad images to the widest in the batch and flatten labels for CTCLoss."""
    imgs, texts, styles = zip(*batch)
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
