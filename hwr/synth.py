"""Synthetic handwriting line generator.

Why synthetic data: real datasets (IAM, CVL, RIMES) need registration and
cannot be downloaded from this environment. Rendering text with handwriting
fonts and distorting it gives unlimited labeled lines for both print and
cursive. It is the standard way to pre-train HTR models before fine-tuning on
real scans.

Weakness: fonts repeat each glyph identically, while real writers vary every
stroke and join letters inconsistently. A model trained only on fonts will
score much worse on real handwriting. Fix: fine-tune on IAM (see data.py),
or mix both sources during training.
"""
import random
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from .charset import CHARS

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "fonts"
WORDS = [w.strip() for w in (ROOT / "data" / "words.txt").read_text().split() if w.strip()]

# Fonts kept out of training so validation measures generalization to
# handwriting styles the model has never seen, not memorization of a font.
HELDOUT = {
    "print": ["Kalam-Regular.ttf", "GochiHand-Regular.ttf", "Neucha.ttf"],
    "cursive": ["Allura-Regular.ttf", "Satisfy-Regular.ttf", "Courgette-Regular.ttf",
                "DawningofaNewDay.ttf"],
}
# AmaticSC draws lowercase as small capitals, so its glyphs would contradict
# the labels. Excluded entirely.
EXCLUDE = {"AmaticSC-Regular.ttf"}


def _covers_charset(path: Path) -> bool:
    """Reject fonts missing glyphs, otherwise they render empty boxes."""
    cmap = TTFont(str(path)).getBestCmap()
    return all(ord(c) in cmap for c in CHARS if c != " ")


def list_fonts(split: str) -> list[tuple[Path, str]]:
    """Return (font_path, style) pairs for 'train' or 'val'."""
    out = []
    for style in ("print", "cursive"):
        for p in sorted((FONT_DIR / style).glob("*.ttf")):
            if p.name in EXCLUDE or not _covers_charset(p):
                continue
            held = p.name in HELDOUT[style]
            if (split == "val") == held:
                out.append((p, style))
    return out


def random_text(max_words: int = 4) -> str:
    """Random phrase with mixed case, digits and punctuation.

    A word list is used instead of uniform random characters so the model also
    picks up English letter statistics, which helps on ambiguous cursive joins.
    """
    words = []
    for _ in range(random.randint(1, max_words)):
        r = random.random()
        if r < 0.08:
            w = str(random.randint(0, 9999))
        else:
            w = random.choice(WORDS)
            r2 = random.random()
            if r2 < 0.25:
                w = w.capitalize()
            elif r2 < 0.30:
                w = w.upper()
        if random.random() < 0.12:
            w += random.choice(list(".,;:!?"))
        if random.random() < 0.03:
            w = random.choice(['"', "'", "("]) + w + random.choice(['"', "'", ")"])
        words.append(w)
    return " ".join(words)


def _elastic(img: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """Elastic distortion: smooth random displacement field.

    Simulates the wobble of a hand-drawn stroke so every rendered glyph
    differs slightly from the font's template.
    """
    h, w = img.shape
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, x + dx, y + dy, cv2.INTER_LINEAR, borderValue=255)


_font_cache: dict = {}


def render(text: str, font_path: Path, augment: bool = True) -> np.ndarray:
    """Render one line of text to a grayscale uint8 image (white background)."""
    size = random.randint(36, 56)
    key = (str(font_path), size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(str(font_path), size)
    font = _font_cache[key]

    l, t, r, b = font.getbbox(text)
    pad = 20
    canvas = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 255)
    ImageDraw.Draw(canvas).text((pad - l, pad - t), text, font=font, fill=0)
    img = np.array(canvas)

    if augment:
        h, w = img.shape
        # Slant: writers tilt letters left or right. Shear in x.
        shear = random.uniform(-0.35, 0.35)
        M = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
        # Small baseline rotation.
        R = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-3, 3), 1.0)
        img = cv2.warpAffine(img, R, (w, h), borderValue=255)
        if random.random() < 0.7:
            img = _elastic(img, alpha=random.uniform(1, 4), sigma=random.uniform(4, 8))
        # Pen width: thicken or thin strokes with morphology.
        k = np.ones((2, 2), np.uint8)
        r = random.random()
        if r < 0.3:
            img = cv2.erode(img, k)   # erode on white bg = thicker ink
        elif r < 0.45:
            img = cv2.dilate(img, k)  # thinner ink
        # Ink and paper intensity, blur, sensor noise.
        ink = random.randint(0, 90)
        paper = random.randint(190, 255)
        img = (img.astype(np.float32) / 255) * (paper - ink) + ink
        if random.random() < 0.5:
            img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 1.0))
        img += np.random.normal(0, random.uniform(0, 12), img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return _crop(img)


def _crop(img: np.ndarray, margin: int = 4) -> np.ndarray:
    """Crop to the ink bounding box so text height is consistent after resizing."""
    mask = img < (img.max() - 40)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, img.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, img.shape[1])
    return img[y0:y1, x0:x1]
