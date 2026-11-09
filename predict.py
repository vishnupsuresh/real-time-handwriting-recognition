"""Transcribe line images.

  python predict.py checkpoints/best.pt line1.png line2.jpg
  python predict.py checkpoints/best.pt line.png --beam   # beam search + LM

Input must be a single line of text (a photo or scan cropped to one line).
Weakness: there is no page segmentation, so a full page fails. Fix: reuse
find_lines() from live.py, or a text-line detector such as Kraken for photos.
"""
import argparse

import cv2
import torch

from hwr.charset import greedy_decode
from hwr.data import binarize, compress_gaps, preprocess
from hwr.decode import beam_decode, default_lm
from hwr.model import load_model


@torch.no_grad()
def transcribe(model, img_gray, lm=None):
    # Same binarization as live.py and the training augmentation, which
    # removes paper texture and uneven lighting from photos.
    x = torch.from_numpy(preprocess(compress_gaps(binarize(img_gray))))[None, None]
    logp = model(x)[0].log_softmax(-1)
    return beam_decode(logp.numpy(), lm) if lm else greedy_decode(logp.argmax(-1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--beam", action="store_true")
    args = ap.parse_args()

    model = load_model(args.ckpt)
    lm = default_lm() if args.beam else None
    for p in args.images:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{p}: cannot read image")
            continue
        print(f"{p}: {transcribe(model, img, lm)}")
