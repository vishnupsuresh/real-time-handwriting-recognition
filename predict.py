"""Transcribe line images.

  python predict.py checkpoints/best.pt line1.png line2.jpg

Input must be a single line of text (a photo or scan cropped to one line).
Weakness: there is no page segmentation, so a full page fails. Fix: split the
page into lines first, e.g. horizontal projection profiles for clean scans, or
a text-line detector such as Kraken's segmenter for photos.
"""
import sys

import cv2
import torch

from hwr.charset import greedy_decode
from hwr.data import preprocess
from hwr.model import CRNN


def load(path):
    model = CRNN()
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return model.eval()


def transcribe(model, img_gray):
    x = torch.from_numpy(preprocess(img_gray))[None, None]
    with torch.no_grad():
        return greedy_decode(model(x)[0].argmax(-1))


if __name__ == "__main__":
    model = load(sys.argv[1])
    for p in sys.argv[2:]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        # Binarize photos with uneven lighting so paper texture is not read as ink.
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 15)
        print(f"{p}: {transcribe(model, img)}")
