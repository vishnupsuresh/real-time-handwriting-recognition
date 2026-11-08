"""Real-time handwriting reader from a webcam.

  python live.py                      # default camera, checkpoints/best.pt
  python live.py --camera 1 --ckpt checkpoints/best.pt

Hold paper with handwriting inside the green box. Each detected text line gets
a blue box and its transcription. Keys: q = quit, space = freeze/unfreeze,
b = toggle the binarized view (shows what the model sees).

Pipeline per frame:
  1. Crop the guide box (region of interest). Restricting the search area
     avoids reading posters, keyboards, etc. in the background.
  2. Binarize with an adaptive threshold, which handles uneven room lighting.
  3. Find text lines: dilate ink horizontally so letters in a line merge into
     one blob, then take each blob's bounding box.
  4. Run the CRNN on every line crop in one padded batch.
  5. Smooth: only replace a line's text when the new reading is stable for a
     few frames, so the overlay does not flicker.
"""
import argparse
import time

import cv2
import numpy as np
import torch

from hwr.charset import greedy_decode
from hwr.data import IMG_H, preprocess
from hwr.model import CRNN


def find_lines(gray: np.ndarray, min_h: int = 12, min_w: int = 30):
    """Return line boxes (x, y, w, h), top to bottom, plus the binary image.

    Weakness: lines that touch (descender of one line meets ascender of the
    next) merge into one box, and strongly tilted lines are cut into pieces.
    Fix: deskew the ROI first (cv2.minAreaRect on all ink), or use a learned
    line detector (Kraken, docTR) for messy pages.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    # Remove specks so noise does not create tiny "lines".
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # Wide, short kernel joins letters and words horizontally but not lines vertically.
    kw = max(gray.shape[1] // 25, 15)
    merged = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 5)))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[3] >= min_h and b[2] >= min_w and b[2] > b[3]]
    return sorted(boxes, key=lambda b: b[1]), binary


@torch.no_grad()
def read_lines(model, binary: np.ndarray, boxes):
    """Batch all line crops through the model in one forward pass."""
    if not boxes:
        return []
    crops = []
    for x, y, w, h in boxes:
        pad = 4
        crop = binary[max(y - pad, 0): y + h + pad, max(x - pad, 0): x + w + pad]
        # preprocess expects dark ink on light paper.
        crops.append(preprocess(255 - crop))
    max_w = max(c.shape[1] for c in crops)
    batch = torch.zeros(len(crops), 1, IMG_H, max_w)
    for i, c in enumerate(crops):
        batch[i, 0, :, : c.shape[1]] = torch.from_numpy(c)
    preds = model(batch).argmax(-1)
    lens = CRNN.out_lengths(torch.tensor([c.shape[1] for c in crops]))
    return [greedy_decode(p[:L]) for p, L in zip(preds, lens)]


class Smoother:
    """Show a line's new text only after it repeats `k` frames in a row.

    Matching is by line index, which is fine for a held-still page.
    Weakness: if a line appears or disappears, indices shift and texts jump.
    Fix: match lines across frames by box overlap (IoU) instead of index.
    """

    def __init__(self, k: int = 3):
        self.k, self.shown, self.cand, self.count = k, [], [], []

    def update(self, texts):
        n = len(texts)
        self.shown = (self.shown + [""] * n)[:n]
        self.cand = (self.cand + [""] * n)[:n]
        self.count = (self.count + [0] * n)[:n]
        for i, t in enumerate(texts):
            if t == self.cand[i]:
                self.count[i] += 1
            else:
                self.cand[i], self.count[i] = t, 1
            if self.count[i] >= self.k or not self.shown[i]:
                self.shown[i] = t
        return self.shown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    model = CRNN()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    torch.set_num_threads(2)  # leave CPU for camera capture and drawing

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}. Try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoother, frozen, show_bin, last, fps = Smoother(), False, False, None, 0.0
    while True:
        if not frozen or last is None:
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
        frame = last.copy()
        H, W = frame.shape[:2]
        # Guide box: central 80% x 50% of the frame.
        x0, y0, x1, y1 = int(W * .1), int(H * .25), int(W * .9), int(H * .75)
        roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

        t0 = time.time()
        boxes, binary = find_lines(roi)
        texts = smoother.update(read_lines(model, binary, boxes))
        fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-3)

        if show_bin:
            frame[y0:y1, x0:x1] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 0), 2)
        for (x, y, w, h), t in zip(boxes, texts):
            cv2.rectangle(frame, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h), (255, 120, 0), 2)
            cv2.putText(frame, t, (x0 + x, y0 + y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
        # Text panel under the guide box with all lines in reading order.
        for i, t in enumerate(texts[:6]):
            cv2.putText(frame, t, (x0, y1 + 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
        status = f"{fps:.0f} fps  lines: {len(boxes)}" + ("  [FROZEN]" if frozen else "")
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Handwriting reader  (q quit, space freeze, b binary)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            frozen = not frozen
        if key == ord("b"):
            show_bin = not show_bin

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
