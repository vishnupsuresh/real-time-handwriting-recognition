# Handwriting Recognition (print + cursive)

Line-level handwritten text recognition with a CRNN (CNN + BiLSTM) trained with CTC loss.

## How it works

| Stage | What it does | File |
|---|---|---|
| Synthetic data | Renders random English phrases with 20 print and 25 cursive handwriting fonts (Google Fonts, OFL/Apache), then applies slant, rotation, elastic distortion, stroke-width change, blur and noise | `hwr/synth.py` |
| Preprocessing | Crop to ink, resize to height 32 (aspect kept), invert so ink = 1 | `hwr/data.py` |
| Model | 7 conv layers reduce a 32 × W image to W/4 feature columns; 2-layer BiLSTM; linear layer over 80 classes (79 chars + blank). 2.07M params | `hwr/model.py` |
| Loss | CTC aligns per-column predictions with the label, so no per-character boxes are needed | `train.py` |
| Decoding | Greedy CTC: argmax per column, collapse repeats, drop blanks | `hwr/charset.py` |

Output length T = W / 4. CTC requires T ≥ label length + number of repeated adjacent characters, which holds for normal letter widths at height 32.

Validation uses 7 **held-out fonts** (3 print, 4 cursive) to measure generalization to unseen styles, reported separately:

- CER = edit distance(prediction, reference) / len(reference)
- WER = word-level edit distance / number of reference words

## Usage

```bash
pip install -r requirements.txt
bash scripts/download_assets.sh   # fonts + word list (not stored in git)

# train on synthetic data (CPU: ~7 min/epoch; GPU much faster)
python train.py --epochs 10

# transcribe single-line images
python predict.py checkpoints/best.pt line.png
```

### Live webcam reader

```bash
python live.py              # --camera 1 for an external webcam
```

Hold handwriting inside the green box. Each line is detected, read and drawn on screen (~30 ms per frame on a 2-core CPU). Keys: `q` quit, `space` freeze, `b` show the binarized view the model sees.

Tips: fill the box with the text, use dark pen on white paper, avoid glare, keep lines level.

### Fine-tune on real handwriting (IAM)

1. Register and download the IAM Handwriting Database (`lines.tgz`, `ascii.tgz`) from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database.
2. Extract to `/data/iam/lines/...` and `/data/iam/ascii/lines.txt`.
3. `python train.py --iam /data/iam --resume checkpoints/best.pt --lr 3e-4 --epochs 20`

Training then mixes IAM lines with synthetic lines (2:1) so cursive font styles are not forgotten.

## Known weaknesses and fixes

| Weakness | Fix |
|---|---|
| Fonts are not real handwriting; synthetic-only models drop sharply on real scans | Fine-tune on IAM / CVL / your own labeled lines |
| Greedy decoding has no language knowledge | Beam search + n-gram LM (pyctcdecode + KenLM) |
| Line input only; full pages fail | Add line segmentation (projection profiles or Kraken) |
| Height 32 loses detail in small cursive loops | Use height 64 and one more pooling layer on GPU |
| IAM random split mixes writers across train/val | Use the official writer-independent split |
| LSTM is sequential and slow on long lines | Transformer encoder, or TrOCR for best accuracy |
| `predict.py` binarizes inputs, training images are grayscale | Add binarization as a training augmentation |
