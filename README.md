# Real-Time Handwriting Recognition (print + cursive)

Line-level handwritten text recognition with a CRNN (CNN + BiLSTM) trained with CTC loss, plus a live webcam reader.

## How it works

| Stage | What it does | File |
|---|---|---|
| Synthetic data | Renders random 1 to 8 word English phrases with 19 print and 26 cursive handwriting fonts (Google Fonts, OFL/Apache), with random word gaps, then applies slant, rotation, elastic distortion, stroke-width change, blur, noise and (30%) binarization | `hwr/synth.py`, `hwr/data.py` |
| Preprocessing | Crop to ink, resize to height 32 (aspect kept), invert so ink = 1 | `hwr/data.py` |
| Model | 7 conv layers reduce a 32 × W image to W/4 feature columns; 2-layer BiLSTM; linear layer over 80 classes (79 chars + blank). 2.07M params | `hwr/model.py` |
| Loss | CTC aligns per-column predictions with the label, so no per-character boxes are needed | `train.py` |
| Decoding | Greedy CTC (default), or prefix beam search with a character bigram LM (`--beam`) | `hwr/charset.py`, `hwr/decode.py` |
| Live reader | Adaptive threshold, line detection, same-line box merging, gap compression, batched inference, frame smoothing | `live.py` |

Output length T = ⌊W / 4⌋. CTC needs T ≥ |label| + (number of repeated adjacent characters); `collate` stretches any image that is too narrow.

Validation uses 7 **held-out fonts** (3 print, 4 cursive) to measure generalization to unseen styles:

- CER = edit distance(prediction, reference) / |reference|
- WER = word-level edit distance / number of reference words

## Results (v0.3, held-out fonts, greedy decoding)

Validation: 2,000 synthetic lines (1 to 8 words) in 7 fonts never seen in training.

| | Print | Cursive |
|---|---|---|
| CER | 4.2% | 8.9% |
| WER | 20.1% | 37.0% |

Compared with the previous model on the same set, CER dropped 42% (print) and 35% (cursive) after training on longer lines, more capitals, varied word spacing and binarized images.

## Usage

```bash
pip install -r requirements.txt
bash scripts/download_assets.sh   # fonts + word list (not stored in git)

# download the trained model
mkdir -p checkpoints
curl -L -o checkpoints/best.pt https://github.com/vishnupsuresh/real-time-handwriting-recognition/releases/latest/download/best.pt

# train on synthetic data (Apple Silicon GPU or CUDA used automatically)
python train.py --epochs 10

# transcribe single-line images
python predict.py checkpoints/best.pt line.png [--beam]

# tests
pip install -r requirements-dev.txt
pytest -q
```

### Live webcam reader

```bash
python live.py              # --camera 1 for an external webcam, --beam for beam search
```

Hold handwriting inside the green box. Each line is detected, read and drawn on screen (~30 ms per frame on a 2-core CPU with greedy decoding; beam search adds ~30 ms per line). Keys: `q` quit, `space` freeze, `b` show the binarized view the model sees.

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
| Character bigram LM barely helps (sees one previous letter only) | Word-level or 5-gram LM with pyctcdecode + KenLM |
| Line tracking in `live.py` matches lines by order, so texts jump when a line appears or disappears | Match boxes across frames by overlap (IoU) |
| Touching or strongly tilted lines merge or split | Deskew the ROI first, or use a learned line detector (Kraken, docTR) |
| Height 32 loses detail in small cursive loops | Height 64 and one more pooling layer, trained on GPU |
| IAM random split mixes writers across train/val | Use the official writer-independent split |
| LSTM is sequential and slow on long lines | Transformer encoder, or TrOCR for best accuracy |
