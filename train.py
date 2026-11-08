"""Train the CRNN with CTC loss.

Examples:
  python train.py --epochs 8                       # synthetic print + cursive
  python train.py --iam /data/iam --resume checkpoints/best.pt --lr 3e-4
                                                   # fine-tune on real IAM lines

Metrics:
  CER = character edit distance / reference length
  WER = word edit distance / reference word count
Validation uses held-out fonts, reported separately for print and cursive.
"""
import argparse
import random
import time
from pathlib import Path

import editdistance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from hwr.charset import BLANK, greedy_decode
from hwr.data import IAMLines, SynthLines, collate
from hwr.model import CRNN


def evaluate(model, loader, device):
    model.eval()
    stats = {}  # style -> [char_errs, chars, word_errs, words]
    examples = []
    with torch.no_grad():
        for x, _, _, widths, texts, styles in loader:
            logits = model(x.to(device))
            preds = logits.argmax(-1).cpu()
            lens = CRNN.out_lengths(widths)
            for p, L, ref, st in zip(preds, lens, texts, styles):
                hyp = greedy_decode(p[:L])
                s = stats.setdefault(st, [0, 0, 0, 0])
                s[0] += editdistance.eval(hyp, ref); s[1] += len(ref)
                s[2] += editdistance.eval(hyp.split(), ref.split()); s[3] += len(ref.split())
                if len(examples) < 6:
                    examples.append((st, ref, hyp))
    model.train()
    return {k: (v[0] / max(v[1], 1), v[2] / max(v[3], 1)) for k, v in stats.items()}, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps-per-epoch", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--iam", help="path to IAM root to train on real data")
    ap.add_argument("--resume", help="checkpoint to start from")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0)
    device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
    Path(args.out).mkdir(exist_ok=True)

    n_train = args.steps_per_epoch * args.batch
    train_ds = SynthLines("train", length=n_train)
    val_ds = SynthLines("val", length=600, seed=12345)
    if args.iam:
        # Simple random 90/10 split. Weakness: IAM writers then appear in both
        # sets, which inflates scores. Fix: use the official writer-independent
        # split files and pass their line ids to IAMLines(ids=...).
        full = IAMLines(args.iam)
        ids = list(range(len(full))); random.shuffle(ids)
        cut = int(0.9 * len(ids))
        tr = torch.utils.data.Subset(IAMLines(args.iam, train=True), ids[:cut])
        train_ds = ConcatDataset([tr, SynthLines("train", length=len(tr) // 2)])
        val_ds = torch.utils.data.Subset(full, ids[cut:])

    train_dl = DataLoader(train_ds, args.batch, shuffle=True, collate_fn=collate,
                          num_workers=args.workers, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, 64, collate_fn=collate, num_workers=args.workers)

    model = CRNN().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["model"])
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  device: {device}")

    # zero_infinity: if a label is longer than the output sequence the loss is
    # inf; zeroing it skips that sample instead of crashing training with NaNs.
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # One-cycle schedule: warm up then decay. Converges faster than a constant
    # rate, which matters with a short CPU budget.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.15)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0, run = time.time(), 0.0
        for step, (x, tgt, tlen, widths, _, _) in enumerate(train_dl, 1):
            logits = model(x.to(device))
            # CTCLoss expects (T, B, C) log-probabilities.
            logp = logits.log_softmax(-1).permute(1, 0, 2)
            # CTC loss may not be supported on MPS; compute it on CPU (gradients still flow back to the GPU)
            loss = ctc(logp.cpu(), tgt, CRNN.out_lengths(widths), tlen)
            opt.zero_grad()
            loss.backward()
            # Clipping guards against the occasional exploding LSTM gradient.
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            run += loss.item()
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{len(train_dl)} loss {run / 100:.3f}", flush=True)
                run = 0.0

        metrics, ex = evaluate(model, val_dl, device)
        overall = sum(c for c, _ in metrics.values()) / len(metrics)
        line = "  ".join(f"{k}: CER {c:.3f} WER {w:.3f}" for k, (c, w) in sorted(metrics.items()))
        print(f"== epoch {ep} ({time.time() - t0:.0f}s)  {line}", flush=True)
        for st, ref, hyp in ex:
            print(f"   [{st}] ref: {ref!r}  hyp: {hyp!r}")
        ckpt = {"model": model.state_dict(), "epoch": ep, "metrics": metrics}
        torch.save(ckpt, Path(args.out) / "last.pt")
        if overall < best:
            best = overall
            torch.save(ckpt, Path(args.out) / "best.pt")


if __name__ == "__main__":
    main()
