"""Train the CRNN with CTC loss.

Examples:
  python train.py --epochs 10                      # synthetic print + cursive
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
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from hwr.charset import BLANK, ctc_min_len, greedy_decode
from hwr.data import IAMLines, SynthLines, collate
from hwr.decode import beam_decode, default_lm
from hwr.model import CRNN, load_model, model_config


def seed_worker(worker_id):
    """Give each DataLoader worker its own NumPy and Python seed.

    PyTorch seeds its own RNG per worker, but on Linux (fork) every worker
    inherits the same NumPy state, so all workers would add identical noise
    and distortions. Must be a top-level function: macOS starts workers with
    'spawn', which cannot pickle a lambda.
    """
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


def evaluate(model, loader, device, use_beam=False):
    model.eval()
    lm = default_lm() if use_beam else None
    stats = {}  # style -> [char_errs, chars, word_errs, words]
    examples = []
    with torch.no_grad():
        for x, _, _, widths, texts, styles in loader:
            logp = model(x.to(device)).log_softmax(-1).cpu()
            lens = CRNN.out_lengths(widths)
            for lp, L, ref, st in zip(logp, lens, texts, styles):
                hyp = beam_decode(lp[:L].numpy(), lm) if use_beam else greedy_decode(lp[:L].argmax(-1))
                s = stats.setdefault(st, [0, 0, 0, 0])
                s[0] += editdistance.eval(hyp, ref); s[1] += len(ref)
                s[2] += editdistance.eval(hyp.split(), ref.split()); s[3] += len(ref.split())
                if len(examples) < 6:
                    examples.append((st, ref, hyp))
    model.train()
    return {k: (v[0] / max(v[1], 1), v[2] / max(v[3], 1)) for k, v in stats.items()}, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--iam", help="path to IAM root to train on real data")
    ap.add_argument("--resume", help="checkpoint to start from")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-size", type=int, default=2000,
                    help="2000 keeps run-to-run noise in CER under ~0.5%%; 600 was ±1%%")
    ap.add_argument("--beam-eval", action="store_true",
                    help="also report beam search + LM scores after the last epoch")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = pick_device()
    Path(args.out).mkdir(exist_ok=True)

    n_train = args.steps_per_epoch * args.batch
    train_ds = SynthLines("train", length=n_train)
    val_ds = SynthLines("val", length=args.val_size, seed=12345)
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

    loader_kw = dict(collate_fn=collate, num_workers=args.workers, worker_init_fn=seed_worker)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True,
                          persistent_workers=args.workers > 0, **loader_kw)
    val_dl = DataLoader(val_ds, 64, **loader_kw)

    model = load_model(args.resume, device).train() if args.resume else CRNN().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  device: {device}")

    # zero_infinity: if a label needs more frames than the output has, the
    # loss is inf; zeroing it skips the sample instead of producing NaNs.
    # Skipped samples are counted below so the problem is visible.
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # One-cycle schedule: warm up then decay. Converges faster than a constant
    # rate, which matters with a short training budget.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.15)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0, run, skipped = time.time(), 0.0, 0
        for step, (x, tgt, tlen, widths, texts, _) in enumerate(train_dl, 1):
            out_lens = CRNN.out_lengths(widths)
            skipped += sum(int(L) < ctc_min_len(t) for L, t in zip(out_lens, texts))
            logits = model(x.to(device))
            # CTCLoss expects (T, B, C) log-probabilities.
            logp = logits.log_softmax(-1).permute(1, 0, 2)
            if device == "mps":
                # CTC loss is not implemented on MPS in all PyTorch versions;
                # computing it on CPU still sends gradients back to the GPU.
                logp = logp.cpu()
            loss = ctc(logp, tgt, out_lens, tlen)
            opt.zero_grad()
            loss.backward()
            # Clipping guards against the occasional exploding LSTM gradient.
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            run += loss.item()
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{len(train_dl)} loss {run / 100:.3f}", flush=True)
                run = 0.0
        if skipped:
            print(f"   warning: {skipped} samples skipped (label longer than image allows)")

        metrics, ex = evaluate(model, val_dl, device)
        overall = sum(c for c, _ in metrics.values()) / len(metrics)
        line = "  ".join(f"{k}: CER {c:.3f} WER {w:.3f}" for k, (c, w) in sorted(metrics.items()))
        print(f"== epoch {ep} ({time.time() - t0:.0f}s)  {line}", flush=True)
        for st, ref, hyp in ex:
            print(f"   [{st}] ref: {ref!r}  hyp: {hyp!r}")
        ckpt = {"model": model.state_dict(), "epoch": ep, "metrics": metrics,
                "config": model_config()}
        torch.save(ckpt, Path(args.out) / "last.pt")
        if overall < best:
            best = overall
            torch.save(ckpt, Path(args.out) / "best.pt")

    if args.beam_eval:
        metrics, _ = evaluate(model, val_dl, device, use_beam=True)
        line = "  ".join(f"{k}: CER {c:.3f} WER {w:.3f}" for k, (c, w) in sorted(metrics.items()))
        print(f"== beam search + LM  {line}")


if __name__ == "__main__":
    main()
