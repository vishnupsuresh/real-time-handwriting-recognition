"""CRNN: convolutional feature extractor + bidirectional LSTM + CTC head.

Pipeline for an input line image of shape (B, 1, 32, W):
  1. CNN turns pixels into a sequence of W/4 feature columns, each describing
     a vertical slice of the line. Height is pooled to 1, width only to W/4,
     so every character keeps a few time steps.
  2. BiLSTM reads the columns left-to-right and right-to-left. Context matters
     most for cursive, where the shape of a join depends on both neighbors.
  3. A linear layer gives per-column character probabilities; CTC loss aligns
     them with the label without needing per-character positions.

Weakness: LSTMs process steps sequentially, so they are slow on long lines and
cap throughput. Fix: replace them with a small Transformer encoder or 1D
convolutions; or move to a TrOCR-style vision transformer when a GPU and real
data are available (higher accuracy, about 100x the parameters).
"""
import torch
import torch.nn as nn

from .charset import NUM_CLASSES


def conv_bn(cin, cout, k=3, p=1):
    # BatchNorm stabilizes training on the heavily augmented synthetic data.
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=p, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    DOWNSAMPLE = 4  # total width reduction of the CNN

    def __init__(self, hidden: int = 128, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            conv_bn(1, 32), nn.MaxPool2d(2, 2),                        # 16 x W/2
            conv_bn(32, 64), nn.MaxPool2d(2, 2),                       # 8  x W/4
            conv_bn(64, 128), conv_bn(128, 128), nn.MaxPool2d((2, 1)),  # 4  x W/4
            conv_bn(128, 256), conv_bn(256, 256), nn.MaxPool2d((2, 1)),  # 2 x W/4
            conv_bn(256, 256, k=(2, 1), p=0),                          # 1  x W/4
            nn.Dropout2d(0.2),
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.25)
        self.head = nn.Linear(2 * hidden, num_classes)

    def forward(self, x):
        f = self.cnn(x)                    # (B, C, 1, T)
        f = f.squeeze(2).permute(0, 2, 1)  # (B, T, C)
        f, _ = self.rnn(f)
        return self.head(f)                # (B, T, classes), raw logits

    @classmethod
    def out_lengths(cls, widths: torch.Tensor) -> torch.Tensor:
        # MaxPool with stride 2 floors odd sizes, matching integer division.
        return widths // cls.DOWNSAMPLE
