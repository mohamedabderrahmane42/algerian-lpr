import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, in_size: int, hidden: int, out_size: int):
        super().__init__()
        self.rnn    = nn.LSTM(in_size, hidden, bidirectional=True, batch_first=False)
        self.linear = nn.Linear(hidden * 2, out_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        T, B, H = out.shape
        out = self.linear(out.view(T * B, H))
        return out.view(T, B, -1)


class CRNN(nn.Module):
    """
    CNN + BiLSTM sequence model.
    Input  : (B, 1, 32, 128)
    Output : (T=32, B, num_classes)  log-softmax
    """
    def __init__(self, num_classes: int):
        super().__init__()

        def _block(cin, cout, pool_k=(2, 2), pool_s=(2, 2)):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, 1, 1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_k, pool_s),
            )

        # After each block: (B, C, H, W)
        # Start: (B,  1, 32, 128)
        self.cnn = nn.Sequential(
            _block(  1,  64),              # → (B,  64, 16,  64)
            _block( 64, 128),              # → (B, 128,  8,  32)
            _block(128, 256, (2,1),(2,1)), # → (B, 256,  4,  32)
            _block(256, 256, (2,1),(2,1)), # → (B, 256,  2,  32)
            _block(256, 512, (2,1),(2,1)), # → (B, 512,  1,  32)
        )

        self.rnn = nn.Sequential(
            BiLSTM(512, 256, 256),
            BiLSTM(256, 256, num_classes),
        )

    def forward(self, x):
        feat = self.cnn(x)                  # (B, 512, 1, 32)
        B, C, H, W = feat.shape
        assert H == 1, f"CNN height should be 1, got {H}"
        feat = feat.squeeze(2)              # (B, 512, 32)
        feat = feat.permute(2, 0, 1)        # (32, B, 512)
        out  = self.rnn(feat)               # (32, B, num_classes)
        return torch.log_softmax(out, dim=2)
