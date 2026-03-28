"""
train_ocr.py  –  Algerian License Plate CRNN Trainer
=====================================================
Trains a CNN + BiLSTM + CTC model on the recognition sub-dataset.
Labels are parsed directly from the image filenames:
  e.g.  00123411916.jpg  →  label digits = "00123411916"

Architecture
------------
  Input  :  (B, 1, 32, 128)      grayscale, normalised
  CNN    :  5 conv-blocks → (B, 512, 1, 32)
  Reshape:  (32, B, 512)
  RNN    :  2 × Bidirectional LSTM → (32, B, num_classes)
  Loss   :  CTCLoss

Usage
-----
  python train_ocr.py
"""

import os
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RECOGNITION_DIR = r"c:\Users\Mohamed\Desktop\projects\one\License_Plates_of_Algeria_Dataset-master\License_Plates_of_Algeria_Dataset-master\recognition"
MODEL_SAVE_PATH = r"c:\Users\Mohamed\Desktop\projects\one\ocr_model.pt"

IMG_HEIGHT   = 32
IMG_WIDTH    = 128
BATCH_SIZE   = 32
NUM_EPOCHS   = 60
LR           = 1e-3
MIN_DIGITS   = 8   # ignore filenames with fewer digits (corrupt/missing labels)

CHARSET      = '0123456789'           # digits only
BLANK_IDX    = len(CHARSET)           # index 10 = CTC blank
NUM_CLASSES  = len(CHARSET) + 1       # 11


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def parse_label(filename: str) -> str:
    """Extract the digit label from a recognition-dataset filename."""
    base = os.path.splitext(filename)[0]      # drop .jpg
    base = re.sub(r'_\d+$', '', base)         # drop _1, _2 … suffixes
    base = re.sub(r'\s*\(\d+\)\s*$', '', base)  # drop ' (2)' suffixes
    digits = re.sub(r'[^0-9]', '', base)      # keep only digits
    return digits


def preprocess_plate(img: np.ndarray) -> np.ndarray:
    """
    Preprocess a grayscale plate crop for CRNN input.
    Returns float32 array of shape (1, IMG_HEIGHT, IMG_WIDTH).
    """
    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img   = clahe.apply(img)
    # Resize
    img   = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    # Normalise [0, 1]
    img   = img.astype(np.float32) / 255.0
    # Add channel dim
    img   = img[np.newaxis, :]          # (1, H, W)
    return img


def ctc_greedy_decode(log_probs: np.ndarray, blank: int) -> str:
    """
    Greedy best-path decoding.
    log_probs : (T, num_classes)  numpy array
    """
    indices = log_probs.argmax(axis=1)  # (T,)
    decoded, prev = [], -1
    for idx in indices:
        if idx != prev and idx != blank:
            decoded.append(int(idx))
        prev = idx
    return ''.join(CHARSET[i] for i in decoded if i < len(CHARSET))


def format_algerian(digits: str) -> str:
    """Format raw digit string into NNNNN WWW WW display form."""
    d = re.sub(r'[^0-9]', '', digits)
    if len(d) == 10:
        return f"{d[:5]} {d[5:8]} {d[8:10]}"
    if len(d) == 9:
        return f"{d[:5]} {d[5:7]} {d[7:9]}"
    return d  # unknown length – return as-is


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class PlateDataset(Dataset):
    def __init__(self, root_dir: str):
        self.samples = []
        extensions   = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        for fname in os.listdir(root_dir):
            if not fname.lower().endswith(extensions):
                continue
            digits = parse_label(fname)
            if len(digits) >= MIN_DIGITS:
                self.samples.append((os.path.join(root_dir, fname), digits))
        if not self.samples:
            raise RuntimeError(f"No valid samples found in: {root_dir}")
        print(f"  Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        img = preprocess_plate(img)
        label_ids = [CHARSET.index(c) for c in label if c in CHARSET]
        return torch.FloatTensor(img), label_ids, label


def collate_fn(batch):
    images, labels_list, raw_labels = zip(*batch)
    images         = torch.stack(images, 0)   # (B, 1, H, W)
    target_lengths = torch.tensor([len(l) for l in labels_list], dtype=torch.long)
    targets        = torch.tensor([c for lbl in labels_list for c in lbl], dtype=torch.long)
    return images, targets, target_lengths, raw_labels


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
class BidirectionalLSTM(nn.Module):
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
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, num_classes),
        )

    def forward(self, x):
        feat = self.cnn(x)                  # (B, 512, 1, 32)
        B, C, H, W = feat.shape
        assert H == 1, f"CNN height should be 1, got {H}"
        feat = feat.squeeze(2)              # (B, 512, 32)
        feat = feat.permute(2, 0, 1)        # (32, B, 512)
        out  = self.rnn(feat)               # (32, B, num_classes)
        return torch.log_softmax(out, dim=2)


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, _, _, raw_labels in loader:
            images   = images.to(device)
            preds_np = model(images).cpu().numpy()   # (T, B, C)
            T, B, C  = preds_np.shape
            for b in range(B):
                pred_str = ctc_greedy_decode(preds_np[:, b, :], BLANK_IDX)
                true_str = raw_labels[b]
                if pred_str == true_str:
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0.0


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    train_dir = os.path.join(RECOGNITION_DIR, 'train')
    val_dir   = os.path.join(RECOGNITION_DIR, 'validation')

    print("Loading datasets …")
    train_ds = PlateDataset(train_dir)
    val_ds   = PlateDataset(val_dir)

    # num_workers=0 avoids Windows multiprocessing issues
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model     = CRNN(NUM_CLASSES).to(device)
    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6,
                                                     factor=0.5, min_lr=1e-5)

    best_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for images, targets, target_lengths, _ in tqdm(train_loader,
                                                       desc=f"Epoch {epoch:3d}/{NUM_EPOCHS}"):
            images  = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)            # (T, B, C)
            T, B, C = preds.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            loss = ctc_loss(preds, targets, input_lengths, target_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validate ──
        val_acc = evaluate(model, val_loader, device)
        scheduler.step(1 - val_acc)

        print(f"  loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes' : NUM_CLASSES,
                'charset'     : CHARSET,
                'blank_idx'   : BLANK_IDX,
                'img_height'  : IMG_HEIGHT,
                'img_width'   : IMG_WIDTH,
            }, MODEL_SAVE_PATH)
            print(f"  >> Best model saved  (acc={val_acc:.3f})")

    print(f"\nDone!  Best val accuracy = {best_acc:.3f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train()
