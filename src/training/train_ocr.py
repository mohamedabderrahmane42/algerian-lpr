"""
train_ocr.py  –  Algerian License Plate CRNN Trainer (with Augmentation)
======================================================================
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import RAW_PLATES, OCR_MODEL, NUM_CLASSES, BLANK_IDX, CHARSET, IMG_H, IMG_W
from src.models.crnn import CRNN
from src.data.ocr_support import PlateDataset, crnn_collate_fn, load_compatible_crnn_weights
from src.utils.formatters import decode_crnn

BATCH_SIZE = 32
NUM_EPOCHS = 60
LR         = 1e-3

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, _, _, raw_labels in loader:
            images   = images.to(device)
            preds_np = model(images).cpu().numpy()   # (T, B, C)
            T, B, C  = preds_np.shape
            for b in range(B):
                pred_str = decode_crnn(preds_np[:, b, :])
                true_str = raw_labels[b]
                if pred_str == true_str:
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0.0

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    train_dir = Path(RAW_PLATES) / 'train'
    val_dir   = Path(RAW_PLATES) / 'validation'

    print("Loading datasets …")
    train_ds = PlateDataset(train_dir, augment=True, use_wavelet=True)
    val_ds   = PlateDataset(val_dir, augment=False, use_wavelet=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=crnn_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=crnn_collate_fn, num_workers=0)

    model     = CRNN(NUM_CLASSES).to(device)
    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5, min_lr=1e-5)

    if OCR_MODEL.exists():
        print(f"Loading existing weights from {OCR_MODEL} to fine-tune...")
        skipped = load_compatible_crnn_weights(model, OCR_MODEL, device)
        if skipped:
            print(f"  Skipped {len(skipped)} incompatible tensors from the old checkpoint")

    best_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, targets, target_lengths, _ in tqdm(train_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS}"):
            images  = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            preds = model(images)
            T, B, C = preds.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            loss = ctc_loss(preds, targets, input_lengths, target_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
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
                'img_height'  : IMG_H,
                'img_width'   : IMG_W,
            }, OCR_MODEL)
            print(f"  >> Best model saved  (acc={val_acc:.3f})")

    print(f"\nDone!  Best val accuracy = {best_acc:.3f}")

if __name__ == '__main__':
    train()
