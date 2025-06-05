import os, sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from model import BlinkDetectorXS as BlinkDetector
import constants


CSV_PATH   = constants.Training_Constnats.CSV_PATH
SEQ_LEN    = constants.Training_Constnats.SEQUENCE_LENGTH
BATCH_SIZE = constants.Training_Constnats.BATCH_SIZE
CURR_BEST_F1 = constants.Training_Constnats.CURRENT_BEST_F1

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = device == "cuda"
print("Device =", device)

if not os.path.exists(CSV_PATH):
    sys.exit(f"\nCSV not found → {CSV_PATH}\nCheck the path or move the file.\n")

class BlinkSeqDataset(Dataset):

    NUM_COLS = [
        "ratio_left", "ratio_right", "ratio_avg",
        "v_left", "h_left", "v_right", "h_right",
    ]

    def __init__(self, csv_path: str, seq_len: int = 50, *, train: bool = True,
                 split_ratio: float = 0.6, numeric_stats=None):
        # ── read once & clean ────────────────────────────────────────────
        df = pd.read_csv(csv_path, low_memory=False).dropna().reset_index(drop=True)
        # Strip accidental whitespace in headers (common source of KeyErrors)
        df.columns = df.columns.str.strip()

        # ── train / val split ────────────────────────────────────────────
        split_idx = int(len(df) * split_ratio)
        df = df.iloc[:split_idx] if train else df.iloc[split_idx:]

        # ── image patch pixels ───────────────────────────────────────────
        px_cols = [c for c in df.columns if c.startswith("px_")]
        X_px = (df[px_cols].values.astype(np.float32) / 255.0).reshape(-1, 1, 24, 12)

        # ── numeric features (EAR etc.) ──────────────────────────────────
        X_num = df[self.NUM_COLS].values.astype(np.float32)
        if numeric_stats is None:  # compute stats from *training* split only
            mean, std = X_num.mean(axis=0), X_num.std(axis=0) + 1e-6
        else:
            mean, std = numeric_stats
        X_num = (X_num - mean) / std

        # ── tensors & bookkeeping ───────────────────────────────────────
        self.X_px      = torch.from_numpy(X_px)
        self.X_num     = torch.from_numpy(X_num)
        self.labels    = torch.from_numpy(df["manual_blink"].values.astype(np.int64))
        self.seq_len   = seq_len
        self.stats     = (mean, std)  # expose for val/test

    # ————————————————————————————————————————————
    def __len__(self):
        return len(self.labels) - self.seq_len + 1

    def __getitem__(self, idx):
        sl = slice(idx, idx + self.seq_len)
        eye_seq = self.X_px[sl]    # (seq,1,24,12)
        num_seq = self.X_num[sl]   # (seq,7)
        label   = self.labels[idx + self.seq_len - 1]
        return eye_seq, num_seq, label

# ── build datasets & loaders ─────────────────────────────────────────────
train_ds = BlinkSeqDataset(CSV_PATH, seq_len=SEQ_LEN, train=True)
val_ds   = BlinkSeqDataset(CSV_PATH, seq_len=SEQ_LEN, train=False,
                           numeric_stats=train_ds.stats)

pin = device == "cuda"
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=0, pin_memory=pin)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=pin)

# -------------------------------------------------------------------------
# ---------- BLOCK 3 : Training setup ------------------------------------
model = BlinkDetector().to(device)

# ░ Address class‑imbalance with pos_weight ░
blink_frac = train_ds.labels.float().mean().item()
pos_weight = torch.tensor([(1.0 - blink_frac) / blink_frac], device=device)
criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
lr_sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.5, patience=3)

scaler = GradScaler(enabled=(device == "cuda"))

# -------------------------------------------------------------------------
# ---------- BLOCK 4 : Train / Val epoch ----------------------------------
def run_epoch(loader, *, training: bool):
    model.train() if training else model.eval()
    loss_sum, y_pred_all, y_true_all = 0.0, [], []

    for eye, num, lbl in loader:
        eye = eye.to(device, non_blocking=pin).float()
        num = num.to(device, non_blocking=pin).float()
        lbl = lbl.to(device, non_blocking=pin).float()

        with autocast(device_type = device):
            logits = model(eye, num)
            loss   = criterion(logits, lbl)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

        loss_sum += loss.item() * lbl.size(0)
        y_pred_all.append((torch.sigmoid(logits) > 0.5).detach().cpu())
        y_true_all.append(lbl.cpu().bool())

    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return loss_sum / len(loader.dataset), acc, prec, rec, f1

# -------------------------------------------------------------------------
# ---------- BLOCK 5 : Training loop -------------------------------------
best_f1, patience, no_improve = 0.0, 30, 0
for epoch in range(1, 101):
    tr_loss, _, _, _, tr_f1 = run_epoch(train_dl, training=True)
    va_loss, _, _, _, va_f1 = run_epoch(val_dl,   training=False)
    lr_sched.step(va_loss)

    print(f"[Epoch {epoch:03}] trainL {tr_loss:.4f} F1 {tr_f1:.3f} | "
          f"valL {va_loss:.4f} F1 {va_f1:.3f}")

    if va_f1 > best_f1 + 1e-3 and va_f1 > CURR_BEST_F1:
        best_f1 = va_f1
        no_improve = 0
        torch.save(model.state_dict(), "blink_best.pth")
        np.savez("blink_stats.npz", mean=train_ds.stats[0], std=train_ds.stats[1])
        print("✓ Saved new best model & stats (F1 ↑)")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stop – no F1 improvement for", patience, "epochs")
            break

# -------------------------------------------------------------------------
# ---------- BLOCK 6 : Inference helper ----------------------------------
# Load best weights (helpful after an early stop during interactive runs)
if os.path.exists("blink_best.pth"):
    model.load_state_dict(torch.load("blink_best.pth", map_location=device))
    model.eval()


def predict_sequence(eye_seq_np: np.ndarray, num_seq_np: np.ndarray) -> float:
    """Return blink probability for a *single* sequence (no batching).

    Parameters
    ----------
    eye_seq_np : ndarray (T,1,24,12) float32 in [0,1]
    num_seq_np : ndarray (T,7)        float32 *already z‑scored*
    """
    assert eye_seq_np.shape[0] == num_seq_np.shape[0] == SEQ_LEN, "wrong seq len"

    eye = torch.from_numpy(eye_seq_np).unsqueeze(0).to(device)
    num = torch.from_numpy(num_seq_np).unsqueeze(0).to(device)

    with torch.no_grad(), autocast(device_type = device):
        prob = torch.sigmoid(model(eye, num)).item()
    return prob

# -------------------------------------------------------------------------
