# ---------- BLOCK 0 : imports & globals ---------------------------------
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn
import torch.amp as amp

CSV_PATH = r"C:/Users/camil/OneDrive/Programming/Smart_Glass/dev/blink_data_20250601_111846.csv"

import os, sys
if not os.path.exists(CSV_PATH):
    sys.exit(f"\nCSV not found → {CSV_PATH}\nCheck the path or move the file.\n")


SEQ_LEN    = 30
BATCH_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = device == "cuda"
print("Device =", device)
# ------------------------------------------------------------------------

# ---------- BLOCK 1 : Dataset & DataLoaders -----------------------------
class BlinkSeqDataset(Dataset):
    """
    eye_seq : (seq_len, 1, 24, 12)  float32  [0‑1]
    num_seq : (seq_len, 7)          float32  (z‑scored)
    label   : int64 0/1
    """
    NUM_COLS = ['ratio_left', 'ratio_right', 'ratio_avg',
                'v_left', 'h_left', 'v_right', 'h_right']

    def __init__(self, csv_path, seq_len=30, train=True,
                 split_ratio=0.8, numeric_stats=None):
        df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
        df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

        print("Columns read from CSV:\n", list(df.columns))   # ← add
        df.columns = df.columns.str.strip()                   # strip hidden spaces


        split = int(len(df) * split_ratio)
        df = df.iloc[:split] if train else df.iloc[split:]

        px_cols = [c for c in df.columns if c.startswith("px_")]
        X_px = (df[px_cols].values.astype(np.float32) / 255.).reshape(-1, 1, 24, 12)

        X_num = df[self.NUM_COLS].values.astype(np.float32)
        if numeric_stats is None:
            mean, std = X_num.mean(0), X_num.std(0) + 1e-6
        else:
            mean, std = numeric_stats
        X_num = (X_num - mean) / std

        self.X_px  = torch.from_numpy(X_px)      # CPU tensors
        self.X_num = torch.from_numpy(X_num)
        self.labels = torch.from_numpy(df["manual_blink"].values.astype(np.int64))
        self.seq_len = seq_len
        self.numeric_stats = (mean, std)        # expose for test set

    def __len__(self):
        return len(self.labels) - self.seq_len + 1

    def __getitem__(self, idx):
        sl = slice(idx, idx + self.seq_len)
        return (self.X_px[sl],               # (seq,1,24,12)
                self.X_num[sl],              # (seq,7)
                self.labels[idx + self.seq_len - 1])

train_ds = BlinkSeqDataset(CSV_PATH, seq_len=SEQ_LEN, train=True)
test_ds  = BlinkSeqDataset(CSV_PATH, seq_len=SEQ_LEN, train=False,
                           numeric_stats=train_ds.numeric_stats)

pin = device == "cuda"
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True,  num_workers=0, pin_memory=pin)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0, pin_memory=pin)
# ------------------------------------------------------------------------

# ---------- BLOCK 2 : Model --------------------------------------------
class BlinkDetector(nn.Module):
    def __init__(self, num_features=7, img_height=24, img_width=12,
                 conv_channels=(16, 32), lstm_hidden=64,
                 lstm_layers=1, bidirectional=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv_channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2)

        conv_out_h = img_height // 4   # 24 -> 6
        conv_out_w = img_width  // 4   # 12 -> 3
        flat_dim   = conv_out_h * conv_out_w * conv_channels[1]

        self.img_fc = nn.Linear(flat_dim, 64)
        self.num_fc = nn.Linear(num_features, 16)

        lstm_in = 64 + 16
        self.lstm = nn.LSTM(lstm_in, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=bidirectional)
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

        self.bidirectional = bidirectional

    def forward(self, eye_seq, num_seq):
        B, T, C, H, W = eye_seq.shape
        x = eye_seq.view(B*T, 1, H, W)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.img_fc(x.view(B*T, -1)))

        n = self.relu(self.num_fc(num_seq.view(B*T, -1)))
        z = torch.cat([x, n], dim=1).view(B, T, -1)

        _, (h, _) = self.lstm(z)
        seq_rep = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.fc(seq_rep).squeeze(1)
# ------------------------------------------------------------------------

# ---------- BLOCK 3 : Training setup -----------------------------------
model = BlinkDetector(num_features=7).to(device)

blink_frac = train_ds.labels.float().mean().item()
pos_weight = torch.tensor([(1 - blink_frac) / blink_frac], device=device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

scaler = amp.GradScaler('cuda')

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    loss_sum, y_pred_all, y_true_all = 0.0, [], []

    for eye, num, lbl in loader:
        eye = eye.to(device, non_blocking=pin).float()
        num = num.to(device, non_blocking=pin).float()
        lbl = lbl.to(device, non_blocking=pin).float()

        with amp.autocast('cuda'):
            logits = model(eye, num)
            loss   = criterion(logits, lbl)

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

        loss_sum += loss.item() * lbl.size(0)
        y_pred_all.append(torch.sigmoid(logits).detach().cpu() > 0.5)
        y_true_all.append(lbl.cpu().bool())

    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return loss_sum / len(loader.dataset), acc, prec, rec, f1
# ------------------------------------------------------------------------

# ---------- BLOCK 4 : Training loop ------------------------------------
best_f1, patience, no_improve = 0.0, 30, 0
for epoch in range(100):
    tr_loss, _, _, _, tr_f1 = run_epoch(train_dl, train=True)
    va_loss, _, _, _, va_f1 = run_epoch(test_dl,  train=False)
    scheduler.step(va_loss)

    print(f"[{epoch+1:02}] trainL {tr_loss:.4f}  F1 {tr_f1:.3f} | "
          f"valL {va_loss:.4f}  F1 {va_f1:.3f}")

    if va_f1 > best_f1 + 0.001:
        best_f1 = va_f1
        no_improve = 0
        torch.save(model.state_dict(), "blink_best.pth")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stop (no F1 improvement)")
            break
# ------------------------------------------------------------------------

# ---------- BLOCK 5 : Inference helper ---------------------------------
model.load_state_dict(torch.load("blink_best.pth", map_location=device))
model.eval()

def predict_sequence(eye_seq_np, num_seq_np):
    """
    eye_seq_np : (30,1,24,12) numpy float32  [0‑1]
    num_seq_np : (30,7)        numpy float32  (z‑scored using train mean/std)
    """
    eye = torch.from_numpy(eye_seq_np).unsqueeze(0).to(device)
    num = torch.from_numpy(num_seq_np).unsqueeze(0).to(device)
    with torch.no_grad(), amp.autocast(enabled=(device == "cuda")):
        prob = torch.sigmoid(model(eye, num)).item()
    return prob
# ------------------------------------------------------------------------
