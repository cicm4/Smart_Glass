import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from model import BlinkRatioNet as BlinkDetector
from data_preparation import BlinkSeqDataset
import constants


CSV_PATH   = constants.Training_Constants.CSV_PATH
SEQ_LEN    = constants.Training_Constants.SEQUENCE_LENGTH
BATCH_SIZE = constants.Training_Constants.BATCH_SIZE
CURR_BEST_F1 = constants.Training_Constants.CURRENT_BEST_F1

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = device == "cuda"
print("Device =", device)
print(f"Training with {constants.Model_Constants.NUM_FEATURES} features")

if not os.path.exists(CSV_PATH):
    sys.exit(
        f"\nCSV not found → {CSV_PATH}\nCheck the path or move the file.\n"
    )

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
model = BlinkDetector(num_features=constants.Model_Constants.NUM_FEATURES).to(device)

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

    for num, lbl in loader:
        num = num.to(device, non_blocking=pin).float()
        lbl = lbl.to(device, non_blocking=pin).float()

        with autocast(device_type = device):
            logits = model(num)
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
        torch.save(model.state_dict(), constants.Paths.NUM_WEIGHTS)
        np.savez(constants.Paths.NUM_STATS_NPZ, mean=train_ds.stats[0], std=train_ds.stats[1])
        print("✓ Saved new best model & stats (F1 ↑)")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stop – no F1 improvement for", patience, "epochs")
            break

# -------------------------------------------------------------------------
# ---------- BLOCK 6 : Inference helper ----------------------------------
# Load best weights (helpful after an early stop during interactive runs)
if os.path.exists(constants.Paths.NUM_WEIGHTS):
    model.load_state_dict(torch.load(constants.Paths.NUM_WEIGHTS, map_location=device))
    model.eval()


def predict_sequence(num_seq_np: np.ndarray) -> float:
    """Return blink probability for a *single* sequence (no batching).

    Parameters
    ----------
    num_seq_np : ndarray (T,N) float32 *already z‑scored*
    """
    assert num_seq_np.shape[0] == SEQ_LEN, "wrong seq len"

    num = torch.from_numpy(num_seq_np).unsqueeze(0).to(device)

    with torch.inference_mode(), autocast(device_type = device):
        prob = torch.sigmoid(model(num)).item()
    return prob

# -------------------------------------------------------------------------
