import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import EyeBlinkNet, EyeConvNet, EyeConvLSTMNet
from data_preparation import EyeSeqDataset
import constants

CSV_PATH = constants.Training_Constnats.IMG_CSV_PATH
SEQ_LEN = constants.Training_Constnats.SEQUENCE_LENGTH

BATCH_MIN = 2
BATCH_MAX = 16
BATCH_STEP = 2
LR_MIN = 1e-4
LR_MAX = 1e-3
LR_STEP = 2e-4

if not os.path.exists(CSV_PATH):
    sys.exit(f"CSV not found → {CSV_PATH}")

train_ds = EyeSeqDataset(CSV_PATH, seq_len=SEQ_LEN, train=True)
val_ds   = EyeSeqDataset(CSV_PATH, seq_len=SEQ_LEN, train=False, img_stats=train_ds.stats)

device = "cuda" if torch.cuda.is_available() else "cpu"
pin = device == "cuda"


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    loss_sum, y_pred_all, y_true_all = 0.0, [], []
    for img, lbl in loader:
        img = img.to(device, non_blocking=pin).float()
        lbl = lbl.to(device, non_blocking=pin).float()
        logits = model(img)
        loss = criterion(logits, lbl)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        loss_sum += loss.item() * lbl.size(0)
        y_pred_all.append((torch.sigmoid(logits) > 0.5).detach().cpu())
        y_true_all.append(lbl.cpu().bool())
    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return loss_sum / len(loader.dataset), acc, prec, rec, f1


best_global_f1 = 0.0
best_state = None
best_desc = ""

for bs in range(BATCH_MIN, BATCH_MAX + 1, BATCH_STEP):
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin)
    lr = LR_MIN
    while lr <= LR_MAX + 1e-9:
        for Net in [EyeBlinkNet, EyeConvNet, EyeConvLSTMNet]:
            model = Net(input_size=train_ds.n_pixels) if Net is EyeBlinkNet else Net()
            model.to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            best_f1, no_imp = 0.0, 0
            for epoch in range(1, 501):
                tr_loss, _, _, _, tr_f1 = run_epoch(model, train_dl, criterion, optimizer)
                va_loss, _, _, _, va_f1 = run_epoch(model, val_dl, criterion)
                print(f"{Net.__name__} bs{bs} lr{lr:.4f} ep{epoch} valF1 {va_f1:.3f}")
                if va_f1 > best_f1 + 1e-3:
                    best_f1 = va_f1
                    no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= 30:
                        break
            if best_f1 > best_global_f1:
                best_global_f1 = best_f1
                best_state = model.state_dict()
                best_desc = f"{Net.__name__}_bs{bs}_lr{lr:.4f}"
        lr += LR_STEP

if best_state is not None:
    os.makedirs(constants.Paths.MODEL_DIR, exist_ok=True)
    torch.save(best_state, constants.Paths.IMG_WEIGHTS)
    np.savez(constants.Paths.IMG_STATS_NPZ, mean=train_ds.stats[0], std=train_ds.stats[1])
    print("Best model saved:", best_desc, "F1", best_global_f1)
else:
    print("No model trained")
