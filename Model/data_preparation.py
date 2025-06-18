import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import constants


class BlinkSeqDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        seq_len: int = constants.Training_Constnats.SEQUENCE_LENGTH,
        *,
        train: bool = True,
        split_ratio: float = 0.6,
        numeric_stats=None
    ):
        
        data_frame = pd.read_csv(csv_path, low_memory=False).dropna().reset_index(drop=True)
        # Strip accidental whitespace in headers (common source of KeyErrors)
        data_frame.columns = data_frame.columns.str.strip()

        # Some recorded CSVs may lack the newest feature columns. Create them
        # on the fly so old datasets remain compatible with the latest model.
        expected_cols = list(constants.Data_Gathering_Constants.NUM_COLS)
        for col in expected_cols:
            if col not in data_frame:
                data_frame[col] = 0.0
        data_frame = data_frame.reindex(columns=expected_cols + ["blink_count", "manual_blink"])  # keep order

        # ── train / val split ────────────────────────────────────────────
        split_idx = int(len(data_frame) * split_ratio)
        data_frame = data_frame.iloc[:split_idx] if train else data_frame.iloc[split_idx:]

        # ── numeric features (EAR ratios) ───────────────────────────────
        X_num = data_frame[constants.Data_Gathering_Constants.NUM_COLS].values.astype(np.float32)
        if numeric_stats is None:  # compute stats from *training* split only
            mean, std = X_num.mean(axis=0), X_num.std(axis=0) + 1e-6
        else:
            mean, std = numeric_stats
        X_num = (X_num - mean) / std

        # ── tensors & bookkeeping ───────────────────────────────────────
        self.X_num = torch.from_numpy(X_num)
        self.labels = torch.from_numpy(data_frame["manual_blink"].values.astype(np.int64))
        self.seq_len = seq_len
        self.stats = (mean, std)  # expose for val/test

    # ————————————————————————————————————————————
    def __len__(self):
        return len(self.labels) - self.seq_len + 1

    def __getitem__(self, idx):
        sl = slice(idx, idx + self.seq_len)
        num_seq = self.X_num[sl]  # (seq, num_features)
        label = self.labels[idx + self.seq_len - 1]
        return num_seq, label


class EyeSeqDataset(Dataset):
    """Sequence dataset for grayscale eye patches."""

    def __init__(
        self,
        csv_path: str,
        seq_len: int = constants.Training_Constnats.SEQUENCE_LENGTH,
        *,
        train: bool = True,
        split_ratio: float = 0.6,
        img_stats=None,
    ):
        data_frame = pd.read_csv(csv_path, low_memory=False).dropna().reset_index(drop=True)
        data_frame.columns = data_frame.columns.str.strip()

        pixel_cols = [c for c in data_frame.columns if c.startswith("pixel_")]
        split_idx = int(len(data_frame) * split_ratio)
        data_frame = data_frame.iloc[:split_idx] if train else data_frame.iloc[split_idx:]

        X_img = data_frame[pixel_cols].values.astype(np.float32)
        if img_stats is None:
            mean, std = X_img.mean(axis=0), X_img.std(axis=0) + 1e-6
        else:
            mean, std = img_stats
        X_img = (X_img - mean) / std

        self.X_img = torch.from_numpy(X_img)
        self.labels = torch.from_numpy(data_frame["manual_blink"].values.astype(np.int64))
        self.seq_len = seq_len
        self.stats = (mean, std)
        self.n_pixels = len(pixel_cols)

    def __len__(self):
        return len(self.labels) - self.seq_len + 1

    def __getitem__(self, idx):
        sl = slice(idx, idx + self.seq_len)
        img_seq = self.X_img[sl]
        label = self.labels[idx + self.seq_len - 1]
        return img_seq, label
