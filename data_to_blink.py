import torch
from torch.utils.data import Dataset
import pandas as pd

class BlinkDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # Drop rows with missing values (optional)
        df = df.dropna()
        # Features: ratio, ratio_avg, distance_vertical, distance_horizontal
        self.X = df[['ratio', 'ratio_avg', 'distance_vertical', 'distance_horizontal']].values.astype('float32')
        # Labels: manual_blink (0 or 1)
        self.y = df['manual_blink'].values.astype('int64')
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])