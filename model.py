import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn
import torch.amp as amp

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