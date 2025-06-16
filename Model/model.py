import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import constants

class BlinkRatioNet(nn.Module):
    """Small LSTM classifier operating on extended eye metrics.

    The network ingests sequences of numeric features such as EAR ratios,
    vertical distances, and eye width. The feature count is controlled by
    ``Model_Constants.NUM_FEATURES`` so training and inference stay in sync.
    """

    def __init__(
        self,
        num_features: int = constants.Model_Constants.NUM_FEATURES,
        fc_sizes = constants.Model_Constants.RATIO_MODEL_CONSTANTS.FC_SIZES,
        lstm_hidden: int = constants.Model_Constants.RATIO_MODEL_CONSTANTS.LTSM_HIDDEN,
        lstm_layers: int = constants.Model_Constants.RATIO_MODEL_CONSTANTS.LTSM_LAYERS,
        bidirectional: bool = constants.Model_Constants.RATIO_MODEL_CONSTANTS.BIDIRECTIONAL,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = num_features
        for s in fc_sizes:
            self.layers.append(nn.Linear(in_dim, s))
            in_dim = s
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(
            in_dim,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.out_fc = nn.Linear(out_dim, 1)
        self.bidirectional = bidirectional

    def forward(self, num_seq):
        batch, time_steps, _ = num_seq.shape
        x = num_seq.view(batch * time_steps, -1)
        for fc in self.layers:
            x = self.relu(fc(x))
        x = x.view(batch, time_steps, -1)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.out_fc(h).squeeze(1)
