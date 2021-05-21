import os

import torch
from torch.utils.data import Dataset
import numpy as np


class SeqDataset(Dataset):
    """Sequence learning dataset."""

    def __init__(self, directory, split):
        self.seq = torch.from_numpy(np.load(os.path.join(directory, f"X_{split}.npy")))[:, :, None].float()
        self.targets = torch.from_numpy(np.load(os.path.join(directory, f"y_{split}.npy"))).float()

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        target = self.targets[idx]

        return seq, target
