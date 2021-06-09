import os

import torch
from torch.utils.data import Dataset
import numpy as np

import pickle


class SeqDataset(Dataset):
    """Sequence learning dataset."""

    def __init__(self, directory, split):
        # self.seq = torch.from_numpy(np.load(os.path.join(directory, f"X_{split}.npy")))[:, :, None].float()
        # self.targets = torch.from_numpy(np.load(os.path.join(directory, f"y_{split}.npy")))[:, None].float()

        Xfile = open(os.path.join(directory, f"X_{split}"), 'rb')
        self.seq = pickle.load(Xfile)
        Xfile.close()

        yfile = open(os.path.join(directory, f"y_{split}"), 'rb')
        self.targets = pickle.load(yfile)
        yfile.close()

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        target = self.targets[idx]

        return seq, target, len(seq)

    def collate_fn(self, data):
        
        seqs, targets, lengths = zip(*data)
        max_len = max(lengths)
        batch_size= len(data)
       
        targets = torch.tensor(targets)
        batch = torch.zeros((batch_size, max_len))
        masks = torch.zeros((batch_size, max_len))


        for i in range(batch_size):
            padding = torch.zeros(max_len-lengths[i]) 
            batch[i] = torch.cat((torch.from_numpy(seqs[i]), padding))
            masks[i][lengths[i]:] = 1

        return batch[:, :, None].float(), targets[:, None].float(), masks.bool()
