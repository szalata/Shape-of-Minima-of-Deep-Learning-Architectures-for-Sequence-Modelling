import torch
from torch.utils.data import Dataset
import numpy as np

class SeqDataset(Dataset):
    """Sequence learning dataset."""

    def __init__(self, Xs, Ys):
        """
        Args:
            Xs (string): Path to the npy file with features
            Ys (string): Path to the npy file with targets
        """
        self.featurs = np.load(Xs)
        self.targets = np.load(Ys) 

    def __len__(self):
        return len(self.featurs)

    def __getitem__(self, idx):
        #convert to tensor
        feature = torch.from_numpy(self.featurs[idx])
        target = torch.Tensor([self.targets[idx]])

        return feature,target

    def len_unique(self):
        
        return len(np.unique(self.featurs))


