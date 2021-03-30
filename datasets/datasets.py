from torch.utils.data import Dataset
import numpy as np
import torch

class BinarizedMNIST(Dataset):
    def __init__(self, file):
        self.data = np.load(file)
        self.data = torch.tensor(self.data)

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
