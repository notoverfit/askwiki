import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TokenSequence(Dataset):
    def __init__(self, path: str, dtype: type, seq_len: int):
        if not os.path.exists(path) or not path.endswith('.bin'):
            raise FileNotFoundError('TokenSequence: path does not exist or is not a binary (.bin) file.')

        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.dtype = dtype

    def __len__(self):
        """
        returns the length of the data set, measured by number of sequences.
        if the data set is not perfectly divisible by the sequence length, 
        the last partial sequence is deleted. remove one token for the target.
        """
        return (self.data.shape[0] - 1) // self.seq_len

    def __getitem__(self, idx: int):
        if idx > self.__len__() - 1 or idx < 0:
            raise IndexError('TokenSequence: attempt to get index out of range.')

        # get sequence length + 1 of tokens, for shifted target tensor
        start_idx = idx * self.seq_len
        chunk = self.data[start_idx:start_idx+self.seq_len+1].astype(self.dtype)

        # cast to tensors
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y