import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ShakespeareDataset(Dataset):
    def __init__(self, file_path, seq_length):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.text = file.read()
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        seq = self.text[idx:idx + self.seq_length]
        seq_indices = [self.char_to_idx[ch] for ch in seq]
        return torch.tensor(seq_indices)

# Set up the dataset and dataloader
file_path = './POC/tinyshakespeare.py'
seq_length = 100
dataset = ShakespeareDataset(file_path, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)