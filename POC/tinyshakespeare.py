import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TernaryTensor:
    def __init__(self, tensor):
        self.shape = tensor.shape
        self.packed_tensor = self.pack_tensor(tensor)

    def pack_tensor(self, tensor):
        # Flatten the tensor
        flat_tensor = tensor.view(-1)

        # Convert ternary values to two bits
        bit_values = torch.where(flat_tensor == 1, torch.tensor(0b11),
                                 torch.where(flat_tensor == -1, torch.tensor(0b01),
                                             torch.tensor(0b00)))

        # Pack the bits into bytes
        packed_tensor = torch.bitwise_or(bit_values[0::4], bit_values[1::4] << 2)
        packed_tensor = torch.bitwise_or(packed_tensor, bit_values[2::4] << 4)
        packed_tensor = torch.bitwise_or(packed_tensor, bit_values[3::4] << 6)

        return packed_tensor.type(torch.uint8)

    def unpack_tensor(self):
        # Unpack the bits from bytes
        unpacked_tensor = torch.empty(self.packed_tensor.shape[0] * 4, dtype=torch.int8)
        unpacked_tensor[0::4] = torch.bitwise_and(self.packed_tensor, 0b00000011)
        unpacked_tensor[1::4] = torch.bitwise_and(self.packed_tensor >> 2, 0b00000011)
        unpacked_tensor[2::4] = torch.bitwise_and(self.packed_tensor >> 4, 0b00000011)
        unpacked_tensor[3::4] = torch.bitwise_and(self.packed_tensor >> 6, 0b00000011)

        # Convert two bits to ternary values
        ternary_tensor = torch.where(unpacked_tensor == 0b11, torch.tensor(1),
                                     torch.where(unpacked_tensor == 0b01, torch.tensor(-1),
                                                 torch.tensor(0)))

        return ternary_tensor.view(self.shape)


def ternary_quantize(weights, gamma=1.0):
    # Compute the scaling factor
    eps = 1e-7
    W_bar = torch.mean(torch.abs(weights))
    gamma_prime = gamma / (W_bar + eps)

    # Perform ternary quantization
    quantized_weights = torch.where(weights > 0.5 * gamma_prime, torch.ones_like(weights),
                                    torch.where(weights < -0.5 * gamma_prime, -torch.ones_like(weights),
                                                torch.zeros_like(weights)))

    # Pack the ternary values into two bits
    ternary_tensor = TernaryTensor(quantized_weights)

    return ternary_tensor

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