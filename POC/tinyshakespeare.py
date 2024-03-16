import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Data processing and loading
class ShakespeareDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index:index+self.sequence_length], dtype=torch.long),
            torch.tensor(self.data[index+1:index+self.sequence_length+1], dtype=torch.long)
        )

def load_data(file_path, sequence_length=100):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = list(set(text))
    vocab_size = len(chars)
    char_to_index = {ch: idx for idx, ch in enumerate(chars)}
    encoded_text = [char_to_index[ch] for ch in text]

    dataset = ShakespeareDataset(encoded_text, sequence_length)
    return DataLoader(dataset, batch_size=64, shuffle=True), vocab_size

# Mapping and packing functions
def map_weights_to_2bit(quantized_weights):
    mapped_weights = quantized_weights + 1  # Mapping: -1 -> 0, 0 -> 1, 1 -> 2
    return mapped_weights.to(torch.uint8)

def pack_weights(mapped_weights):
    # Initialize an empty list to hold the packed rows
    packed_rows = []

    for row in mapped_weights:
        # Calculate the number of full groups and the remainder for the current row
        full_groups = row.numel() // 4
        remainder = row.numel() % 4

        # Initialize the packed array for the current row
        packed_row = torch.zeros(full_groups + (1 if remainder > 0 else 0), dtype=torch.uint8, device=row.device)

        # Pack every four 2-bit weights into one byte for the current row
        for i in range(full_groups):
            idx = i * 4
            packed_value = (row[idx] << 6) | (row[idx + 1] << 4) | (row[idx + 2] << 2) | row[idx + 3]
            packed_row[i] = packed_value

        # Handle the remainder for the current row
        if remainder:
            last_value = 0
            for j in range(remainder):
                last_value |= row[full_groups * 4 + j] << ((3 - j) * 2)
            packed_row[-1] = last_value

        # Append the packed row to the list
        packed_rows.append(packed_row)

    # Convert the list of packed rows into a 2D tensor
    packed_weights = torch.stack(packed_rows)

    return packed_weights

def unpack_weights(packed_weights, row_length):
    # Initialize an empty list to hold the unpacked rows
    unpacked_rows = []

    # The number of 2-bit values that each row of the unpacked tensor will contain
    total_elements_per_row = row_length * 4

    for packed_row in packed_weights:
        # Initialize the unpacked row tensor
        unpacked_row = torch.zeros(total_elements_per_row, dtype=torch.int8, device=packed_row.device)

        # Iterate over each packed byte to unpack it into four 2-bit values
        for i, packed_byte in enumerate(packed_row):
            for bit_idx in range(4):
                # Extract the 2-bit value from the packed byte
                shift = (3 - bit_idx) * 2
                value = (packed_byte >> shift) & 0b11
                # Adjust the extracted value to match the original ternary quantization range
                adjusted_value = value - 1  # Assuming the original range was [-1, 0, 1]
                unpacked_row[i * 4 + bit_idx] = adjusted_value

        unpacked_rows.append(unpacked_row)

    # Convert the list of unpacked rows into a 2D tensor
    unpacked_weights = torch.stack(unpacked_rows)

    return unpacked_weights


# Transformer model with custom QuantizedLinear layer
class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        gamma = torch.mean(torch.abs(self.weight)) + 1e-5
        quantized_weight = torch.round(self.weight / gamma).clamp(-1, 1)
        quantized_weight = quantized_weight.to(device)  # Ensure weight is on the correct device
        packed_weights = pack_weights(map_weights_to_2bit(quantized_weight))
        unpacked_quantized_weights = unpack_weights(packed_weights, quantized_weight.numel())
        unpacked_quantized_weights = unpacked_quantized_weights.to(
            device)  # Ensure unpacked weights are on the correct device
        return ternary_matrix_operation(x, unpacked_quantized_weights) + self.bias


def ternary_matrix_operation(input, weights):
    # Initialize the output tensor
    output = torch.zeros(input.size(0), weights.size(1), device=input.device)

    # Iterate over each column of the weight matrix
    for j in range(weights.size(1)):
        for i in range(weights.size(0)):
            # For each weight, apply the ternary operation
            if weights[i, j] == 1:
                output[:, j] += input[:, i]  # Equivalent to adding the input column if weight is 1
            elif weights[i, j] == -1:
                output[:, j] -= input[:, i]  # Equivalent to subtracting the input column if weight is -1
            # No operation if weights[i, j] is 0, as it contributes nothing to the output

    return output


# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_layers=3, heads=8, forward_expansion=4):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([QuantizedLinear(embed_size, embed_size) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)

# Training and inference
def train_model(data_loader, model, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        for batch_idx, (input, target) in enumerate(data_loader):
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output.view(-1, model.vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item()}")


# Main execution
sequence_length = 100
data_loader, vocab_size = load_data('shakespeare.txt', sequence_length)
model = Transformer(vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using ", device)
train_model(data_loader, model)
