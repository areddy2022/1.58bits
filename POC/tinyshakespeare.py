import torch
import torch.nn as nn
import torch.optim as optim

# Ternary quantization functions
def quantization_function(weights, gamma, epsilon):
    return torch.clamp(torch.round(weights / (gamma + epsilon)), -1, 1)


def round_clip(x, a, b):
    return torch.max(a, torch.min(b, torch.round(x)))


def gamma(weights):
    return torch.mean(torch.abs(weights))

# Ternary quantized linear layer
class TernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.quantized_weight = None

    def forward(self, x):
        if self.quantized_weight is None:
            gamma_val = gamma(self.weight)
            ternary_weights = quantization_function(self.weight, gamma_val, 1e-6)
            self.quantized_weight = self.pack_weights(ternary_weights)
        return self.ternary_linear(x, self.quantized_weight, self.bias)

    def pack_weights(self, ternary_weights):
        # Calculate the number of packed values needed: each int8 can store 4 2-bit values
        packed_shape = (ternary_weights.shape[0], (ternary_weights.shape[1] + 3) // 4)
        packed_weights = torch.zeros(packed_shape, dtype=torch.uint8)

        for i in range(ternary_weights.shape[0]):
            for j in range(ternary_weights.shape[1]):
                # Find the index in the packed representation
                packed_idx = j // 4
                bit_idx = (j % 4) * 2  # Each weight needs 2 bits, so multiply by 2

                # Convert ternary (-1, 0, 1) to binary (00, 01, 10) representation
                # Adjusting the value: -1 -> 00 (0), 0 -> 01 (1), 1 -> 10 (2)
                value = int(ternary_weights[i, j].item()) + 1
                if not 0 <= value <= 2:
                    raise ValueError(f"Invalid ternary value: {value}")

                # Shift the binary representation to the correct position and OR it with the packed byte
                packed_weights[i, packed_idx] |= (value << bit_idx)

        return packed_weights

    def ternary_linear(self, x, packed_weights, bias):
        batch_size, seq_length, input_dim = x.shape
        x_reshaped = x.reshape(batch_size * seq_length, input_dim)
        output_dim = packed_weights.shape[0]
        output = torch.zeros((batch_size * seq_length, output_dim), dtype=torch.float32, device=x.device)

        for i in range(output_dim):
            for j in range(packed_weights.shape[1]):
                packed_value = packed_weights[i, j].item()
                for k in range(4):
                    # Extract the 2-bit value (value can be 0, 1, or 2)
                    two_bit_value = (packed_value >> (2 * k)) & 0b11

                    # Map the 2-bit value back to ternary (-1, 0, 1)
                    if two_bit_value == 0:
                        weight = -1
                    elif two_bit_value == 2:
                        weight = 1
                    else:
                        weight = 0

                    # Apply the weight value
                    if weight != 0:
                        output[:, i] += weight * x_reshaped[:, j * 4 + k]

        if bias is not None:
            output += bias.unsqueeze(0)

        return output.reshape(batch_size, seq_length, output_dim)

    def ternary_linear_optimized(self, x, packed_weights, bias):
        # Reshape x for matrix operations: [Batch size * Sequence length, Input dimension]
        x_reshaped = x.reshape(-1, x.size(-1))

        # Initialize the output tensor
        output = torch.zeros(x_reshaped.size(0), self.out_features, device=x.device)

        # Iterate over each output dimension
        for i in range(self.out_features):
            # Extract the packed weights for the i-th output neuron
            for j in range((self.in_features + 3) // 4):
                # Get the 8-bit packed weight
                packed_value = packed_weights[i, j].item()

                # Process 4 weights at a time within each packed byte
                for k in range(4):
                    # Decode the 2-bit weight (-1, 0, 1)
                    weight = ((packed_value >> (k * 2)) & 0b11) - 1

                    # Skip if weight is 0 (no operation needed)
                    if weight == 0:
                        continue

                    # Determine the index of the current weight
                    weight_idx = j * 4 + k

                    # Check for index out of bounds (for the last byte)
                    if weight_idx >= self.in_features:
                        break

                    # Update the output: Add or subtract the input based on the weight
                    if weight == 1:
                        output[:, i] += x_reshaped[:, weight_idx]
                    elif weight == -1:
                        output[:, i] -= x_reshaped[:, weight_idx]

        # Add bias if it's not None
        if bias is not None:
            output += bias.unsqueeze(0)

        return output.reshape(x.size(0), x.size(1), self.out_features)


# Transformer model with ternary quantized weights
class TernaryTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(100, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, activation='gelu',
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = TernaryLinear(d_model, vocab_size)

    def forward(self, x):
        positions = torch.arange(x.size(1)).unsqueeze(0).expand(x.size(0), -1).to(x.device)
        x = self.embedding(x) + self.pos_encoder(positions)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x


# Load and preprocess the Shakespeare dataset
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [char_to_idx[ch] for ch in text]


def decode(indices):
    return ''.join(idx_to_char[i] for i in indices)


# Hyperparameters
vocab_size = len(chars)
d_model = 128
nhead = 4
num_layers = 2
batch_size = 16
seq_length = 64
num_epochs = 10

# Create the model and optimizer
model = TernaryTransformer(vocab_size, d_model, nhead, num_layers)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    encoded_text = encode(text)
    num_batches = len(encoded_text) // (batch_size * seq_length)
    encoded_text = encoded_text[:num_batches * batch_size * seq_length]
    encoded_text = torch.tensor(encoded_text).view(batch_size, -1)

    for i in range(0, encoded_text.size(1) - seq_length, seq_length):
        x = encoded_text[:, i:i + seq_length]
        y = encoded_text[:, i + 1:i + seq_length + 1]

        optimizer.zero_grad()
        outputs = model(x)

        # Reshape the outputs tensor to match the target tensor shape
        outputs_reshaped = outputs.view(batch_size, seq_length, -1)
        loss = nn.functional.cross_entropy(outputs_reshaped.reshape(-1, outputs_reshaped.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()

        if (i // seq_length) % 100 == 0:
            print(f"Batch {i // seq_length}, Loss: {loss.item():.4f}")

# Generate sample text
context = "To be, or not to be:"
context_encoded = encode(context)
context_tensor = torch.tensor(context_encoded).unsqueeze(0)

generated_text = context
with torch.no_grad():
    for _ in range(100):
        outputs = model(context_tensor)
        probs = torch.softmax(outputs[:, -1], dim=-1)
        next_char_idx = torch.multinomial(probs, 1).item()
        next_char = idx_to_char[next_char_idx]

        generated_text += next_char
        context_tensor = torch.cat([context_tensor[:, 1:], torch.tensor([[next_char_idx]])], dim=1)

print(generated_text)