import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary_quantization(weights):
    # Quantize weights to -1, 0, or 1
    return torch.sign(weights)


class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(TernaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.normal_(self.weight)  # Initialize weights

    def forward(self, x):
        # Apply ternary quantization to weights
        quantized_weights = ternary_quantization(self.weight)
        return F.linear(x, quantized_weights, None)

    def view_quantized_weights(model):
        for module in model.modules():
            if isinstance(module, TernaryLinear):
                weight = module.weight.data  # original weight data
                quantized_weight = ternary_quantization(weight)
                print(quantized_weight)


class SimpleGPT(nn.Module):
    def __init__(self, num_tokens, dim, num_layers, num_heads):
        super(SimpleGPT, self).__init__()
        self.token_embedding = nn.Embedding(num_tokens, dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True))
            # Replace linear layers in Transformer with TernaryLinear
            self.layers[-1].linear1 = TernaryLinear(dim, dim * 4)
            self.layers[-1].linear2 = TernaryLinear(dim * 4, dim)

        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x):
        seq_length = x.size(1)
        # Ensure positional_embedding is the right size
        if seq_length > self.positional_embedding.size(1):
            raise ValueError(
                "Input sequence length exceeds the maximum length for which positional embeddings are defined.")

        # Use the relevant portion of the positional_embedding
        positional_embedding = self.positional_embedding[:, :seq_length, :]

        x = self.token_embedding(x) + positional_embedding
        for layer in self.layers:
            x = layer(x)
        logits = self.to_logits(x)
        return logits


# Example usage
model = SimpleGPT(num_tokens=10000, dim=512, num_layers=6, num_heads=8)
input_ids = torch.randint(0, 10000, (1, 1024))
logits = model(input_ids)
TernaryLinear.view_quantized_weights(model)
