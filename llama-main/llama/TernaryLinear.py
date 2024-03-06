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

    @staticmethod
    def view_quantized_weights(model):
        for module in model.modules():
            if isinstance(module, TernaryLinear):
                weight = module.weight.data  # original weight data
                quantized_weight = ternary_quantization(weight)
                print(quantized_weight)
