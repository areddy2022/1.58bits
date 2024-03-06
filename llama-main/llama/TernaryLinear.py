import torch
import torch.nn as nn
import torch.nn.functional as F

def RoundClip(x, a, b):
    """Round and clip the tensor values within [a, b]."""
    return torch.clamp(torch.round(x), a, b)

def ternary_quantize(weights, gamma=1.0):
    """Perform ternary quantization on the input weights."""
    eps = 1e-7
    W_bar = torch.mean(torch.abs(weights))
    gamma_prime = gamma / (W_bar + eps)
    quantized_weights = torch.where(weights > 0.5, torch.ones_like(weights),
                                    torch.where(weights < -0.5, -torch.ones_like(weights),
                                                torch.zeros_like(weights)))
    return RoundClip(gamma_prime * quantized_weights, -1, 1)

class TernaryLinear(nn.Module):
    """A ternary quantized linear layer."""
    def __init__(self, in_features, out_features):
        super(TernaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.normal_(self.weight)

    def forward(self, x):
        quantized_weights = ternary_quantize(self.weight)
        return F.linear(x, quantized_weights, None)

    @staticmethod
    def view_quantized_weights(model):
        for module in model.modules():
            if isinstance(module, TernaryLinear):
                weight = module.weight.data
                quantized_weight = ternary_quantize(weight)
                print(quantized_weight)
