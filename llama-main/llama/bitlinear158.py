import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_activations(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def quantize_weights(W):
    scale = 1 / W.abs().mean().clamp_(1e-5)
    u = (W * scale).round().clamp_(-1, 1) / scale
    return u


def activation_norm_quant(x):
    x = RMSNorm(x)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale


def RMSNorm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def map_weights_to_2bit(quantized_weights):
    mapped_weights = quantized_weights + 1  # Mapping: -1 -> 0, 0 -> 1, 1 -> 2
    return mapped_weights.to(torch.uint8)


def pack_weights(mapped_weights):
    packed_rows = []
    # This can be parallelized because each row is independent of each other
    # We can use the Rust bindings to generate high performant code to cut down on the run time for this function
    # Okay maybe I'm going crazy but what happens if we divided each row into four number sub blocks then parallely
    # converted those sub blocks into a single int8????
    for row in mapped_weights:
        full_groups, remainder = divmod(row.numel(), 4)
        packed_row = torch.zeros(full_groups + (remainder > 0), dtype=torch.uint8, device=row.device)
        for i in range(full_groups):
            idx = i * 4
            packed_value = (row[idx] << 6) | (row[idx + 1] << 4) | (row[idx + 2] << 2) | row[idx + 3]
            packed_row[i] = packed_value
        if remainder:
            last_value = sum(row[full_groups * 4 + j] << ((3 - j) * 2) for j in range(remainder))
            packed_row[-1] = last_value
        packed_rows.append(packed_row)
    return torch.stack(packed_rows)


class BitLinear158Optimized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_groups=1, activation_bits=8):
        super(BitLinear158Optimized, self).__init__(in_features, out_features, bias=False)  # No bias in LLaMa models
        self.num_groups = num_groups
        self.eps = 1e-5
        self.weight_scale = torch.tensor([1.0])
        self.activation_bits = activation_bits  # Assuming 8-bit activations as described in the BitNet Paper
        self.latent_weight = nn.Parameter(self.weight.data.clone().half())

        # I don't think we need this because we recompute the quantized weights on every forward pass
        # self.register_buffer('quantized_weights', torch.zeros_like(self.weight))

        # Initialize buffer for packed weights
        self.register_buffer('packed_weights', None)

        # Indicates whether the model is in training or inference mode
        self.is_training = True

    def forward(self, input):
        # Use the ternary quantized weights for the forward computation
        if self.is_training:
            weights = self.latent_weight
            x_norm = RMSNorm(input)
            x_quant = x_norm + (quantize_activations(x_norm) - x_norm).detach()
            quantized_weights = weights + (quantize_weights(weights) - weights).detach()
            y = F.linear(x_quant, quantized_weights)
            return y

        else:
            x_quant, x_scale = activation_norm_quant(input)
            y = custom_cuda_kernel(x_quant, self.packed_weights) / self.weight_scale / x_scale
            return y

            # Placeholder for using packed weights, e.g., with custom CUDA kernel

    def switch_to_inference(self):
        # The is_training flag is used to switch the behavior of the forward pass between training and inference
        # During training, the class uses the quantized weights directly
        # When switching to inference (by calling switch_to_inference),
        # the class will pack the weights and use them for computation,
        # which is intended to be handled by a custom CUDA kernel (represented as a placeholder custom_cuda_forward function).
        self.is_training = False

        # Quantize and map the weights to 2-bit
        quantized_weights = self.latent_weight + (quantize_weights(self.latent_weight) - self.latent_weight).detach()
        mapped_weights = map_weights_to_2bit(quantized_weights)

        # Pack the mapped weights
        self.packed_weights = pack_weights(mapped_weights)
