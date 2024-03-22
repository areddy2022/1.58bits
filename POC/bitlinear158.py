import torch
import torch.nn as nn


class BitLinear158Optimized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_groups=1, activation_bits=8):
        super(BitLinear158Optimized, self).__init__(in_features, out_features, bias=False)  # No bias in LLaMa models
        self.num_groups = num_groups
        self.eps = 1e-5
        self.activation_bits = activation_bits  # Assuming 8-bit activations as described in the BitNet Paper
        self.latent_weight = nn.Parameter(self.weight.data.clone().half())

        # Initialize buffer for ternary quantized weights
        self.register_buffer('ternary_quantized_weights', torch.zeros_like(self.weight))

        # Indicates whether the model is in training or inference mode
        self.is_training = True

    def quantize_weights(self, W):
        gamma = torch.mean(torch.abs(W), dim=1, keepdim=True) + self.eps
        W_scaled = W.float() / gamma
        W_quantized = torch.sign(W_scaled) * torch.clamp(torch.abs(W_scaled).round(), max=1.0)
        return W_quantized

    def quantize_activations(self, x):
        scale = torch.max(torch.abs(x))
        max_val = 2**(self.activation_bits - 1) - 1
        x_quantized = torch.round(x / scale * max_val)
        return x_quantized, scale

    def map_weights_to_2bit(self, quantized_weights):
        mapped_weights = quantized_weights + 1  # Mapping: -1 -> 0, 0 -> 1, 1 -> 2
        return mapped_weights.to(torch.uint8)

    def pack_weights(self, mapped_weights):
        packed_rows = []
        # This can be parallelized because each row is independent of each other
        # We can use the Rust bindings to generate high performant code to cut down on the run time for this function
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

    def forward(self, input):
        # Quantize weights and update the register buffer
        quantized_weights = self.quantize_weights(self.latent_weight).detach()

        # Use the ternary quantized weights for the forward computation
        if self.is_training:
            # During training, re-attach the quantized weights to the computational graph
            quantized_weights_ste = quantized_weights.clone().requires_grad_()
            # Use STE: allow gradients to pass through the quantized weights unchanged
            quantized_weights_ste.register_hook(lambda grad: grad)
            # Placeholder for using packed weights, e.g., with custom CUDA kernel
            output = torch.nn.functional.linear(input, quantized_weights_ste, self.bias)

        else:
            packed_weights = self.pack_weights(self.map_weights_to_2bit(quantized_weights))

            # Placeholder for using packed weights, e.g., with custom CUDA kernel
            # output = self.custom_cuda_forward(input, packed_weights) # Uncomment once implemented
            output = torch.nn.functional.linear(input, packed_weights, self.bias)

        output = self.quantize_output(output)

        return output

    def custom_cuda_forward(self, input, packed_weights):
        # Placeholder function to represent the forward pass using the custom CUDA kernel
        # In a real scenario, this would involve a call to a CUDA kernel with packed_weights
        return torch.nn.functional.linear(input, packed_weights.float(), self.bias)


    def switch_to_inference(self):
        # The is_training flag is used to switch the behavior of the forward pass between training and inference
        # During training, the class uses the quantized weights directly
        # When switching to inference (by calling switch_to_inference),
        # the class will pack the weights and use them for computation,
        # which is intended to be handled by a custom CUDA kernel (represented as a placeholder custom_cuda_forward function).
        self.is_training = False
