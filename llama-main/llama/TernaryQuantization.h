#ifndef TERNARY_QUANTIZATION_H
#define TERNARY_QUANTIZATION_H

// library for libtorch needs to be fixed

#include <torch/torch.h>
#include <vector>

typedef struct {
    unsigned char TwoBits:2;
    unsigned char Unused:6;
} TernaryWeight;

std::vector<TernaryWeight> ternaryQuantizeToPacked(const torch::Tensor& weights, double gamma = 1.0) {
    const double eps = 1e-7;
    auto w_bar = weights.abs().mean();
    auto gamma_prime = gamma / (w_bar + eps);
    auto quantized = torch::where(weights > 0.5, torch::ones_like(weights) * 2,
                                  torch::where(weights < -0.5, torch::zeros_like(weights),
                                               torch::ones_like(weights)));
    
    std::vector<TernaryWeight> packedWeights;
    packedWeights.reserve(weights.numel());

    auto quantized_accessor = quantized.accessor<float,1>(); // Assuming a 1D tensor for simplicity

    for (int i = 0; i < quantized.numel(); ++i) {
        TernaryWeight tw;
        tw.TwoBits = static_cast<unsigned char>(quantized_accessor[i]);
        tw.Unused = 0;
        packedWeights.push_back(tw);
    }

    return packedWeights;
}

#endif // TERNARY_QUANTIZATION_H
