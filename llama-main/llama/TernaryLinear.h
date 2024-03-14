#ifndef TERNARY_LINEAR_H
#define TERNARY_LINEAR_H

// similar to other class this needs to be fixed

#include <torch/torch.h>
#include <cmath>

torch::Tensor roundClip(const torch::Tensor& x, double a, double b) {
    return x.clamp(a, b).round();
}

torch::Tensor ternaryQuantize(const torch::Tensor& weights, double gamma = 1.0) {
    const double eps = 1e-7;
    auto w_bar = weights.abs().mean();
    auto gamma_prime = gamma / (w_bar + eps);
    auto quantized_weights = torch::where(weights > 0.5, torch::ones_like(weights),
                                          torch::where(weights < -0.5, -torch::ones_like(weights),
                                                       torch::zeros_like(weights)));
    return roundClip(gamma_prime * quantized_weights, -1.0, 1.0);
}

class TernaryLinearImpl : public torch::nn::Module {
public:
    TernaryLinearImpl(int64_t in_features, int64_t out_features)
        : in_features(in_features), out_features(out_features) {
        weight = register_parameter("weight", torch::randn({out_features, in_features}));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto quantized_weights = ternaryQuantize(weight);
        return torch::matmul(x, quantized_weights);
    }

private:
    int64_t in_features, out_features;
    torch::Tensor weight;
};

TORCH_MODULE(TernaryLinear);

#endif // TERNARY_LINEAR_H
