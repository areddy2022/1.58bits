/*
This code depends on the tch-rs package for pytorch bindings: https://github.com/LaurentMazare/tch-rs
And upon rust-cython for python bindings: https://github.com/dgrunwald/rust-cpython

So rust is not for this, we'll need C++ i guess since we cant manipulate less than a byte
*/

extern crate tch;
use tch::{nn, nn::Module, Tensor};

fn round_clip(x: &Tensor, a: f64, b: f64) -> Tensor {
    x.clamp(a, b).round()
}

fn ternary_quantize(weights: &Tensor, gamma: f64) -> Tensor {
    let eps = 1e-7;
    let w_bar = weights.abs().mean(Kind::Float);
    let gamma_prime = gamma / (w_bar + eps);
    let quantized_weights = weights.where_gt_scalar(0.5, &Tensor::ones_like(weights))
                                    .where_lt_scalar(-0.5, &Tensor::full_like(weights, -1.0))
                                    .where_within(-0.5..=0.5, &Tensor::zeros_like(weights));
    round_clip(&(gamma_prime * quantized_weights), -1.0, 1.0)
}

struct TernaryLinear {
    in_features: i64,
    out_features: i64,
    weight: Tensor,
}

impl TernaryLinear {
    fn new(vs: &nn::Path, in_features: i64, out_features: i64) -> TernaryLinear {
        let weight = Tensor::randn(&[out_features, in_features], (tch::Kind::Float, vs.device()))
                        .set_requires_grad(true);
        TernaryLinear {
            in_features,
            out_features,
            weight,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let quantized_weights = ternary_quantize(&self.weight, 1.0);
        x.matmul(&quantized_weights)
    }
}

impl Module for TernaryLinear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward(xs)
    }
}
