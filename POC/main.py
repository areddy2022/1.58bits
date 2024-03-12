import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TernaryQuantization(nn.Module):
    def __init__(self):
        super(TernaryQuantization, self).__init__()

    def forward(self, x):
        mean_abs = x.abs().mean()
        return torch.clip(torch.round(x / mean_abs), -1, 1)

    def encode_to_packed_bytes(self, x):
        # Assuming x is a tensor of ternary quantized values (-1, 0, 1)
        # First, remap values from -1,0,1 to 0,1,2 for bit representation
        x = x.add(1).to(torch.uint8)

        # Pack 4 2-bit values into one byte
        packed = torch.bitwise_or(torch.bitwise_or(x[::4], x[1::4].left_shift(2)),
                                  torch.bitwise_or(x[2::4].left_shift(4), x[3::4].left_shift(6)))

        return packed

    def decode_from_packed_bytes(self, packed):
        # Unpack bytes to 4 values
        unpacked = torch.stack([(packed & 3),
                                (packed >> 2) & 3,
                                (packed >> 4) & 3,
                                (packed >> 6) & 3], dim=-1).flatten()

        # Remap 0,1,2 back to ternary values -1,0,1
        return unpacked.sub(1).to(torch.float32)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Assuming QuantizedTransformer wraps around GPT2LMHeadModel instead of GPT2Model
class QuantizedTransformer(nn.Module):
    def __init__(self, original_model):
        super(QuantizedTransformer, self).__init__()
        self.original_model = original_model
        self.quantize = TernaryQuantization()

    def forward(self, input_ids):
        with torch.no_grad():
            self.original_model.eval()
            for param in self.original_model.parameters():
                quantized = self.quantize(param.data)
                param.data = quantized

        return self.original_model(input_ids, use_cache=False)

    def generate(self, *args, **kwargs):
        return self.original_model.generate(*args, **kwargs)

# Load a pre-trained GPT-2 LM Head model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Replace the model with the quantized version
quantized_model = QuantizedTransformer(model)

# Function to generate text from the model
def generate_text(model, input_text, max_length=50):
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate predictions
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length)

    # Decode and return the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Example usage
input_text = "What color is the sky?"
generated_text = generate_text(quantized_model, input_text)
print(generated_text)