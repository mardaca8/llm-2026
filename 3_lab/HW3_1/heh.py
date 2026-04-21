import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored


model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# extract all parameters
parameters = []
for name, param in model.named_parameters():
    if param.requires_grad:
        parameters.append((name, param.data.cpu().numpy()))

# analyze model parameters
for name, param in parameters:
    print(colored(f"Parameter: {name}, Shape: {param.shape}, Mean: {param.mean():.4f}, Std: {param.std():.4f}, Min: {param.min():.4f}, Max: {param.max():.4f}", "yellow"))


# extract weights 
weights = model.transformer.h[0].attn.c_attn.weight.detach().cpu().numpy().flatten()
weights_series = pd.Series(weights)

def plot_transformer_weights_only(model):
    weight_tensors = []
    
    for name, param in model.named_parameters():
        # target the linear layers and ensure we aren't grabbing 1d vectors
        if "weight" in name and param.dim() > 1:
            
            w = param.data.cpu().numpy().flatten()
            weight_tensors.append(w[np.isfinite(w)])
    
    if not weight_tensors:
        print("No finite weights found.")
        return

    all_actual_weights = np.concatenate(weight_tensors)

    plt.figure(figsize=(10, 4))
    
    # Plotting using the robust percentile range
    plt.hist(all_actual_weights, bins=500, color='blue', log=True)
    
    plt.title("Weight Distribution", fontweight='bold')
    plt.xlabel("Value of weight", fontweight='bold')
    plt.ylabel("Relative scale [log(count)]", fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

plot_transformer_weights_only(model)

def linear_quantization(weights, bits=8):
    q_min, q_max = -2**(bits-1), 2**(bits-1) - 1
    scale = (weights.max() - weights.min()) / (q_max - q_min)
    zero_point = q_min - weights.min() / scale
    
    quantized = np.round(weights / scale + zero_point)
    quantized = np.clip(quantized, q_min, q_max)
    
    # Dequantize to compare in the same scale as original
    dequantized = (quantized - zero_point) * scale
    return dequantized

quantized_weights = linear_quantization(weights)
quantized_weights_series = pd.Series(quantized_weights)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(weights_series, bins=400, color='blue', log=True, rwidth=1.0)

axes[0].set_title("Original Weight Distribution")
axes[0].set_xlabel("Weight Value")

axes[1].hist(quantized_weights_series, bins=400, color='blue', log=True, rwidth=1.0)

axes[1].set_title("8-bit Linear Quantized Distribution")
axes[1].set_xlabel("Weight Value")

plt.show()

input_text = "How works internal combustion engine?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Original Inference
with torch.no_grad():
    original_output = model.generate(**inputs, max_new_tokens=30)
    print(colored(f"Original Result: {tokenizer.decode(original_output[0])}", "cyan"))


with torch.no_grad():
    # weights with quantized versions
    q_tensor = torch.tensor(quantized_weights).view(model.transformer.h[0].attn.c_attn.weight.shape)
    model.transformer.h[0].attn.c_attn.weight.copy_(q_tensor)
    
    quantized_output = model.generate(**inputs, max_new_tokens=30)
    print(colored(f"Quantized Result: {tokenizer.decode(quantized_output[0])}", "cyan"))



df_stats = pd.DataFrame({
    "Original": [weights.mean(), weights.min(), weights.max()],
    "Quantized": [quantized_weights.mean(), quantized_weights.min(), quantized_weights.max()]
}, index=["Mean", "Min", "Max"])
print(colored(df_stats,"cyan"))