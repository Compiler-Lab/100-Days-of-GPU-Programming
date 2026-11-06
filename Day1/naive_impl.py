import torch
import torch.nn as nn


# input, model, and output are on the GPU
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
        with torch.no_grad():
            result = model(input)
            output.copy_(result)
#0.54ms in A100