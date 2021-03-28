import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class InferenceNet(nn.Module):
    def __init__(self, hidden_sizes: List[int], dim_input: int, activation=nn.ReLU):
        super().__init__()
        
        layers = []
        
        hidden_sizes = [dim_input] + hidden_sizes
        
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation)

        self.net = nn.ModuleList(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)