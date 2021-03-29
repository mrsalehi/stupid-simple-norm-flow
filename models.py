import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from torch.distributions.multivariate_normal import MultivariateNormal
from flows import *


class FCNEncoder(nn.Module):
    def __init__(self, hidden_sizes: List[int], dim_input: int, activation=nn.ReLU()):
        super().__init__()
        
        
        hidden_sizes = [dim_input] + hidden_sizes
        self.net = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.net.append(activation)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class FlowModel(nn.Module):
    def __init__(self, flows: List[str], D: int, activation=F.elu):
        super().__init__()
        
        self.prior = MultivariateNormal(torch.zeros(D), torch.eye(D))
        self.net = nn.ModuleList()

        for i in range(len(flows)):
            layer_class = eval(flows[i])
            self.net.append(layer_class(D, activation))

        self.D = D


    def forward(self, mu: torch.Tensor, sigma: torch.Tensor):
        """
        mu: tensor with shape (batch_size, D)
        sigma: tensor with shape (batch_size, D)
        """
        batch_size = mu.shape[0]
        samples = self.prior(mu, sigma * torch.eye(D)).sample(batch_size)
        z = samples * sigma + mu 
        
        log_det = torch.zeros((batch_size, 1))
        for layer in self.layers:
            z, ld = layer(z)
            log_det += ld
        
        return z, log_det


class FCNDecoder(nn.Module):
    def __init__(self, hidden_sizes: List[int], dim_input: int, activation=nn.ReLU()):
        super().__init__()
        
        hidden_sizes = [dim_input] + hidden_sizes
        self.net = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.net.append(activation)

    def forward(self, z: torch.Tensor):
        return self.net(z)