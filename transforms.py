import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanarFlowLayer(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.linear = nn.Linear(in_features=D, out_features=1)
        self.u = nn.Parameter(requires_grad=True)

        nn.init.xavier_normal_(self.z)
        nn.init.xavier_normal_(self.b)
        nn.init.xavier_normal_(self.u.data) 

    def forward(self, z: torch.Tensor):
        return z + u @ F.relu(self.linear(z))  #TODO: Figure out what h is (the paper says that it is a smooth nonlinearity)


class RadialFlowLayer(nn.Module):
    def __init__(self, D, nonlinearity):
        super().__init__()

        self.linear = nn.Linear(in_features=D, out_features=1)
        self.u = nn.Parameter(torch.empty(D,), requires_grad=True)
        self.z0 = nn.Parameter(torch.empty(D), requires_grad=True)
        self.log_alpha = nn.Paramter(torch.empty(1), requires_grad=True)
        self.beta = nn.Paramter(torch.empty(1), requires_grad=True)

        nn.init.xavier_normal_(self.z)
        nn.init.xavier_normal_(self.b)
        nn.init.xavier_normal_(self.u.data)
        nn.init.normal_(self.z0.data) 
        nn.init.uniform_(self.log_alpha)
        nn.init.uniform_(self.beta)


    def forward(self, z: torch.Tensor):
        z_ = z - z0
        return z + beta * (1 / (torch.exp(self.log_alpha) + torch.norm(z_))) *  (z_ - self.z0)
       