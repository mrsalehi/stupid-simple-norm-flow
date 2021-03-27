import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanarFlow(nn.Module):
    def __init__(self, D, activation=F.elu):
        super().__init__()
        self.D = D
        self.w = nn.Parameter(torch.empty(D))
        self.b = nn.Parameter(torch.empty(1))
        self.u = nn.Parameter(torch.empty(D))
        self.activation = activation

        nn.init.xavier_normal_(self.z)
        nn.init.xavier_normal_(self.b)
        nn.init.xavier_normal_(self.u.data) 

    def forward(self, z: torch.Tensor):
        lin = self.activation(z @ self.w + b).unsqueeze(1)
        phi = F_prime(lin) * w 
        log_det = torch.log(1 + phi @ u)

        return log_det, z + u * self.activation(lin)


class RadialFlow(nn.Module):
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
       