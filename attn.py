import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.functional import softmax
from torch.nn import Sequential

class attn_layer(nn.Module):
    def __init__(self, d, a, normalize=True):
        super(attn_layer, self).__init__()

        self.Q = Parameter(torch.randn((1, 1, a, d), requires_grad=True))
        self.K = Parameter(torch.randn((1, 1, a, d), requires_grad=True))
        self.V = Parameter(torch.randn((1, 1, d, d), requires_grad=True))
        self.normalize = normalize

    def forward(self, x):
        Qx = (self.Q @ x.unsqueeze(-1)).squeeze(-1)
        Kx = (self.K @ x.unsqueeze(-1)).squeeze(-1)
        Vx = (self.V @ x.unsqueeze(-1)).squeeze(-1)
        A = (Qx.unsqueeze(1) * Kx.unsqueeze(2)).sum(-1)
        if self.normalize:
            A = softmax(A, 1)

        res = (A.unsqueeze(-1) * Vx.unsqueeze(2)).sum(1)
        return res


class deep_attn(nn.Module):
    def __init__(self, l, d, a, normalize=True):
        super(deep_attn, self).__init__()
        self.net = Sequential()
        for i in range(l):
            self.net.append(attn_layer(d, a, normalize=normalize))

    def forward(self, x):
        return self.net(x)
