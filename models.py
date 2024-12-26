import torch
from torch import nn
from torch.nn import Parameter, Module
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=90):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class GATLayer(Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attention(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attention(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :])
        return self.leakyrelu(Wh1 + Wh2.transpose(0, 1))


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(nfeat, nhid, dropout, alpha, concat=True)
        self.attention = Attention(nhid)
        self.fc = nn.Linear(nhid, 2)

    def forward(self, x, adj):
        h = self.layer1(x, adj)
        h, _ = self.attention(h)
        return F.log_softmax(self.fc(h), dim=1)