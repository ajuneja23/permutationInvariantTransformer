import torch

import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, input_dim=16, query_dim=32, value_dim=32):
        super(Attention, self).__init__()
        self.input_dim, self.query_dim, self.value_dim = input_dim, query_dim, value_dim
        self.query = nn.Linear(input_dim, query_dim)
        self.key = nn.Linear(input_dim, query_dim)
        self.value = nn.Linear(input_dim, value_dim)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = queries @ keys.T
        scores /= self.query_dim**0.5
        scores = nn.Softmax(dim=-1)(scores)
        newVals = scores @ values
        return newVals

    def forward(self, x, y):
        queries = self.query(x)
        keys = self.key(y)
        values = self.value(y)
        scores = queries @ keys.T
        scores /= self.query_dim**0.5
        scores = nn.Softmax(dim=-1)(scores)
        newVals = scores @ values
        return newVals


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim=16, query_dim=32, value_dim=32, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.input_dim, self.query_dim, self.value_dim = input_dim, query_dim, value_dim
        self.numHeads = heads
        self.attnHeads = nn.ModuleList(
            Attention(input_dim, query_dim, value_dim) for _ in range(heads)
        )
        self.fc = nn.Linear(heads * value_dim, input_dim)

    def forward(self, x):
        head_outputs = [head(x) for head in self.attnHeads]
        head_outputs = torch.cat(head_outputs, dim=-1)
        return self.fc(head_outputs)

    def forward(self, x, y):
        head_outputs = [head(x, y) for head in self.attnHeads]
        head_outputs = torch.cat(head_outputs, dim=-1)
        return self.fc(head_outputs)


class MAB_SAB(
    nn.Module
):  # USE single param forward func for SAB(Set Attention Block), double param forward for MAB(MultiHead Attn Block)
    def __init__(self, input_dim=16, query_dim=32, value_dim=32, heads=8):
        super(MAB_SAB, self).__init__()
        self.input_dim, self.query_dim, self.value_dim = input_dim, query_dim, value_dim
        self.numHeads = heads
        self.mha = MultiHeadAttention(input_dim, query_dim, value_dim, heads)
        self.fc = nn.Linear(input_dim, input_dim)
        self.lnorm1 = nn.LayerNorm(input_dim)
        self.lnorm2 = nn.LayerNorm(input_dim)

    def forward(self, x, y):
        h = self.lnorm1(x + self.mha(x, y))
        postFC_H = self.fc(h)
        return self.lnorm2(h + postFC_H)

    def forward(self, x):
        h = self.lnorm1(x + self.mha(x, x))
        postFC_H = self.fc(h)
        return self.lnorm2(h + postFC_H)


class ISAB(nn.Module):  # Inducted Set Attention Block
    def __init__(self, input_dim=16, query_dim=32, value_dim=32, heads=8, m=4):
        super(ISAB, self).__init__()
        self.input_dim, self.query_dim, self.value_dim = input_dim, query_dim, value_dim
        self.numHeads = heads
        self.mab1 = MAB_SAB(input_dim, query_dim, value_dim, heads)
        self.mab2 = MAB_SAB(input_dim, query_dim, value_dim, heads)
        self.inducingPoints = nn.Parameter(torch.randn(m, input_dim))

    def forward(self, x):
        compressediPointReps = self.mab1(self.inducingPoints, x)
        return self.mab2(x, compressediPointReps)


class PMA(nn.Module):  # Pooling by MultiHead Attention
    def __init__(self, input_dim=16, query_dim=32, value_dim=32, heads=8, k=4):
        super(PMA, self).__init__()
        self.input_dim, self.query_dim, self.value_dim = input_dim, query_dim, value_dim
        self.numHeads = heads
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.kVectors = nn.Parameter((k, input_dim))  # S matrix in original paper
        self.mab1 = MAB_SAB(input_dim, query_dim, value_dim, heads)

    def forward(self, x):
        y = self.fc1(x)
        return self.mab1(self.kVectors, y)
