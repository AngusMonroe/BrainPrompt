# coding: utf-8

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
# from layers.seq_enc_layer import SequenceEncoder


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, node_num, d, bias=False):
        super(E2EBlock, self).__init__()
        self.d = d  # example.size(3)
        self.node_num = node_num
        self.cnn1 = nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = nn.Conv2d(in_planes, planes, (self.node_num, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.node_num, 2)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, net_params):
        super(BrainNetCNN, self).__init__()
        # example: [203, 1, 64, 64]
        in_planes = 1  # example.size(1)
        d = net_params['in_dim']  # example.size(3)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        self.node_num = net_params['node_num']

        self.e2econv1 = E2EBlock(in_planes, 32, self.node_num, d, bias=True)
        self.e2econv2 = E2EBlock(32, hidden_dim, self.node_num, d, bias=True)
        self.E2N = nn.Conv2d(hidden_dim, 1, (1, d))
        self.N2G = nn.Conv2d(1, out_dim, (self.node_num, 1))
        self.dense1 = nn.Linear(out_dim, 128)
        self.dense2 = nn.Linear(128, 30)
        self.dense3 = nn.Linear(30, n_classes)
        # self.seq_enc = SequenceEncoder('gru', d, 8)

    def forward(self, g, h, e):
        # h = self.seq_enc(h)
        x = h.reshape(-1, 1, self.node_num, h.size(1))

        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
