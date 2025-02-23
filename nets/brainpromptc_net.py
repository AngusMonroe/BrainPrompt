# coding: utf-8

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from layers.attention_layer import LMPE
from layers.gcn_layer import simpleGCNLayer


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


class BrainPromptCNet(torch.nn.Module):
    def __init__(self, net_params):
        super(BrainPromptCNet, self).__init__()
        in_planes = 1  # example.size(1)
        d = net_params['in_dim']  # example.size(3)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        self.node_num = net_params['node_num']
        self.label_embs = net_params['label_embs']
        self.lambda1 = net_params['lambda1']
        self.lm_dim = 2048
        dropout = 0.5

        self.e2econv1 = E2EBlock(in_planes, 32, self.node_num, d, bias=True)
        self.e2econv2 = E2EBlock(32, hidden_dim, self.node_num, d, bias=True)
        self.E2N = nn.Conv2d(hidden_dim, 1, (1, d))
        self.N2G = nn.Conv2d(1, out_dim, (self.node_num, 1))
        self.dense1 = nn.Linear(out_dim, 128)
        self.dense2 = nn.Linear(128, 30)
        self.dense3 = nn.Linear(30, n_classes)

        self.linear_layer1 = nn.Linear(self.lm_dim, out_dim)
        self.pos_emb = LMPE(emb_dim=self.lm_dim, d_model=out_dim, node_num=self.node_num, dropout=dropout)
        self.linear_layer2 = nn.Linear(out_dim, out_dim)
        self.label_transform = nn.Linear(self.lm_dim, out_dim)
        self.fused_repr = None
        self.global_gcs = simpleGCNLayer(hidden_dim, F.relu, 0.0, batch_norm=False, residual=True)

    def forward(self, g, h, e, batch_llms):

        hidden_dim = h.shape[-1]
        h = self.pos_emb(h.reshape(-1, self.node_num, hidden_dim)).reshape(-1, hidden_dim)

        x = h.reshape(-1, 1, self.node_num, h.size(1))

        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)

        device = h.device
        llm = batch_llms.squeeze(1)
        meta_repr = self.linear_layer1(llm)

        hg_sim = self.sim(out)
        llm_sim = self.sim(meta_repr)
        binary_matrix = ((hg_sim * llm_sim) > 0.7).float()
        fused_repr = self.global_gcs(out, binary_matrix)

        self.fused_repr = fused_repr
        self.label_reprs = self.label_transform(self.label_embs.to(device))

        out = F.dropout(F.leaky_relu(self.dense1(fused_repr), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        loss += self.lambda1 * self.compute_label_loss(label)

        return loss

    def compute_label_loss(self, label, weight=None):
        text_features = self.label_reprs

        # normalized features
        graph_features = self.fused_repr / self.fused_repr.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = graph_features @ text_features.t()

        # cross entropy loss
        loss = F.cross_entropy(logits_per_image, label, weight=weight)
        return loss


    def sim(self, matrix):
        # Compute cosine similarity matrix
        x_norm = F.normalize(matrix, p=2, dim=1)  # Normalize each row
        sim_matrix = torch.mm(x_norm, x_norm.T)  # Compute cosine similarity

        return torch.sigmoid(sim_matrix)
