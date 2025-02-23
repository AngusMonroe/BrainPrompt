
import math

import dgl
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from layers.mlp_readout_layer import MLPReadout
from layers.gcn_layer import GCNLayer, simpleGCNLayer
from layers.mlp_readout_layer import MLPReadout
from layers.attention_layer import LMPE


class BrainPromptGNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        edge_dim = net_params['edge_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.e_feat = net_params['edge_feat']
        self.node_num = net_params['node_num']
        self.label_embs = net_params['label_embs']
        self.lambda1 = net_params['lambda1']
        self.lm_dim = 2048

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(edge_dim, hidden_dim)
        # self.in_feat_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.batch_norm, self.residual,
                                              e_feat=self.e_feat) for _ in range(n_layers - 1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual, e_feat=self.e_feat))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        self.linear_layer1 = nn.Linear(self.lm_dim, out_dim)
        self.pos_emb = LMPE(emb_dim=self.lm_dim, d_model=out_dim, node_num=self.node_num, dropout=dropout)
        self.linear_layer2 = nn.Linear(out_dim, out_dim)
        self.label_transform = nn.Linear(self.lm_dim, out_dim)
        self.fused_repr = None
        self.global_gcs = simpleGCNLayer(hidden_dim, F.relu, 0.0, batch_norm=False, residual=True)

    def forward(self, g, h, e, batch_llms):

        device = h.device

        h = self.embedding_h(h)
        # e = self.embedding_e(e)
        # h = self.in_feat_dropout(h)

        hidden_dim = h.shape[-1]
        h = self.pos_emb(h.reshape(-1, self.node_num, hidden_dim)).reshape(-1, hidden_dim)

        for conv in self.layers:
            h, e = conv(g, h, e)

        g.ndata['h'] = h
        # g.edata['h'] = e

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        llm = batch_llms.squeeze(1)
        meta_repr = self.linear_layer1(llm)

        hg_sim = self.sim(hg)
        llm_sim = self.sim(meta_repr)
        binary_matrix = self.dropout(((hg_sim * llm_sim) > 0.8).float())
        fused_repr = self.global_gcs(hg, binary_matrix)

        self.fused_repr = fused_repr
        self.label_reprs = self.label_transform(self.label_embs.to(device))

        scores = self.MLP_layer(fused_repr)

        return scores

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
