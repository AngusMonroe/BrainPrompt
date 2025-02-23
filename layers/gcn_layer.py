import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from norm.GraphNorm import Norm

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""


class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}


class GCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """
    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, residual=False, dgl_builtin=True, e_feat=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin
        
        if in_dim != out_dim:
            self.residual = False

        # self.batchnorm_h = nn.BatchNorm1d(out_dim)
        # self.batchnorm_e = nn.BatchNorm1d(out_dim)
        self.batchnorm_h = Norm('bn', out_dim)
        if e_feat:
            self.batchnorm_e = Norm('bn', out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if self.dgl_builtin == False:
            self.apply_mod = NodeApplyModule(in_dim, out_dim)
        elif dgl.__version__ < "0.5":
            self.conv = GraphConv(in_dim, out_dim)
        else:
            self.conv = GraphConv(in_dim, out_dim, allow_zero_in_degree=True)

        # Sends a message of node feature h
        # Equivalent to => return {'m': edges.src['h']}
        self.message_func = fn.copy_u(u='h', out='m') if not e_feat else fn.u_mul_e('h', 'e', 'm')
        self.reduce_func = fn.mean('m', 'h')

    # def forward(self, g, feature):
    #     h_in = feature   # to be used for residual connection
    #
    #     if self.dgl_builtin == False:
    #         g.ndata['h'] = feature
    #         g.update_all(self.message_func, self.reduce_func)
    #         g.apply_nodes(func=self.apply_mod)
    #         h = g.ndata['h'] # result of graph convolution
    #     else:
    #         h = self.conv(g, feature)
    #
    #     if self.batch_norm:
    #         h = self.batchnorm_h(h) # batch normalization
    #
    #     if self.activation:
    #         h = self.activation(h)
    #
    #     if self.residual:
    #         h = h_in + h # residual connection
    #
    #     h = self.dropout(h)
    #     return h
        
    def forward(self, g, feature, e):
        h_in = feature   # to be used for residual connection
        e_in = e

        if self.dgl_builtin == False:
            g.ndata['h'] = feature
            g.edata['e'] = e
            g.update_all(self.message_func, self.reduce_func)
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata['h'] # result of graph convolution
            e = g.edata['e']
        else:
            h = self.conv(g, feature)

        if self.batch_norm:
            h = self.batchnorm_h(g, h) # batch normalization
            # e = self.batchnorm_e(g, e) # batch normalization
       
        if self.activation:
            h = self.activation(h)
            # e = self.activation(e)
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e
            
        h = self.dropout(h)
        e = self.dropout(e)
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channels,
                                                                         self.out_channels, self.residual)


class simpleGCNLayer(nn.Module):
    def __init__(self, hidden_dim, activation, dropout, batch_norm, residual=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.residual = residual

        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(1, hidden_dim))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        self.batchnorm_h = nn.BatchNorm1d(hidden_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        support = torch.matmul(X, self.weight)
        h = torch.matmul(A, support)
        h = h + self.bias

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = X + h  # residual connection

        h = self.dropout(h)

        return h