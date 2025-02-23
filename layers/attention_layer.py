import torch
import torch.nn as nn
import math
import csv
import numpy as np
from data.BrainNet import name2coor_path


def get_3d_corr(name):
    path = name2coor_path[name]
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='\n')
        if name not in ['abide_schaefer100', 'abide_AAL116']:
            coor = [row[1:] for row in spamreader][1:]
        else:
            coor = [row[1:] for row in spamreader]
    return np.array(coor, dtype='float')


class Gate(nn.Module):
    def __init__(self, hidden_size):
        super(Gate, self).__init__()
        self.hidden_size = hidden_size

        # Linear layers to generate gate values
        self.gate_linear = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # Calculate gate values (between 0 and 1)
        gate_values = torch.sigmoid(self.gate_linear(x))

        # Generate the output using the gate values
        output = self.output_linear(x)

        # Apply the gate to the output
        gated_output = gate_values * output

        return gated_output


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, feat_dim, device=None, learnable_q=False, pos_enc=None):
        super().__init__()

        self.learnable_q = learnable_q
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.dropout = nn.Dropout(dropout)
        self.q = torch.nn.Parameter(torch.ones([pf_dim, feat_dim, hid_dim])) if self.learnable_q else None

    def forward(self, src, src_mask=None):
        #  src = [batch size, src len, hid dim]
        #  src_mask = [batch size, 1, 1, src len]

        #  self attention
        if self.learnable_q:
            _src, _ = self.self_attention(self.q, src, src, src_mask)
        else:
            _src, _ = self.self_attention(src, src, src, src_mask)

        #  dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        #  src = [batch size, src len, hid dim]

        return src


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, node_num, dropout, feat_dim, learnable_q=False, pos_enc_type=None):
        super().__init__()

        self.learnable_q = learnable_q
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, hid_dim, dropout)
        self.dropout = nn.Dropout(dropout)

        self.pos_enc_type = pos_enc_type
        if pos_enc_type is not None:
            if pos_enc_type == 'index':
                self.pos_emb = PositionalEncoding(d_model=hid_dim, dropout=dropout, max_len=hid_dim)
            elif pos_enc_type == 'identity':
                self.pos_emb = IdentitylEncoding(d_model=hid_dim, node_num=node_num, dropout=dropout)
            elif pos_enc_type == 'contrast':
                self.pos_emb = ContrastPE(d_model=hid_dim, node_num=node_num, dropout=dropout)
            elif pos_enc_type[-4:] == 'lm':
                self.pos_emb = LMPE(d_model=hid_dim, node_num=node_num, dropout=dropout)
            elif pos_enc_type[-4:] == '_dis':
                self.pos_emb = DistancePE(name=pos_enc_type[:-4], d_model=hid_dim, dropout=dropout)
            else:
                self.pos_emb = CoordinatePE(name=pos_enc_type, d_model=hid_dim, dropout=dropout)

        self.q = torch.nn.Parameter(torch.ones([hid_dim, feat_dim, hid_dim])) if self.learnable_q else None

    def forward(self, src, src_mask=None, pos_enc=None):
        #  src = [batch size, src len, hid dim]
        #  src_mask = [batch size, 1, 1, src len]

        if pos_enc is not None and self.pos_enc_type is not None:
            if self.pos_enc_type == 'contrast':
                src = self.pos_emb(src, pos_enc)
            else:
                src = self.pos_emb(src)

        #  self attention
        if self.learnable_q:
            _src, _ = self.self_attention(self.q, src, src, src_mask)
        else:
            _src, _ = self.self_attention(src, src, src, src_mask)

        #  dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        #  src = [batch size, src len, hid dim]

        #  positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #  dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, no_params=False, learnable_q=False):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.no_params = no_params

        # d_model // h 仍然是要能整除，换个名字仍然意义不变
        assert hid_dim % n_heads == 0

        if not self.no_params:
            self.w_q = nn.Linear(hid_dim, hid_dim)
            self.w_k = nn.Linear(hid_dim, hid_dim)
            self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):

        scale = torch.sqrt(torch.FloatTensor([self.hid_dim // self.n_heads])).to(query.device)

        # Q,K,V计算与变形：
        bsz = query.shape[0]

        if not self.no_params:
            Q = self.w_q(query)
            K = self.w_k(key)
            V = self.w_v(value)
        else:
            Q = query
            K = key
            V = value

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale

        # 如果没有mask，就生成一个
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        attention = self.dropout(torch.softmax(energy, dim=-1))

        # 第三步，attention结果与V相乘

        x = torch.matmul(attention, V)

        # 最后将多头排列好，就是multi-head attention的结果了

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        return x, attention.squeeze()


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #  x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        #  x = [batch size, seq len, pf dim]
        x = self.fc_2(x)
        #  x = [batch size, seq len, hid dim]

        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class IdentitylEncoding(nn.Module):
    "Implement the IdentitylEncoding function."
    def __init__(self, d_model, node_num, dropout):
        super(IdentitylEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.node_identity = nn.Parameter(torch.zeros(node_num, d_model), requires_grad=True)
        self.mlp = nn.Linear(d_model, d_model)

        # self.node_identity = nn.Parameter(torch.zeros(node_num, 1), requires_grad=True)
        # self.node_identity = nn.Parameter(torch.zeros(node_num, node_num), requires_grad=True)
        nn.init.kaiming_normal_(self.node_identity)

    def forward(self, x):
        bz, _, _, = x.shape
        # identity_mat = torch.mm(self.node_identity, self.node_identity.T)
        # emb = identity_mat.expand(bz, *identity_mat.shape)
        # x = x + emb
        pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
        #emb = torch.cat([x, pos_emb], dim=-1)
        emb = x + pos_emb 
        x = x + self.mlp(emb)
        #x = self.mlp(x + pos_emb)
        #return x
        return self.dropout(x)


class LMPE(nn.Module):
    def __init__(self, emb_dim, d_model, node_num, dropout):
        super(LMPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        node_num2emb_path = {
            116: 'data/prompts/ROI_prompts/AAL116_short_datatoken.pt'
        }

        self.init_emb = torch.stack(torch.load(node_num2emb_path[node_num])).squeeze()

        # Compute the positional encodings once in log space.
        self.mlp = nn.Linear(emb_dim, d_model)

    def forward(self, x):
        bz, _, _, = x.shape
        device = x.device
        pos_emb = self.mlp(self.init_emb.to(device)).repeat(bz, 1, 1)
        emb = x + pos_emb
        return self.dropout(emb)


class CoordinatePE(nn.Module):
    "Implement the CoordinatePE function."
    def __init__(self, name, d_model, dropout):
        super(CoordinatePE, self).__init__()
        self.coor = torch.from_numpy(get_3d_corr(name)).float()
        in_dim = self.coor.shape[-1]
        self.mlp = nn.Linear(in_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.mlp(self.coor.to(x.device))
        return self.dropout(x)


class DistancePE(nn.Module):
    "Implement the DistancePE function."
    def __init__(self, name, d_model, dropout):
        super(DistancePE, self).__init__()
        self.coor = torch.from_numpy(get_3d_corr(name)).float()
        self.distance = torch.cdist(self.coor, self.coor, p=2)
        in_dim = self.distance.shape[-1]
        self.mlp = nn.Linear(in_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.mlp(self.distance.to(x.device))
        return self.dropout(x)


class ContrastPE(nn.Module):
    "Implement the ContrastPE function."
    def __init__(self,  d_model, node_num, dropout):
        super(ContrastPE, self).__init__()
        self.mlp = nn.Linear(node_num, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, pos_enc):
        x = x + self.mlp(pos_enc)
        return self.dropout(x)
