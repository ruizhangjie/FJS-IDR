import torch
from torch import nn
import torch.nn.functional as F


class GATop(nn.Module):
    # 工序节点嵌入
    def __init__(self, W_sizes_op, hidden_size_op, out_size_op, negative_slope=0.2):
        '''
        :param W_sizes_op: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_op: hidden dimensions of the MLPs
        :param out_size_op: dimension of the embedding of operation nodes
        '''
        super(GATop, self).__init__()
        self.in_sizes_op = W_sizes_op
        self.hidden_size_op = hidden_size_op
        self.out_size_op = out_size_op
        self.gnn_layers = nn.ModuleList()
        self.attn_pre = nn.Parameter(torch.rand(size=(1, 1, out_size_op), dtype=torch.float))
        self.attn_sub = nn.Parameter(torch.rand(size=(1, 1, out_size_op), dtype=torch.float))
        self.attn_self = nn.Parameter(torch.rand(size=(1, 1, out_size_op), dtype=torch.float))
        self.attn_ma = nn.Parameter(torch.rand(size=(1, 1, out_size_op), dtype=torch.float))
        for i in range(len(self.in_sizes_op)):
            self.gnn_layers.append(MLPs(self.in_sizes_op[i], self.out_size_op, self.hidden_size_op))
        self.gnn_layers.append(MLPs(self.out_size_op, self.out_size_op, self.hidden_size_op))
        self.leaky_relu = nn.LeakyReLU(negative_slope)  # 是一种改进的 ReLU激活函数 negative_slope指定斜率
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(
            'relu')  # 用来计算激活函数增益的函数 在网络初始化时用于缩放权重的值。激活函数的增益决定了每一层输出的方差是否保持不变，即网络是否具有相同的信号强度和方差

        nn.init.xavier_normal_(self.attn_pre, gain=gain)
        nn.init.xavier_normal_(self.attn_sub, gain=gain)
        nn.init.xavier_normal_(self.attn_self, gain=gain)
        nn.init.xavier_normal_(self.attn_ma, gain=gain)

    def forward(self, op_ma_adj_batch, op_pre_adj_batch, op_sub_adj_batch, batch_idxes, feats):
        '''
        :param op_ma_adj_batch: Adjacency matrix of operation and machine nodes
        :param op_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param op_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_idxes: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        '''
        op = feats[0]
        ma = feats[1]
        eg = feats[2]
        num = op.shape[1]
        # Identity matrix for self-loop of nodes
        self_adj = torch.eye(feats[0].size(-2),
                             dtype=torch.int64).unsqueeze(0).expand_as(op_pre_adj_batch[batch_idxes])
        # Calculate a return operation embedding
        pre_f1 = self.gnn_layers[1](op, op_pre_adj_batch[batch_idxes])  # (20,num,18)
        sub_f1 = self.gnn_layers[2](op, op_sub_adj_batch[batch_idxes])  # (20,num,18)
        self_f1 = self.gnn_layers[3](op, self_adj)  # (20,num,18)
        ma_eg = torch.cat((op_ma_adj_batch[batch_idxes].unsqueeze(-1) * ma.unsqueeze(1), eg), dim=-1)  # (20,mum,ma,12)
        ma_f1 = self.gnn_layers[0](ma_eg, op_ma_adj_batch[batch_idxes], 1)  # (20,num,ma,18)
        ma_f2 = self.gnn_layers[4](ma_f1, op_ma_adj_batch[batch_idxes], 2)  # (20,num,18)
        pre_f1_ut = (pre_f1 * self.attn_pre).sum(dim=-1).unsqueeze(-1)  # (20,num,1)
        sub_f1_ut = (sub_f1 * self.attn_pre).sum(dim=-1).unsqueeze(-1)  # (20,num,1)
        self_f1_ut = (self_f1 * self.attn_pre).sum(dim=-1).unsqueeze(-1)  # (20,num,1)
        ma_f2_ut = (ma_f2 * self.attn_pre).sum(dim=-1).unsqueeze(-1)  # (20,num,1)
        e_pre = self.leaky_relu(pre_f1_ut + self_f1_ut)
        e_sub = self.leaky_relu(sub_f1_ut + self_f1_ut)
        e_self = self.leaky_relu(self_f1_ut + self_f1_ut)
        e_ma = self.leaky_relu(ma_f2_ut + self_f1_ut)
        b = torch.cat((e_pre, e_sub, e_self, e_ma), dim=1)  # (20,4*num,1)
        alpha = F.softmax(b, dim=1)
        alpha_pre = alpha[:, :num, :]
        alpha_sub = alpha[:, num:num * 2, :]
        alpha_self = alpha[:, num * 2:num * 3, :]
        alpha_ma = alpha[:, num * 3:num * 4, :]
        pre_embed = alpha_pre * pre_f1
        sub_embed = alpha_sub * sub_f1
        self_embed = alpha_self * self_f1
        ma_embed = alpha_ma * ma_f2
        g = torch.sigmoid(pre_embed + sub_embed + self_embed + ma_embed)
        return g


class MLPs(nn.Module):
    # 聚合四类信息
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 negative_slope=0.2):
        '''
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        '''
        super(MLPs, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),  # ELU激活函数是一种具有负半段的非线性函数，它可以通过保留一些负值来避免ReLU 函数的“神经元死亡”问题 参数alpha控制负半段的斜率，默认为1.0
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats),
        )

    def forward(self, feat, adj, flag_ma=0):
        if flag_ma == 1:
            a = adj.unsqueeze(-1) * feat
            c = self.project(a)
        elif flag_ma == 0:
            a = adj.unsqueeze(-1) * feat.unsqueeze(-3)
            b = torch.sum(a, dim=-2)
            c = self.project(b)
        elif flag_ma == 2:
            a = torch.sum(feat, dim=-2)
            b = torch.sum(feat != 0., dim=-2)
            c = self.project(a / b)
        else:
            a = adj.unsqueeze(-1) * feat.unsqueeze(-2)
            b = torch.sum(a, dim=1)
            c = self.project(b)
        return c


class MLPma(nn.Module):
    # 机器节点嵌入
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_size_ma
                 ):
        '''
        :param in_feats: tuple, input dimension of (operation node, machine node)
        :param out_feats: Dimension of the output (machine embedding)
        :param num_head: Number of heads
        '''
        super(MLPma, self).__init__()
        self.in_sizes = in_feats
        self._out_feats = out_feats
        self.gnn_layers = nn.ModuleList()

        for i in range(len(self.in_sizes)):
            self.gnn_layers.append(MLPs(self.in_sizes[i], self._out_feats, hidden_size_ma))
        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self._out_feats * len(self.in_sizes), hidden_size_ma),
            nn.ELU(),
            nn.Linear(hidden_size_ma, hidden_size_ma),
            nn.ELU(),
            nn.Linear(hidden_size_ma, self._out_feats),
        )

    def forward(self, op_ma_adj_batch, batch_idxes, feat):
        h = (feat[0], feat[1])
        self_adj = torch.eye(feat[1].size(-2),
                             dtype=torch.int64).unsqueeze(0).expand(op_ma_adj_batch[batch_idxes].shape[0], -1, -1)
        adj = (op_ma_adj_batch[batch_idxes], self_adj)
        MLP_embeddings = []
        MLP_embeddings.append(self.gnn_layers[0](h[0], adj[0], 3))
        MLP_embeddings.append(self.gnn_layers[1](h[1], adj[1]))
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        mu_ij_prime = self.project(MLP_embedding_in)
        return mu_ij_prime
