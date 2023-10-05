"""
@author: Yyujie
@email: yyj17320071233@163.com
@Date: 2023/3/26 11:07
@Description
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from uitls import re_normalization, relations_to_matrix


class LayerGCN(nn.Module):
    def __init__(self, lncRNA_num, disease_num, miRNA_num, latent_dim, n_layers, dropout):
        super(LayerGCN, self).__init__()
        self.lncRNA_num = lncRNA_num
        self.disease_num = disease_num
        self.miRNA_num = miRNA_num
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.weight = nn.Parameter(torch.Tensor(self.latent_dim, self.latent_dim))
        self.reset_parameters()

        self.lncRNA_embeddings = LncRNAEmbedding(self.lncRNA_num, self.latent_dim, dropout)
        self.disease_embeddings = DiseaseEmbedding(self.disease_num, self.latent_dim, dropout)
        self.miRNA_embeddings = MirRNAEmbedding(self.miRNA_num, self.latent_dim, dropout)

    def reset_parameters(self):
        # 神经网络权重初始化
        init.kaiming_uniform_(self.weight)

    def forward(self, A_stack, lnc_sim, dis_sim, miR_sim):
        ego_embeddings_tmp = torch.cat([self.lncRNA_embeddings(lnc_sim), self.disease_embeddings(dis_sim)], 0)
        ego_embeddings = torch.cat([ego_embeddings_tmp, self.miRNA_embeddings(miR_sim)], 0)
        # A：只有用作训练的数据中的正样本为1，其余均为0
        all_layer_embedding = []

        A_ = re_normalization(A_stack).type(torch.FloatTensor)

        x_embeddings = ego_embeddings
        for layer_index in range(self.n_layers):
            layer_embeddings = F.relu(torch.sparse.mm(A_, torch.mm(x_embeddings, self.weight)))  # 做矩阵乘法运算  A
            _weights = F.cosine_similarity(layer_embeddings, ego_embeddings, dim=-1)
            x_embeddings = layer_embeddings
            layer_embeddings = torch.einsum('k,kj->kj', _weights, layer_embeddings)  # 矩阵乘法
            all_layer_embedding.append(layer_embeddings)

        all_embeddings = torch.mean(torch.stack(all_layer_embedding, dim=0), dim=0)  # 注意力向量，维度同层数
        ld_all_embeddings, m_all_embeddings = torch.split(all_embeddings, [self.lncRNA_num + self.disease_num, self.miRNA_num])
        l_all_embeddings, d_all_embeddings = torch.split(ld_all_embeddings, [self.lncRNA_num, self.disease_num])
        predict_score = torch.mm(l_all_embeddings, d_all_embeddings.t())
        return predict_score


class DiseaseEmbedding(nn.Module):
    def __init__(self, in_feats, out_feats, dropout):
        super(DiseaseEmbedding, self).__init__()
        seq_d = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.repr = seq_d

    def forward(self, in_feat):
        # Generate disease ego_embeddings
        disease_ego_embeddings = self.repr(in_feat)

        return disease_ego_embeddings


class LncRNAEmbedding(nn.Module):
    def __init__(self, in_feats, out_feats, dropout):
        super(LncRNAEmbedding, self).__init__()

        seq_l = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.repr = seq_l

    def forward(self, in_feat):
        # Generate lncRNA ego_embeddings
        lncRNA_ego_embeddings = self.repr(in_feat)

        return lncRNA_ego_embeddings


class MirRNAEmbedding(nn.Module):
    def __init__(self, in_feats, out_feats, dropout):
        super(MirRNAEmbedding, self).__init__()

        seq_m = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.repr = seq_m

    def forward(self, in_feat):
        # Generate mirRNA ego_embeddings
        mirRNA_ego_embeddings = self.repr(in_feat)

        return mirRNA_ego_embeddings
