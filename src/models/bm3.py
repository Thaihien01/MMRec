# coding: utf-8
# @email: enoche.chow@gmail.com
r"""

################################################
paper:  Bootstrap Latent Representations for Multi-modal Recommendation
https://arxiv.org/abs/2207.05969
"""
import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class BM3(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BM3, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']

        # config flags for improvements
        self.gated_fusion = config['gated_fusion'] if 'gated_fusion' in config else True
        self.multimodal_gcn = config['multimodal_gcn'] if 'multimodal_gcn' in config else True
        self.mlp_predictor = config['mlp_predictor'] if 'mlp_predictor' in config else True
        self.separate_predictors = config['separate_predictors'] if 'separate_predictors' in config else True
        self.mlp_features = config['mlp_features'] if 'mlp_features' in config else True

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.cf_predictor = self.build_predictor(self.embedding_dim, self.mlp_predictor)
        self.cf_predictor.apply(self._init_weights)

        self.reg_loss = EmbLoss()

        if self.separate_predictors:
            if self.t_feat is not None:
                self.text_predictor = self.build_predictor(self.embedding_dim, self.mlp_predictor)
                self.text_predictor.apply(self._init_weights)
            if self.v_feat is not None:
                self.image_predictor = self.build_predictor(self.embedding_dim, self.mlp_predictor)
                self.image_predictor.apply(self._init_weights)
        else:
            self.text_predictor = self.cf_predictor
            self.image_predictor = self.cf_predictor

        if self.gated_fusion:
            if self.v_feat is not None:
                self.v_gate = nn.Sequential(
                    nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                    nn.Sigmoid()
                )
                self.v_gate.apply(self._init_weights)
            if self.t_feat is not None:
                self.t_gate = nn.Sequential(
                    nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                    nn.Sigmoid()
                )
                self.t_gate.apply(self._init_weights)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if self.mlp_features:
                self.image_trs = nn.Sequential(
                    nn.Linear(self.v_feat.shape[1], self.feat_embed_dim * 2),
                    nn.LayerNorm(self.feat_embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
                )
            else:
                self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_trs.apply(self._init_weights)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if self.mlp_features:
                self.text_trs = nn.Sequential(
                    nn.Linear(self.t_feat.shape[1], self.feat_embed_dim * 2),
                    nn.LayerNorm(self.feat_embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
                )
            else:
                self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_trs.apply(self._init_weights)

    def build_predictor(self, dim, use_mlp):
        if use_mlp:
            return nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
        else:
            return nn.Linear(dim, dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def get_norm_adj_mat(self, interaction_matrix):
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()

        # Combine user-item and item-user interactions into a single COO matrix
        row = np.concatenate([inter_M.row, inter_M_t.row + self.n_users])
        col = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
        data = np.ones_like(row, dtype=np.float32)

        A = sp.coo_matrix((data, (row, col)),
                           shape=(self.n_users + self.n_items, self.n_users + self.n_items))

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self):
        h = self.item_id_embedding.weight

        if self.multimodal_gcn:
            item_feats = h
            if self.t_feat is not None:
                t_feat = self.text_trs(self.text_embedding.weight)
                if self.gated_fusion:
                    t_gate_val = self.t_gate(torch.cat([h, t_feat], dim=-1))
                    item_feats = item_feats + t_gate_val * t_feat
                else:
                    item_feats = item_feats + t_feat
            if self.v_feat is not None:
                v_feat = self.image_trs(self.image_embedding.weight)
                if self.gated_fusion:
                    v_gate_val = self.v_gate(torch.cat([h, v_feat], dim=-1))
                    item_feats = item_feats + v_gate_val * v_feat
                else:
                    item_feats = item_feats + v_feat
            ego_embeddings = torch.cat((self.user_embedding.weight, item_feats), dim=0)
        else:
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)

        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def calculate_loss(self, interactions):
        # online network
        u_online_ori, i_online_ori = self.forward()
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online, i_online = self.cf_predictor(u_online_ori), self.cf_predictor(i_online_ori)

        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = self.text_predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None:
            v_feat_online = self.image_predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        return (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
               self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online = self.forward()
        u_online, i_online = self.cf_predictor(u_online), self.cf_predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui

