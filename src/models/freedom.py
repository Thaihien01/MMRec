# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
FREEDOM++: Enhanced Freezing and Denoising Graph Structures with Multimodal Contrastive Alignment & Dynamic Attention
"""

import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss
from utils.utils import build_sim, compute_normalized_laplacian


class FREEDOM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FREEDOM, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']

        # Advanced configurations (FREEDOM++)
        self.reg_weight_l2 = config['reg_weight_l2'] if 'reg_weight_l2' in config else 0.0001
        self.cl_weight = config['cl_weight'] if 'cl_weight' in config else 0.1
        self.mm_edge_dropout = config['mm_edge_dropout'] if 'mm_edge_dropout' in config else 0.0
        self.mlp_features = config['mlp_features'] if 'mlp_features' in config else True
        self.gated_fusion = config['gated_fusion'] if 'gated_fusion' in config else True
        self.dynamic_attention = config['dynamic_attention'] if 'dynamic_attention' in config else True
        self.residual_gcn = config['residual_gcn'] if 'residual_gcn' in config else True
        self.rebuild_mm_adj = config['rebuild_mm_adj'] if 'rebuild_mm_adj' in config else False

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.reg_loss = EmbLoss()

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

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

        # Dynamic Modality Attention Networks
        if self.dynamic_attention:
            if self.v_feat is not None:
                self.v_att = nn.Sequential(
                    nn.Linear(self.feat_embed_dim, self.feat_embed_dim // 2),
                    nn.Tanh(),
                    nn.Linear(self.feat_embed_dim // 2, 1)
                )
                self.v_att.apply(self._init_weights)
            if self.t_feat is not None:
                self.t_att = nn.Sequential(
                    nn.Linear(self.feat_embed_dim, self.feat_embed_dim // 2),
                    nn.Tanh(),
                    nn.Linear(self.feat_embed_dim // 2, 1)
                )
                self.t_att.apply(self._init_weights)

        if self.gated_fusion:
            self.combine_gate = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.Sigmoid()
            )
            self.combine_gate.apply(self._init_weights)

        # Move feature embeddings to model device before constructing kNN graph
        if self.v_feat is not None:
            self.image_embedding = self.image_embedding.to(self.device)
            self.image_trs = self.image_trs.to(self.device)
        if self.t_feat is not None:
            self.text_embedding = self.text_embedding.to(self.device)
            self.text_trs = self.text_trs.to(self.device)

        # Construct or load item kNN graph using projected feature embeddings
        if os.path.exists(mm_adj_file) and not self.rebuild_mm_adj:
            self.mm_adj = torch.load(mm_adj_file, map_location=self.device)
        else:
            if self.v_feat is not None:
                v_proj = self.image_trs(self.image_embedding.weight).detach()
                indices, image_adj = self.get_knn_adj_mat(v_proj)
                self.mm_adj = image_adj
            if self.t_feat is not None:
                t_proj = self.text_trs(self.text_embedding.weight).detach()
                indices, text_adj = self.get_knn_adj_mat(t_proj)
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        self.mm_adj = self.mm_adj.to(self.device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def sparse_dropout(self, x, keep_prob):
        if not self.training or keep_prob >= 1.0 or keep_prob <= 0.0:
            return x
        x = x.coalesce()
        rc = x.indices()
        val = x.values()
        noise_shape = val.shape[0]
        mask = (torch.rand(noise_shape, device=val.device) + keep_prob).floor().type(torch.bool)
        val = val[mask] / keep_prob
        rc = rc[:, mask]
        return torch.sparse.FloatTensor(rc, val, x.shape).coalesce().to(x.device)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True) + 1e-7)
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0], device=mm_embeddings.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size).to(indices.device)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size).to(indices.device)

    def get_norm_adj_mat(self):
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()

        row = np.concatenate([inter_M.row, inter_M_t.row + self.n_users])
        col = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
        data = np.ones_like(row, dtype=np.float32)

        A = sp.coo_matrix((data, (row, col)),
                           shape=(self.n_users + self.n_items, self.n_users + self.n_items))

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        keep_indices = self.edge_indices[:, degree_idx]
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        h = self.item_id_embedding.weight
        mm_adj = self.mm_adj
        if mm_adj.device != h.device:
            mm_adj = mm_adj.to(h.device)
            self.mm_adj = mm_adj
        if adj.device != h.device:
            adj = adj.to(h.device)

        if self.training and self.mm_edge_dropout > 0:
            mm_adj = self.sparse_dropout(mm_adj, 1.0 - self.mm_edge_dropout)

        # Multimodal graph convolution with residual connection
        for i in range(self.n_layers):
            h_next = torch.sparse.mm(mm_adj, h)
            if self.residual_gcn:
                h = h + F.elu(h_next)
            else:
                h = h_next

        # Dynamic Item-level Attention Fusion across modalities
        if self.dynamic_attention and (self.v_feat is not None or self.t_feat is not None):
            mm_embeds = []
            att_weights = []
            if self.v_feat is not None:
                v_proj = self.image_trs(self.image_embedding.weight)
                v_score = self.v_att(v_proj)
                mm_embeds.append(v_proj)
                att_weights.append(v_score)
            if self.t_feat is not None:
                t_proj = self.text_trs(self.text_embedding.weight)
                t_score = self.t_att(t_proj)
                mm_embeds.append(t_proj)
                att_weights.append(t_score)
            
            if len(mm_embeds) > 1:
                att_scores = torch.cat(att_weights, dim=-1)
                att_probs = F.softmax(att_scores, dim=-1)
                mm_fused = att_probs[:, 0:1] * mm_embeds[0] + att_probs[:, 1:2] * mm_embeds[1]
            else:
                mm_fused = mm_embeds[0]
            
            h = h + F.normalize(mm_fused, p=2, dim=-1)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        if self.gated_fusion:
            h_norm = F.normalize(h, p=2, dim=-1)
            gate = self.combine_gate(torch.cat([i_g_embeddings, h_norm], dim=-1))
            fused_i_embeddings = i_g_embeddings + gate * h_norm
        else:
            fused_i_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=-1)

        return u_g_embeddings, fused_i_embeddings

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        mf_v_loss, mf_t_loss = 0.0, 0.0
        cl_loss = 0.0

        if self.t_feat is not None:
            text_feats = F.normalize(self.text_trs(self.text_embedding.weight), p=2, dim=-1)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
            # Contrastive Alignment: Item Graph Embedding vs Text Feature
            cl_t = 1.0 - F.cosine_similarity(pos_i_g_embeddings, text_feats[pos_items], dim=-1).mean()
            cl_loss = cl_loss + cl_t

        if self.v_feat is not None:
            image_feats = F.normalize(self.image_trs(self.image_embedding.weight), p=2, dim=-1)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])
            # Contrastive Alignment: Item Graph Embedding vs Image Feature
            cl_v = 1.0 - F.cosine_similarity(pos_i_g_embeddings, image_feats[pos_items], dim=-1).mean()
            cl_loss = cl_loss + cl_v

        if self.t_feat is not None and self.v_feat is not None:
            # Cross-modal Contrastive Alignment: Image Feature vs Text Feature
            cl_vt = 1.0 - F.cosine_similarity(image_feats[pos_items], text_feats[pos_items], dim=-1).mean()
            cl_loss = cl_loss + cl_vt

        # L2 Regularization Loss
        reg_loss = 0.0
        if self.reg_weight_l2 > 0:
            u_ego = self.user_embedding(users)
            pos_i_ego = self.item_id_embedding(pos_items)
            neg_i_ego = self.item_id_embedding(neg_items)
            reg_loss = self.reg_loss(u_ego, pos_i_ego, neg_i_ego)

        return batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss) + self.cl_weight * cl_loss + self.reg_weight_l2 * reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
