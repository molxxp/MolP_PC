import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from torch import Tensor
from typing import Optional


class Extract3DFeatures(nn.Module):
    def __init__(self, node_feats_dim, edge_feats_dim, node_mlp_hidden_dim, edge_mlp_hidden_dim, dropout=0.2,
                 edge_attr_dim=0, update_coors=False, update_feats=True, soft_edge=True, ):
        super(Extract3DFeatures, self).__init__()
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.soft_edge = soft_edge

        edge_feats_dim += edge_attr_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_feats_dim + node_mlp_hidden_dim, node_mlp_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(node_mlp_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(node_mlp_hidden_dim, node_feats_dim)

        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(node_feats_dim * 2 + edge_feats_dim, edge_mlp_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(edge_mlp_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(edge_mlp_hidden_dim, edge_mlp_hidden_dim)
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(edge_mlp_hidden_dim, 1),
            nn.ReLU()

        )

        if self.soft_edge:
            self.edge_weight = nn.Sequential(
                nn.Linear(edge_mlp_hidden_dim, 1),
                nn.Sigmoid()
            )


        self.node_norm = nn.LayerNorm(node_feats_dim)
        self.coors_norm = CoorsNorm(scale_init=1e-2)

    def forward(self, graph, x, coords, edge_attr=None):

        edge_src, edge_dst = graph.edges()
        rel_coors = coords[edge_src] - coords[edge_dst]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)
        dist = torch.sqrt(rel_dist + 1e-8)
        rbf_vals = [torch.exp(-(dist ** 2) / (2 * sigma ** 2)) for sigma in [0.1, 0.5, 1.0]]
        rel_dist=torch.stack(rbf_vals, dim=-1).squeeze(1)

        if edge_attr is not None:
            if edge_attr.dim()==1:
                edge_attr=edge_attr.unsqueeze(1)
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist


        graph.edata['edge_attr'] = edge_attr_feats

        graph.ndata['x'] = x
        graph.ndata['coords'] = coords

        graph.update_all(self.message_func, self.reduce_func)
        if self.update_coors:

            return graph.ndata['h'], graph.ndata['coords']
        else:
            return graph.ndata['h']

    def message_func(self, edges):

        src_feats = edges.src['x']
        dst_feats = edges.dst['x']
        edge_feats = edges.data['edge_attr']

        m_ij = self.edge_mlp(torch.cat([src_feats, dst_feats, edge_feats], dim=-1))


        if self.soft_edge:
            m_ij = m_ij * self.edge_weight(m_ij)

        return {'msg': m_ij}

    def reduce_func(self, nodes):



        aggregated_msg = torch.mean(nodes.mailbox['msg'], dim=1)

        if self.update_feats:
            node_feats = nodes.data['x']
            hidden_out = self.node_mlp(torch.cat([node_feats, aggregated_msg], dim=-1))
            hidden_feats = self.node_norm(hidden_out)
            hidden_out = node_feats + hidden_feats
        else:
            hidden_out = nodes.data['x']

        nodes.data['h'] = hidden_out


        if self.update_coors:
            coors_wij = self.coors_mlp(aggregated_msg)
            rel_coors = nodes.data['coords']
            normed_coors = self.coors_norm(rel_coors)
            coors_out = rel_coors + coors_wij * normed_coors
        else:
            coors_out = nodes.data['coords']

        nodes.data['coords'] = coors_out

        return nodes.data


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale
