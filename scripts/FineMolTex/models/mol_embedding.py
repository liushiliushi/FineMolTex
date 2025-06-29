import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_scatter import scatter


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(5, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(3, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_attr != None:
            # add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:, 0] = 4  # bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

            edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        else:
            edge_embeddings = torch.zeros((edge_index[0].shape[1], x.shape[-1])).to(x.device)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GIN(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0, atom=True):
        super(GIN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if atom:
            self.x_embedding1 = torch.nn.Embedding(120, emb_dim)
            self.x_embedding2 = torch.nn.Embedding(3, emb_dim)

            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        else:
            self.embedding = nn.Embedding(908, emb_dim)

        self.atom = atom

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        if self.atom:
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        else:
            x = self.embedding(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        node_representation = h_list[-1]

        return node_representation


class MolEmbedding(torch.nn.Module):
    def __init__(self,
                 emb_dim=300,
                 num_gnn_layers=5,
                 dropout=0.0):
        super(MolEmbedding, self).__init__()

        self.gnn = GIN(num_layer=num_gnn_layers,
                       emb_dim=emb_dim,
                       drop_ratio=dropout,
                       atom=True)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),  # projection
        )

        self.frag_pred = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),  # projection
            nn.Linear(emb_dim, 768)
        )

        self.tree_pred = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),  # projection
            nn.Linear(emb_dim, 3000)
        )

        self.pool = global_mean_pool

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = self.gnn(x, edge_index, edge_attr)
        atom_embedding = self.proj(x)
        atom_sum_embedding = scatter(atom_embedding, data.map, dim=0, reduce="mean")
        # frag_pred = self.frag_pred(atom_sum_embedding)
        frag_pred = atom_sum_embedding
        # return atom_embedding, frag_pred, self.pool(self.tree_pred(x), batch)
        return frag_pred


class FragEmbedding(torch.nn.Module):
    def __init__(self,
                 emb_dim=300,
                 num_gnn_layers=5,
                 dropout=0.0):
        super(FragEmbedding, self).__init__()

        self.gnn = GIN(num_layer=num_gnn_layers,
                       emb_dim=emb_dim,
                       drop_ratio=dropout,
                       atom=False)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x, edge_index, edge_attr=None):
        x = self.gnn(x, edge_index, edge_attr)
        return self.proj(x)

