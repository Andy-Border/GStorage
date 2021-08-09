import torch as th
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GATConv, SGConv, SAGEConv#, GCN2Conv
import torch.nn.functional as F
from utils.data_utils import mp_to_relations
from itertools import combinations
import torch.nn as nn
import os
from utils.data_utils import *
import utils.util_funcs as uf
from models.GSR.config import GSRConfig
from utils.data_utils import *
import numpy as np
from utils.util_funcs import init_random_state
from copy import deepcopy
import dgl

#
class GSR_pretrain(nn.Module):
    def __init__(self, g, cf: GSRConfig):
        # ! Initialize variabless
        super(GSR_pretrain, self).__init__()
        self.__dict__.update(cf.model_conf)
        init_random_state(cf.seed)
        self.device = cf.device
        self.g = g

        # ! Encoder: Pretrained GNN Modules
        self.views = views = ['F', 'S']
        self.encoder = nn.ModuleDict({
            src_view: self._get_encoder(src_view, cf) for src_view in views})
        # ! Decoder: Cross-view MLP Mappers
        self.decoder = nn.ModuleDict(
            {f'{src_view}->{tgt_view}':
                 MLP(n_layer=self.decoder_layer,
                     input_dim=self.n_hidden,
                     n_hidden=self.decoder_n_hidden,
                     output_dim=self.n_hidden, dropout=0,
                     activation=nn.ELU(),
                     )
             for tgt_view in views
             for src_view in views if src_view != tgt_view})

    def _get_encoder(self, src_view, cf):
        input_dim = cf.feat_dim[src_view]
        if cf.gnn_model == 'GCN':
            # GCN emb should not be dropped out
            return TwoLayerGCN(input_dim, cf.n_hidden, cf.n_hidden, cf.activation, cf.pre_dropout, is_out_layer=True)
        if cf.gnn_model == 'GAT':
            return TwoLayerGAT(input_dim, cf.gat_hidden, cf.gat_hidden, cf.in_head, cf.in_head, cf.activation, cf.pre_dropout, cf.pre_dropout,  is_out_layer=True)
        if cf.gnn_model == 'GraphSage':
            return TwoLayerGraphSage(input_dim, cf.n_hidden, cf.n_hidden, cf.aggregator, cf.activation, cf.pre_dropout, is_out_layer=True)

    def forward(self, edge_subgraph, blocks, input, mode='q'):
        def _get_emb(x):
            # Query embedding is stored in source nodes, key embedding in target
            q_nodes, k_nodes = edge_subgraph.edges()
            return x[q_nodes] if mode == 'q' else x[k_nodes]

        # ! Step 1: Encode node properties to embeddings
        Z = {src_view: _get_emb(encoder(blocks, input[src_view], stochastic=True))
             for src_view, encoder in self.encoder.items()}
        # ! Step 2: Decode embeddings if inter-view
        Z.update({dec: decoder(Z[dec[0]])
                  for dec, decoder in self.decoder.items()})
        return Z

    @uf.time_logger
    def refine_graph(self, g, feat):
        '''
        Find the neighborhood candidates for each candidate graph
        :param g: DGL graph
        '''

        # # ! Get Node Property
        emb = {_: self.encoder[_](g.to(self.device), feat[_].to(self.device), stochastic=False).detach()
               for _ in self.views}
        edges = set(graph_edge_to_lot(g))
        rm_num, add_num = [int(float(_) * self.g.num_edges())
                           for _ in (self.rm_ratio, self.add_ratio)]
        batched_implementation = True
        if batched_implementation:
            # if not self.fsim_norm or self.dataset in ['arxiv']:
            if self.device != th.device('cpu'):
                emb = {k: v.half() for k, v in emb.items()}
            edges = scalable_graph_refine(
                g, emb, rm_num, add_num, self.cos_batch_size, self.fsim_weight, self.device, self.fsim_norm)
        else:
            # ! Filter Graphs
            sim_mats = {v: cosine_similarity_n_space(np_.detach(), dist_batch_size=self.cos_batch_size)
                        for v, np_ in emb.items()}
            if self.fsim_norm:
                sim_mats = {v: min_max_scaling(sim_mat, type='global') for v, sim_mat in sim_mats.items()}
            sim_adj = self.fsim_weight * sim_mats['F'] + (1 - self.fsim_weight) * sim_mats['S']
            # ! Remove the lowest K existing edges.
            # Only the existing edges should be selected, other edges are guaranteed not to be selected with similairty 99
            if rm_num > 0:
                low_candidate_mat = th.ones_like(sim_adj) * 99
                low_candidate_mat[self.g.edges()] = sim_adj[self.g.edges()]
                low_inds = edge_lists_to_set(global_topk(low_candidate_mat, k=rm_num, largest=False))
                edges -= low_inds
            # ! Add the highest K from non-existing edges.
            # Exisiting edges and shouldn't be selected
            if add_num > 0:
                sim_adj.masked_fill_(th.eye(sim_adj.shape[0]).bool(), -1)
                sim_adj[self.g.edges()] = -1
                high_inds = edge_lists_to_set(global_topk(sim_adj, k=add_num, largest=True))
                edges |= high_inds
            uf.save_pickle(sorted(edges), 'EdgesGeneratedByOriImplementation')
        row_id, col_id = map(list, zip(*list(edges)))
        # print(f'High inds {list(high_inds)[:5]}')
        g_new = dgl.add_self_loop(
            dgl.graph((row_id, col_id), num_nodes=self.g.num_nodes())).to(self.device)
        # g_new.ndata['sim'] = sim_adj.to(self.device)
        return g_new


class GSR_finetune(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, cf: GSRConfig):
        # ! Initialize variables
        super(GSR_finetune, self).__init__()
        self.__dict__.update(cf.model_conf)
        init_random_state(cf.seed)
        if cf.dataset != 'arxiv':
            if cf.gnn_model == 'GCN':
                self.gnn = TwoLayerGCN(cf.n_feat, cf.n_hidden, cf.n_class, cf.activation, cf.dropout, is_out_layer=True)
            if cf.gnn_model == 'GAT':
                self.gnn = TwoLayerGAT(cf.n_feat, cf.gat_hidden, cf.n_class, cf.in_head, 1, cf.activation, cf.dropout, cf.dropout, is_out_layer=True)
            if cf.gnn_model == 'GraphSage':
                self.gnn = TwoLayerGraphSage(cf.n_feat, cf.n_hidden, cf.n_class, cf.aggregator, cf.activation, cf.dropout, is_out_layer=True)
        else:
            if cf.gnn_model == 'GCN':
                self.gnn = ThreeLayerGCN_BN(cf.n_feat, cf.n_hidden, cf.n_class, cf.activation, cf.dropout)
            if cf.gnn_model == 'GAT':
                self.gnn = ThreeLayerGAT_BN(cf.n_feat, cf.gat_hidden, cf.n_class, cf.in_head, 1, cf.activation, cf.input_dropout, cf.dropout)
            if cf.gnn_model == 'GraphSage':
                self.gnn = ThreeLayerGraphSage_BN(cf.n_feat, cf.n_hidden, cf.n_class, cf.activation, cf.dropout, cf.aggregator)


    def forward(self, g, x):
        return self.gnn(g, x)


class MLP(nn.Module):
    def __init__(self, n_layer, input_dim, output_dim, n_hidden, dropout=0.5, activation=th.nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(input_dim, n_hidden))
        # hidden layers
        for i in range(n_layer - 2):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, output_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        return

    def forward(self, input):
        h = input
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            # h = F.relu(layer(h))
            h = self.activation(layer(h))
        return h


class TwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, activation, dropout=0.5, is_out_layer=True):
        super().__init__()
        # Fixme: Deal zero degree
        self.conv1 = GraphConv(in_features, hidden_features, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_features, out_features, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(p=dropout)
        self.is_out_layer = is_out_layer
        self.activation = F.elu_ if activation == 'Elu' else F.relu


    def _stochastic_forward(self, blocks, x):
        x = self.activation(self.conv1(blocks[0], x))
        x = self.dropout(x)

        if self.is_out_layer:  # Last layer, no activation and dropout
            x = self.conv2(blocks[1], x)
        else:  # Middle layer, activate and dropout
            x = self.activation(self.conv2(blocks[1], x))
            x = self.dropout(x)
        return x

    def forward(self, g, x, stochastic=False):
        if stochastic:  # Batch forward
            return self._stochastic_forward(g, x)
        else:  # Normal forward
            x = self.activation(self.conv1(g, x))
            x = self.dropout(x)
            if self.is_out_layer:  # Last layer, no activation and dropout
                x = self.conv2(g, x)
            else:  # Middle layer, activate and dropout
                x = self.activation(self.conv2(g, x))
                x = self.dropout(x)
            return x


class ThreeLayerGCN_BN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, activation, dropout=0.5):
        super().__init__()
        # Fixme: Deal zero degree
        self.conv1 = GraphConv(in_features, hidden_features, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_features, hidden_features, allow_zero_in_degree=True)
        self.conv3 = GraphConv(hidden_features, out_features, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_features)
        self.bn2 = torch.nn.BatchNorm1d(hidden_features)
        self.activation = F.elu_ if activation == 'Elu' else F.relu

    def forward(self, g, x):
        x = self.activation(self.bn1(self.conv1(g, x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.conv2(g, x)))
        x = self.dropout(x)
        x = self.conv3(g, x)
        return x



class TwoLayerGAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, in_head, out_head, activation,
                 feat_drop=0.6, attn_drop=0.6, negative_slope=0.2, residual=False, is_out_layer=True):
        super().__init__()
        # Fixme: Deal zero degree
        heads = [in_head, out_head]
        self.conv1 = GATConv(in_features, hidden_features, heads[0], residual=residual,  allow_zero_in_degree=True)
        self.conv2 = GATConv(hidden_features*heads[0], out_features, heads[-1], residual=residual, allow_zero_in_degree=True)
        self.is_out_layer = is_out_layer
        self.activation = F.elu_ if activation == 'Elu' else F.relu
        self.input_dropout = nn.Dropout(p=feat_drop)
        self.dropout = nn.Dropout(p=attn_drop)


    def _stochastic_forward(self, blocks, x):
        x = self.input_dropout(x)
        x = self.conv1(blocks[0], x).flatten(1)
        if self.is_out_layer:  # Last layer, no activation and dropout
            x = self.conv2(blocks[1], x).flatten(1)
        else:  # Middle layer, activate and dropout
            x = self.activation(self.conv2(blocks[1], x).flatten(1))
            x = self.dropout(x)
        return x

    def forward(self, g, x, stochastic=False):
        if stochastic:  # Batch forward
            return self._stochastic_forward(g, x)
        else:  # Normal forward
            x = self.input_dropout(x)
            x = self.conv1(g, x).flatten(1)
            if self.is_out_layer:  # Last layer, no activation and dropout
                x = self.conv2(g, x).flatten(1)
            else:  # Middle layer, activate and dropout
                x = self.activation(self.conv2(g, x).flatten(1))
                x = self.dropout(x)
            return x



class ThreeLayerGAT_BN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, in_head, out_head, activation,
                 feat_drop=0.6, attn_drop=0.6, negative_slope=0.2, residual=False):
        super().__init__()
        # Fixme: Deal zero degree
        heads = [in_head, in_head, out_head]
        self.conv1 = GATConv(in_features, hidden_features, heads[0], residual=residual, allow_zero_in_degree=True)
        self.conv2 = GATConv(hidden_features*heads[0], hidden_features, heads[1], residual=residual, allow_zero_in_degree=True)
        self.conv3 = GATConv(hidden_features*heads[1], out_features, heads[-1], residual=residual, allow_zero_in_degree=True)
        self.bn1 = torch.nn.BatchNorm1d(hidden_features*in_head)
        self.bn2 = torch.nn.BatchNorm1d(hidden_features*in_head)
        self.input_drop = nn.Dropout(feat_drop)
        self.dropout = nn.Dropout(attn_drop)
        self.activation = F.elu_ if activation == 'Elu' else F.relu


    def forward(self, g, x):

        x = self.input_drop(x)
        x = self.conv1(g, x).flatten(1)
        x = self.bn1(x)
        x = self.dropout(self.activation(x))
        x = self.conv2(g, x).flatten(1)
        x = self.bn2(x)
        x = self.dropout(self.activation(x))
        x = self.conv3(g, x).mean(1)
        return x



class TwoLayerGraphSage(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, aggregator, activation, dropout=0.5, is_out_layer=True):
        super().__init__()
        # Fixme: Deal zero degree
        self.conv1 = SAGEConv(in_features, hidden_features, aggregator)
        self.conv2 = SAGEConv(hidden_features, out_features, aggregator)
        self.dropout = nn.Dropout(p=dropout)
        self.is_out_layer = is_out_layer
        self.activation = F.elu_ if activation == 'Elu' else F.relu


    def _stochastic_forward(self, blocks, x):
        x = self.activation(self.conv1(blocks[0], x))
        x = self.dropout(x)

        if self.is_out_layer:  # Last layer, no activation and dropout
            x = self.conv2(blocks[1], x)
        else:  # Middle layer, activate and dropout
            x = self.activation(self.conv2(blocks[1], x))
            x = self.dropout(x)
        return x

    def forward(self, g, x, stochastic=False):
        if stochastic:  # Batch forward
            return self._stochastic_forward(g, x)
        else:  # Normal forward
            x = self.activation(self.conv1(g, x))
            x = self.dropout(x)
            if self.is_out_layer:  # Last layer, no activation and dropout
                x = self.conv2(g, x)
            else:  # Middle layer, activate and dropout
                x = self.activation(self.conv2(g, x))
                x = self.dropout(x)
            return x


class ThreeLayerGraphSage_BN(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 activation,
                 dropout,
                 aggregator):
        super().__init__()

        self.conv1 = SAGEConv(in_features, hidden_features, aggregator)
        self.conv2 = SAGEConv(hidden_features, hidden_features, aggregator)
        self.conv3 = SAGEConv(hidden_features, out_features, aggregator)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_features)
        self.bn2 = torch.nn.BatchNorm1d(hidden_features)
        self.activation = F.elu_ if activation == 'Elu' else F.relu

    def forward(self, g, x):
        x = self.activation(self.bn1(self.conv1(g, x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.conv2(g, x)))
        x = self.dropout(x)
        x = self.conv3(g, x)
        return x



class OneLayerSGC(nn.Module):
    def __init__(self, in_features, out_features, k, is_out_layer=True):
        super().__init__()
        # Fixme: Deal zero degree
        self.conv1 = SGConv(in_features, out_features, k, cached=True, allow_zero_in_degree=True)
        self.is_out_layer = is_out_layer

    def _stochastic_forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        return x

    def forward(self, g, x, stochastic=False):
        if stochastic:  # Batch forward
            return self._stochastic_forward(g, x)
        else:  # Normal forward
            x = self.conv1(g, x)
            return x



def para_copy(model_to_init, pretrained_model, paras_to_copy):
    # Pass parameters (if exists) of old model to new model
    para_dict_to_update = model_to_init.gnn.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in paras_to_copy}
    para_dict_to_update.update(pretrained_dict)
    model_to_init.gnn.load_state_dict(para_dict_to_update)
